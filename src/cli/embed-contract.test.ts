import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { createStore, getHashesNeedingEmbedding, clearAllEmbeddings } from '../store.ts';

let root: string;
let server: ReturnType<typeof Bun.serve>;
let requests: string[][];
let failAll: boolean;
let failAfterFirst: boolean;
let env: Record<string, string | undefined>;
const cli = resolve(import.meta.dir, 'qmd.ts');
const model = 'BAAI/bge-m3';

async function run(...args: string[]) {
  const child = Bun.spawn([process.execPath, cli, ...args], {cwd:root, env, stdout:'pipe', stderr:'pipe'});
  const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  return {stdout, stderr, code};
}

beforeEach(async () => {
  root = mkdtempSync(join(tmpdir(), 'qmd-embed-contract-'));
  requests = []; failAll = false; failAfterFirst = false;
  server = Bun.serve({hostname:'127.0.0.1', port:0, async fetch(request) {
    const body = await request.json() as {input:string[];model:string};
    requests.push(body.input);
    if (failAll || (failAfterFirst && requests.length > 1)) return new Response('synthetic failure', {status:500});
    return Response.json({model:body.model, data:body.input.map((_, index) => ({index, embedding:[0.1,0.2,0.3]}))});
  }});
  const config = join(root, 'config'); mkdirSync(config);
  const collections: Record<string, unknown> = {};
  for (const name of ['alpha','beta','outside']) {
    const path = join(root,name); mkdirSync(path);
    writeFileSync(join(path,'shared.md'), '# Shared\nSame synthetic content.');
    writeFileSync(join(path,`${name}.md`), `# ${name}\nUnique synthetic content for ${name}.`);
    collections[name] = {path, pattern:'**/*.md'};
  }
  writeFileSync(join(config,'index.yml'), JSON.stringify({collections}));
  env = {...process.env, INDEX_PATH:join(root,'index.sqlite'), QMD_CONFIG_DIR:config, PWD:root,
    QMD_LLM_PROVIDER:'gonka', GONKA_API_KEY:'test-key', GONKA_EMBED_MODEL:model,
    GONKA_BASE_URL:`http://127.0.0.1:${server.port}/v1`, GONKA_RATE_LIMIT_RPM:'', GONKA_PROXY_URL:'',
    QMD_RERANK_PROVIDER:'gonka'};
  expect((await run('update')).code).toBe(0);
});
afterEach(() => {server.stop(true); rmSync(root,{recursive:true,force:true});});

describe('structured embed contract', () => {
  test('embeds every selected collection, deduplicates hashes, and leaves the outside scope alone', async () => {
    const first = await run('embed','--format','json','-c','alpha');
    expect(first.code).toBe(0);
    expect(JSON.parse(first.stdout)).toMatchObject({schema:'qmd.embed.v1',status:'ok',model,pendingBefore:2,pendingAfter:0,docsProcessed:2});
    const second = await run('embed','--format','json','-c','alpha','-c','beta','-c','alpha');
    expect(second.code).toBe(0);
    expect(JSON.parse(second.stdout)).toMatchObject({status:'ok',pendingBefore:1,pendingAfter:0,docsProcessed:1});
    const store = createStore(env.INDEX_PATH);
    expect(store.db.prepare('SELECT DISTINCT model FROM content_vectors').all()).toEqual([{model}]);
    expect(store.db.prepare('SELECT count(*) AS n FROM content_vectors').get()).toEqual({n:3});
    // Read scope counting under the same provider formatting contract as the child.
    const previous = process.env.QMD_LLM_PROVIDER; process.env.QMD_LLM_PROVIDER='gonka';
    try {
      expect(getHashesNeedingEmbedding(store.db, ['alpha','beta'],model)).toBe(0);
      expect(getHashesNeedingEmbedding(store.db, ['outside'],model)).toBe(1);
      expect(getHashesNeedingEmbedding(store.db, [],model)).toBe(0);
    } finally {if(previous===undefined) delete process.env.QMD_LLM_PROVIDER; else process.env.QMD_LLM_PROVIDER=previous;}
    clearAllEmbeddings(store.db, ['alpha','beta']);
    // Shared content is retained because an unselected collection references it.
    expect(store.db.prepare('SELECT count(*) AS n FROM content_vectors').get()).toEqual({n:1});
    store.close();
  }, 30000);

  test('no-op returns parseable JSON and makes no provider requests', async () => {
    await run('embed','--format','json','-c','alpha'); requests=[];
    const result=await run('embed','--format','json','-c','alpha');
    expect(result.code).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({status:'ok',pendingAfter:0,skippedReason:'no-pending-documents'});
    expect(requests).toHaveLength(0);
  },30000);

  test('rejects an unknown collection before embedding and emits one error object', async () => {
    const result=await run('embed','--format','json','-c','alpha','-c','missing');
    expect(result.code).toBe(1);
    expect(JSON.parse(result.stdout)).toMatchObject({schema:'qmd.embed.v1',status:'error'});
    expect(requests).toHaveLength(0);
  });

  test('provider failure cannot be reported as a successful embed', async () => {
    failAll=true;
    const result=await run('embed','--format','json','-c','beta');
    expect(result.code).toBe(1);
    expect(JSON.parse(result.stdout)).toMatchObject({status:'error'});
  });

  test('partial failures return nonzero and retain pending hashes', async () => {
    failAfterFirst=true;
    const result=await run('embed','--format','json','-c','beta');
    expect(result.code).toBe(1);
    const data=JSON.parse(result.stdout);
    expect(data.status).toBe('partial'); expect(data.pendingAfter).toBe(2); expect(data.errors).toBeGreaterThan(0);
  },30000);

  test('advertises the coordinator capabilities', async () => {
    const result=await run('capabilities','--format','json');
    expect(result.code).toBe(0);
    expect(JSON.parse(result.stdout)).toMatchObject({schema:'qmd.capabilities.v1',embed:{multipleCollections:true,indexScopedLock:true,structuredOutput:true}});
  });

  test('model diagnostics do not require an API key', async () => {
    delete env.GONKA_API_KEY;
    const result=await run('status');
    expect(result.code).toBe(0);
    expect(result.stdout).toContain(model);
  });

  test('a live index lock returns deferred JSON instead of success', async () => {
    const helper=resolve(import.meta.dir,'../../test/_helpers/embed-lock-holder.ts');
    const child=Bun.spawn([process.execPath,helper,env.INDEX_PATH+'.embed.lock','10000','qmd','embed'],{stdout:'pipe',stderr:'pipe'});
    try {
      const reader=child.stdout.getReader();
      const first=await reader.read();
      expect(new TextDecoder().decode(first.value)).toContain('HOLD');
      reader.releaseLock();
      const result=await run('embed','--format','json','-c','alpha');
      expect(JSON.parse(result.stdout)).toMatchObject({status:'deferred',skippedReason:'lock-busy'});
      expect(requests).toHaveLength(0);
    } finally {child.kill(); await child.exited;}
  },15000);
});
