/**
 * store-concurrency.test.ts - concurrent schema-init safety
 *
 * Reproduces the cross-connection race where two processes opening the same
 * database interleave the FTS trigger DROP+CREATE and collide with "trigger
 * already exists". JavaScript is single-threaded, so the race only appears
 * across OS processes — each case spawns N workers that open the store at once.
 *
 * Fails against the pre-fix DROP+CREATE-on-every-open code; passes once the
 * trigger rebuild is gated by PRAGMA user_version inside an IMMEDIATE
 * transaction.
 */
import { describe, test, expect } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { openDatabase } from "../src/db.ts";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const workerScript = join(thisDir, "_helpers", "store-init-worker.ts");
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

const WORKERS = 12;

type WorkerResult = { code: number | null; stderr: string };

function runWorker(dbPath: string, startAtMs: number): Promise<WorkerResult> {
  const args = isBunRuntime
    ? [workerScript, dbPath, String(startAtMs)]
    : [tsxCli, workerScript, dbPath, String(startAtMs)];
  return new Promise((resolve) => {
    const proc = spawn(process.execPath, args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });
    proc.on("close", (code) => resolve({ code, stderr }));
  });
}

async function openConcurrently(dbPath: string, n: number): Promise<WorkerResult[]> {
  const startAtMs = Date.now() + 1000;
  return Promise.all(Array.from({ length: n }, () => runWorker(dbPath, startAtMs)));
}

function expectAllSucceeded(results: WorkerResult[]): void {
  const failed = results.filter(r => r.code !== 0);
  // On failure the joined worker stderr is surfaced by the assertion below.
  expect(failed.map(r => r.stderr.trim()).join("\n---\n")).toBe("");
  expect(failed).toHaveLength(0);
}

function expectSchemaIntact(dbPath: string): void {
  const db = openDatabase(dbPath);
  try {
    const triggers = db
      .prepare(`SELECT name FROM sqlite_master WHERE type = 'trigger'`)
      .all() as { name: string }[];
    expect(new Set(triggers.map(t => t.name))).toEqual(
      new Set(["documents_ai", "documents_ad", "documents_au"])
    );

    const fts = db
      .prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'documents_fts'`)
      .get();
    expect(fts).toBeTruthy();

    const versionRow = db.prepare(`PRAGMA user_version`).get() as Record<string, number>;
    expect(Object.values(versionRow)[0]).toBeGreaterThanOrEqual(1);
  } finally {
    db.close();
  }
}

describe("concurrent store initialization", () => {
  test("cold database: N processes initialize without colliding on triggers", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-store-concurrency-"));
    const dbPath = join(dir, "index.sqlite");
    try {
      const results = await openConcurrently(dbPath, WORKERS);
      expectAllSucceeded(results);
      expectSchemaIntact(dbPath);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60_000);

  test("existing database: N processes reopen without rebuilding triggers", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-store-concurrency-"));
    const dbPath = join(dir, "index.sqlite");
    try {
      // Stamp the schema once, single-process, so every concurrent reopen takes
      // the version-gated fast path.
      const [seed] = await openConcurrently(dbPath, 1);
      expect(seed.code).toBe(0);

      const results = await openConcurrently(dbPath, WORKERS);
      expectAllSucceeded(results);
      expectSchemaIntact(dbPath);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 60_000);
});
