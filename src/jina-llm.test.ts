import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test";
import { JinaLLM } from "./jina-llm";

// Save and restore env
const originalEnv = { ...process.env };

function mockFetch(handler: (url: string, init: RequestInit) => unknown) {
  return mock((url: string | URL | Request, init?: RequestInit) => {
    const urlStr = typeof url === "string" ? url : url instanceof URL ? url.toString() : url.url;
    const body = handler(urlStr, init || {});
    return Promise.resolve({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
      text: () => Promise.resolve(JSON.stringify(body)),
    });
  });
}

describe("JinaLLM", () => {
  beforeEach(() => {
    process.env.JINA_API_KEY = "test-key-123";
  });

  afterEach(() => {
    // Restore env
    for (const key of Object.keys(process.env)) {
      if (!(key in originalEnv)) delete process.env[key];
    }
    Object.assign(process.env, originalEnv);
  });

  it("throws without JINA_API_KEY", () => {
    delete process.env.JINA_API_KEY;
    expect(() => new JinaLLM()).toThrow("JINA_API_KEY");
  });

  it("embed() returns correct structure", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch((_url, _init) => ({
      data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
      model: "jina-embeddings-v3",
    })) as any;

    try {
      const llm = new JinaLLM();
      const result = await llm.embed("hello world");
      expect(result).not.toBeNull();
      expect(result!.embedding).toEqual([0.1, 0.2, 0.3]);
      expect(result!.model).toBe("jina-embeddings-v3");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("embedBatch() batches correctly", async () => {
    const originalFetch = globalThis.fetch;
    let callCount = 0;
    globalThis.fetch = mockFetch((_url, init) => {
      callCount++;
      const body = JSON.parse((init as any).body);
      return {
        data: body.input.map((t: string, i: number) => ({
          embedding: [i * 0.1],
          index: i,
        })),
        model: "jina-embeddings-v3",
      };
    }) as any;

    try {
      const llm = new JinaLLM();
      const texts = Array.from({ length: 5 }, (_, i) => `text ${i}`);
      const results = await llm.embedBatch(texts);
      expect(results.length).toBe(5);
      expect(callCount).toBe(1); // All in one batch
      results.forEach((r) => expect(r).not.toBeNull());
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("rerank() returns sorted results", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch(() => ({
      results: [
        { index: 1, relevance_score: 0.95 },
        { index: 0, relevance_score: 0.5 },
        { index: 2, relevance_score: 0.8 },
      ],
      model: "jina-reranker-v2-base-multilingual",
    })) as any;

    try {
      const llm = new JinaLLM();
      const result = await llm.rerank("query", [
        { file: "a.md", text: "doc a" },
        { file: "b.md", text: "doc b" },
        { file: "c.md", text: "doc c" },
      ]);
      expect(result.results[0].file).toBe("b.md");
      expect(result.results[0].score).toBe(0.95);
      expect(result.results[1].file).toBe("c.md");
      expect(result.results[2].file).toBe("a.md");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("expandQuery() returns fallback queries", async () => {
    const llm = new JinaLLM();
    const results = await llm.expandQuery("Hello World");
    expect(results.length).toBeGreaterThanOrEqual(2);
    expect(results.some((r) => r.type === "vec")).toBe(true);
    expect(results.some((r) => r.type === "lex")).toBe(true);
    // Should include lowercase variant since "Hello World" !== "hello world"
    expect(results.some((r) => r.text === "hello world")).toBe(true);
  });

  it("generate() returns null", async () => {
    const llm = new JinaLLM();
    const result = await llm.generate("test prompt");
    expect(result).toBeNull();
  });

  it("modelExists() always returns exists: true", async () => {
    const llm = new JinaLLM();
    const info = await llm.modelExists("any-model");
    expect(info.exists).toBe(true);
  });

  it("countTokens() approximates ~4 chars per token", async () => {
    const llm = new JinaLLM();
    const count = await llm.countTokens("hello world!"); // 12 chars
    expect(count).toBe(3); // ceil(12/4)
  });
});
