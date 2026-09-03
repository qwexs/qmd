import { afterEach, beforeEach, describe, expect, it, mock } from "bun:test";
import { GonkaLLM } from "./gonka-llm";

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

describe("GonkaLLM", () => {
  beforeEach(() => {
    process.env.GONKA_API_KEY = "test-key";
  });

  afterEach(() => {
    for (const key of Object.keys(process.env)) if (!(key in originalEnv)) delete process.env[key];
    Object.assign(process.env, originalEnv);
  });

  it("requires GONKA_API_KEY", () => {
    delete process.env.GONKA_API_KEY;
    expect(() => new GonkaLLM()).toThrow("GONKA_API_KEY");
  });

  it("uses Gonka's OpenAI-compatible embeddings endpoint and default model", async () => {
    const originalFetch = globalThis.fetch;
    let calledUrl = "";
    let body: Record<string, unknown> = {};
    globalThis.fetch = mockFetch((url, init) => {
      calledUrl = url;
      body = JSON.parse(init.body as string);
      return { data: [{ embedding: [0.1, 0.2], index: 0 }], model: "BAAI/bge-m3" };
    }) as any;
    try {
      const result = await new GonkaLLM().embed("hello");
      expect(calledUrl).toBe("https://proxy.gonkabroker.com/v1/embeddings");
      expect(body).toEqual({ model: "BAAI/bge-m3", input: ["hello"] });
      expect(result).toEqual({ embedding: [0.1, 0.2], model: "BAAI/bge-m3" });
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("reranks with embedding cosine similarity when Gonka has no reranker", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch((_url, init) => {
      const input = JSON.parse(init.body as string).input as string[];
      const vectors: Record<string, number[]> = { query: [1, 0], weak: [0, 1], strong: [1, 0] };
      return { data: input.map((text, index) => ({ embedding: vectors[text]!, index })), model: "BAAI/bge-m3" };
    }) as any;
    try {
      const result = await new GonkaLLM().rerank("query", [
        { file: "weak.md", text: "weak" },
        { file: "strong.md", text: "strong" },
      ]);
      expect(result.results.map((item) => item.file)).toEqual(["strong.md", "weak.md"]);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
