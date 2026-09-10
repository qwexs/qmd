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
    delete process.env.QMD_RERANK_PROVIDER;
    delete process.env.GONKA_RATE_LIMIT_RPM;
    delete process.env.JINA_PROXY_URL;
  });

  afterEach(() => {
    for (const key of Object.keys(process.env)) if (!(key in originalEnv)) delete process.env[key];
    Object.assign(process.env, originalEnv);
  });

  it("requires GONKA_API_KEY", () => {
    delete process.env.GONKA_API_KEY;
    expect(() => new GonkaLLM()).toThrow("GONKA_API_KEY");
  });

  it("reports the configured embedding model for index provenance", () => {
    delete process.env.GONKA_EMBED_MODEL;
    expect(new GonkaLLM().embedModelName).toBe("BAAI/bge-m3");
    process.env.GONKA_EMBED_MODEL = "example/embedding-model";
    expect(new GonkaLLM().embedModelName).toBe("example/embedding-model");
  });

  it("rejects an invalid Gonka request rate limit", () => {
    process.env.GONKA_RATE_LIMIT_RPM = "0";
    expect(() => new GonkaLLM()).toThrow("GONKA_RATE_LIMIT_RPM");
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

  it("evenly spaces Gonka requests when an RPM limit is configured", async () => {
    const originalFetch = globalThis.fetch;
    process.env.GONKA_RATE_LIMIT_RPM = "600";
    const calledAt: number[] = [];
    globalThis.fetch = mockFetch((_url, init) => {
      calledAt.push(Date.now());
      const input = JSON.parse(init.body as string).input as string[];
      return { data: input.map((_text, index) => ({ embedding: [index], index })), model: "BAAI/bge-m3" };
    }) as any;
    try {
      const llm = new GonkaLLM();
      await Promise.all([llm.embed("first"), llm.embed("second")]);
      expect(calledAt).toHaveLength(2);
      expect(calledAt[1]! - calledAt[0]!).toBeGreaterThanOrEqual(95);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("keeps embeddings on Gonka while using Jina's dedicated reranker", async () => {
    process.env.QMD_RERANK_PROVIDER = "jina";
    process.env.JINA_API_KEY = "test-jina-key";
    process.env.JINA_RERANK_MODEL = "jina-reranker-v3.5";
    const originalFetch = globalThis.fetch;
    const requests: { url: string; body: any }[] = [];
    globalThis.fetch = mockFetch((url, init) => {
      const body = JSON.parse(init.body as string);
      requests.push({ url, body });
      if (url === "https://proxy.gonkabroker.com/v1/embeddings") {
        return { data: [{ embedding: [1, 0], index: 0 }], model: "BAAI/bge-m3" };
      }
      expect(url).toBe("https://api.jina.ai/v1/rerank");
      return { results: [{ index: 1, relevance_score: 0.9 }, { index: 0, relevance_score: 0.1 }], model: body.model };
    }) as any;
    try {
      const llm = new GonkaLLM();
      expect((await llm.embed("query"))?.model).toBe("BAAI/bge-m3");
      const result = await llm.rerank("query", [{ file: "weak.md", text: "weak" }, { file: "strong.md", text: "strong" }]);
      expect(result.results.map(r => r.file)).toEqual(["strong.md", "weak.md"]);
      expect(requests).toHaveLength(2);
      expect(requests[1]!.body).toEqual({ model: "jina-reranker-v3.5", query: "query", documents: ["weak", "strong"], top_n: 2 });
      expect(llm.rerankModelName).toBe("jina:jina-reranker-v3.5");
    } finally { globalThis.fetch = originalFetch; }
  });

  it("does not require Jina credentials for embedding-only maintenance", async () => {
    process.env.QMD_RERANK_PROVIDER = "jina";
    delete process.env.JINA_API_KEY;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch(() => ({ data: [{ embedding: [1, 0], index: 0 }], model: "BAAI/bge-m3" })) as any;
    try {
      const llm = new GonkaLLM();
      expect(await llm.embed("query")).not.toBeNull();
      await expect(llm.rerank("query", [{ file: "doc", text: "doc" }])).rejects.toThrow("JINA_API_KEY");
      expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    } finally { globalThis.fetch = originalFetch; }
  });

  it("surfaces Jina errors without falling back to Gonka reranking", async () => {
    process.env.QMD_RERANK_PROVIDER = "jina";
    process.env.JINA_API_KEY = "test-jina-key";
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mock(async (url: string) => {
      expect(url).toBe("https://api.jina.ai/v1/rerank");
      return { ok: false, status: 429, text: async () => "rate limit" };
    }) as any;
    try {
      await expect(new GonkaLLM().rerank("query", [{ file: "doc", text: "doc" }])).rejects.toThrow("Jina API error 429");
      expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    } finally { globalThis.fetch = originalFetch; }
  });

  it("does not amplify a provider 429 into fallback network requests", async () => {
    const originalFetch = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = mock(() => {
      calls++;
      return Promise.resolve({
        ok: false,
        status: 429,
        json: () => Promise.resolve({}),
        text: () => Promise.resolve("rate limit exceeded"),
      });
    }) as any;
    try {
      const llm = new GonkaLLM();
      expect(await llm.embedBatch(["first", "second"])).toEqual([null, null]);
      expect(await llm.embed("third")).toBeNull();
      expect(calls).toBe(1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
