import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test";
import { OpenAILLM } from "./openai-llm.js";

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

describe("OpenAILLM", () => {
  beforeEach(() => {
    process.env.OPENAI_API_KEY = "test-key-123";
  });

  afterEach(() => {
    for (const key of Object.keys(process.env)) {
      if (!(key in originalEnv)) delete process.env[key];
    }
    Object.assign(process.env, originalEnv);
  });

  it("throws without OPENAI_API_KEY", () => {
    delete process.env.OPENAI_API_KEY;
    expect(() => new OpenAILLM()).toThrow("OPENAI_API_KEY");
  });

  it("embed() returns correct structure", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch((_url, _init) => ({
      data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
      model: "text-embedding-3-small",
      usage: { total_tokens: 3 },
    })) as any;

    try {
      const llm = new OpenAILLM();
      const result = await llm.embed("hello world");
      expect(result).not.toBeNull();
      expect(result!.embedding).toEqual([0.1, 0.2, 0.3]);
      expect(result!.model).toBe("text-embedding-3-small");
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
        model: "text-embedding-3-small",
        usage: { total_tokens: body.input.length },
      };
    }) as any;

    try {
      const llm = new OpenAILLM();
      const texts = Array.from({ length: 5 }, (_, i) => `text ${i}`);
      const results = await llm.embedBatch(texts);
      expect(results.length).toBe(5);
      expect(callCount).toBe(1); // All in one batch
      results.forEach((r) => expect(r).not.toBeNull());
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("rerank() returns sorted results via cosine similarity", async () => {
    const originalFetch = globalThis.fetch;
    // Query embedding = [1, 0, 0]
    // Doc A = [0.1, 0.9, 0] -> low similarity
    // Doc B = [0.9, 0.1, 0] -> high similarity
    // Doc C = [0.5, 0.5, 0] -> medium similarity
    globalThis.fetch = mockFetch((_url, init) => {
      const body = JSON.parse((init as any).body);
      const embeddings: Record<string, number[]> = {
        "query": [1, 0, 0],
        "doc a": [0.1, 0.9, 0],
        "doc b": [0.9, 0.1, 0],
        "doc c": [0.5, 0.5, 0],
      };
      return {
        data: body.input.map((t: string, i: number) => ({
          embedding: embeddings[t] || [0, 0, 0],
          index: i,
        })),
        model: "text-embedding-3-small",
        usage: { total_tokens: body.input.length },
      };
    }) as any;

    try {
      const llm = new OpenAILLM();
      const result = await llm.rerank("query", [
        { file: "a.md", text: "doc a" },
        { file: "b.md", text: "doc b" },
        { file: "c.md", text: "doc c" },
      ]);
      expect(result.results[0].file).toBe("b.md"); // highest similarity
      expect(result.results[1].file).toBe("c.md"); // medium
      expect(result.results[2].file).toBe("a.md"); // lowest
      expect(result.results[0].score).toBeGreaterThan(result.results[1].score);
      expect(result.results[1].score).toBeGreaterThan(result.results[2].score);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("expandQuery() parses generate response", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch((url, init) => {
      if (url.includes("/chat/completions")) {
        return {
          choices: [{ message: { content: "lex: test keywords\nvec: semantic test\nhyde: a document about test" } }],
          model: "gpt-4o-mini",
        };
      }
      return {};
    }) as any;

    try {
      const llm = new OpenAILLM();
      const results = await llm.expandQuery("test");
      expect(results.length).toBe(3);
      expect(results.some((r) => r.type === "lex")).toBe(true);
      expect(results.some((r) => r.type === "vec")).toBe(true);
      expect(results.some((r) => r.type === "hyde")).toBe(true);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("generate() returns text", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch(() => ({
      choices: [{ message: { content: "Generated response" } }],
      model: "gpt-4o-mini",
    })) as any;

    try {
      const llm = new OpenAILLM();
      const result = await llm.generate("test prompt");
      expect(result).not.toBeNull();
      expect(result!.text).toBe("Generated response");
      expect(result!.done).toBe(true);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("OPENAI_BASE_URL is used when set", async () => {
    process.env.OPENAI_BASE_URL = "https://custom.api.com/v1";
    const originalFetch = globalThis.fetch;
    let calledUrl = "";
    globalThis.fetch = mockFetch((url) => {
      calledUrl = url;
      return {
        data: [{ embedding: [0.1], index: 0 }],
        model: "custom-model",
        usage: { total_tokens: 1 },
      };
    }) as any;

    try {
      const llm = new OpenAILLM();
      await llm.embed("test");
      expect(calledUrl).toStartWith("https://custom.api.com/v1");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("modelExists() always returns exists: true", async () => {
    const llm = new OpenAILLM();
    const info = await llm.modelExists("any-model");
    expect(info.exists).toBe(true);
  });

  it("countTokens() approximates ~4 chars per token", async () => {
    const llm = new OpenAILLM();
    const count = await llm.countTokens("hello world!"); // 12 chars
    expect(count).toBe(3); // ceil(12/4)
  });
});
