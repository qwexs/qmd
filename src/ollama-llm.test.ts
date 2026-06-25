import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test";
import { OllamaLLM } from "./ollama-llm.js";

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

describe("OllamaLLM", () => {
  beforeEach(() => {
    process.env.OLLAMA_API_KEY = "test-key-123";
    delete process.env.OLLAMA_BASE_URL;
    delete process.env.OLLAMA_EMBED_MODEL;
    delete process.env.OLLAMA_EMBED_DIMENSIONS;
    delete process.env.OLLAMA_PROXY_URL;
  });

  afterEach(() => {
    // Restore env
    for (const key of Object.keys(process.env)) {
      if (!(key in originalEnv)) delete process.env[key];
    }
    Object.assign(process.env, originalEnv);
  });

  it("throws without OLLAMA_API_KEY", () => {
    delete process.env.OLLAMA_API_KEY;
    expect(() => new OllamaLLM()).toThrow("OLLAMA_API_KEY");
  });

  it("uses default base URL https://ollama.com and default embed model nomic-embed-text", async () => {
    const originalFetch = globalThis.fetch;
    let calledUrl = "";
    let calledBody = "";
    globalThis.fetch = mockFetch((url, init) => {
      calledUrl = url;
      calledBody = String((init as any)?.body ?? "");
      return { model: "nomic-embed-text", embeddings: [[0.1, 0.2, 0.3]] };
    }) as any;

    try {
      const llm = new OllamaLLM();
      await llm.embed("hello");
      expect(calledUrl.startsWith("https://ollama.com/api/embed")).toBe(true);
      const parsed = JSON.parse(calledBody);
      expect(parsed.model).toBe("nomic-embed-text");
      expect(parsed.input).toEqual(["hello"]);
      expect(parsed.truncate).toBe(true);
      // No OLLAMA_EMBED_DIMENSIONS → omit
      expect("dimensions" in parsed).toBe(false);
      void llm;
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("honours OLLAMA_BASE_URL and OLLAMA_EMBED_MODEL", async () => {
    process.env.OLLAMA_BASE_URL = "http://localhost:11434";
    process.env.OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b";

    const originalFetch = globalThis.fetch;
    let calledUrl = "";
    globalThis.fetch = mockFetch((url) => {
      calledUrl = url;
      return { model: "qwen3-embedding:0.6b", embeddings: [[0.5]] };
    }) as any;

    try {
      const llm = new OllamaLLM();
      const result = await llm.embed("hi");
      expect(result).not.toBeNull();
      expect(result!.model).toBe("qwen3-embedding:0.6b");
      expect(calledUrl.startsWith("http://localhost:11434/api/embed")).toBe(true);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("passes dimensions when OLLAMA_EMBED_DIMENSIONS is set", async () => {
    process.env.OLLAMA_EMBED_DIMENSIONS = "256";

    const originalFetch = globalThis.fetch;
    let calledBody = "";
    globalThis.fetch = mockFetch((_url, init) => {
      calledBody = String((init as any).body);
      return { model: "nomic-embed-text", embeddings: [Array(256).fill(0.01)] };
    }) as any;

    try {
      const llm = new OllamaLLM();
      await llm.embed("x");
      const parsed = JSON.parse(calledBody);
      expect(parsed.dimensions).toBe(256);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("rejects non-positive OLLAMA_EMBED_DIMENSIONS", () => {
    process.env.OLLAMA_EMBED_DIMENSIONS = "0";
    expect(() => new OllamaLLM()).toThrow("OLLAMA_EMBED_DIMENSIONS");
  });

  it("embed() returns correct structure", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch(() => ({
      model: "nomic-embed-text",
      embeddings: [[0.1, 0.2, 0.3]],
    })) as any;

    try {
      const llm = new OllamaLLM();
      const result = await llm.embed("hello world");
      expect(result).not.toBeNull();
      expect(result!.embedding).toEqual([0.1, 0.2, 0.3]);
      expect(result!.model).toBe("nomic-embed-text");
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
        model: "nomic-embed-text",
        embeddings: body.input.map((_t: string, i: number) => [i * 0.1]),
      };
    }) as any;

    try {
      const llm = new OllamaLLM();
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
        query: [1, 0, 0],
        "doc a": [0.1, 0.9, 0],
        "doc b": [0.9, 0.1, 0],
        "doc c": [0.5, 0.5, 0],
      };
      return {
        model: "nomic-embed-text",
        embeddings: body.input.map((t: string) => embeddings[t] ?? [0, 0, 0]),
      };
    }) as any;

    try {
      const llm = new OllamaLLM();
      const result = await llm.rerank("query", [
        { file: "a.md", text: "doc a" },
        { file: "b.md", text: "doc b" },
        { file: "c.md", text: "doc c" },
      ]);
      expect(result.results[0]!.file).toBe("b.md");
      expect(result.results[1]!.file).toBe("c.md");
      expect(result.results[2]!.file).toBe("a.md");
      expect(result.results[0]!.score).toBeGreaterThan(result.results[1]!.score);
      expect(result.results[1]!.score).toBeGreaterThan(result.results[2]!.score);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("expandQuery() returns fallback queries without LLM", async () => {
    const llm = new OllamaLLM();
    const results = await llm.expandQuery("Hello World");
    expect(results.length).toBeGreaterThanOrEqual(2);
    expect(results.some((r) => r.type === "vec")).toBe(true);
    expect(results.some((r) => r.type === "lex")).toBe(true);
    // Should include lowercase variant since "Hello World" !== "hello world"
    expect(results.some((r) => r.text === "hello world")).toBe(true);
  });

  it("expandQuery() omits lowercase variant when includeLexical: false", async () => {
    const llm = new OllamaLLM();
    const results = await llm.expandQuery("Hello World", { includeLexical: false });
    // Consistent with JinaLLM: base lex:query stays, but the extra lowercase variant is omitted.
    expect(results.some((r) => r.type === "lex" && r.text === "Hello World")).toBe(true);
    expect(results.some((r) => r.type === "lex" && r.text === "hello world")).toBe(false);
  });

  it("generate() returns null (search-only provider)", async () => {
    const llm = new OllamaLLM();
    const result = await llm.generate("test prompt");
    expect(result).toBeNull();
  });

  it("modelExists() returns true when model is in /api/tags", async () => {
    const originalFetch = globalThis.fetch;
    let calledUrl = "";
    globalThis.fetch = mockFetch((url) => {
      calledUrl = url;
      if (url.endsWith("/api/tags")) {
        return { models: [{ name: "nomic-embed-text:latest" }, { name: "qwen3:8b" }] };
      }
      return { model: "nomic-embed-text", embeddings: [[0.1]] };
    }) as any;

    try {
      const llm = new OllamaLLM();
      // Match by base name (no tag) against a model present with a tag
      const info = await llm.modelExists("nomic-embed-text");
      expect(info.exists).toBe(true);
      expect(calledUrl.endsWith("/api/tags")).toBe(true);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("modelExists() returns false for unknown model", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = mockFetch(() => ({ models: [{ name: "nomic-embed-text:latest" }] })) as any;

    try {
      const llm = new OllamaLLM();
      const info = await llm.modelExists("nonexistent-model");
      expect(info.exists).toBe(false);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("modelExists() caches /api/tags responses", async () => {
    const originalFetch = globalThis.fetch;
    let tagsCalls = 0;
    globalThis.fetch = mockFetch((url) => {
      if (url.endsWith("/api/tags")) {
        tagsCalls++;
        return { models: [{ name: "nomic-embed-text:latest" }] };
      }
      return { model: "nomic-embed-text", embeddings: [[0.1]] };
    }) as any;

    try {
      const llm = new OllamaLLM();
      await llm.modelExists("nomic-embed-text");
      await llm.modelExists("nomic-embed-text");
      await llm.modelExists("nomic-embed-text");
      expect(tagsCalls).toBe(1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("countTokens() approximates ~4 chars per token", async () => {
    const llm = new OllamaLLM();
    const count = await llm.countTokens("hello world!"); // 12 chars
    expect(count).toBe(3); // ceil(12/4)
  });

  it("getDeviceInfo() describes the configured host", async () => {
    process.env.OLLAMA_BASE_URL = "http://localhost:11434";
    const llm = new OllamaLLM();
    const info = await llm.getDeviceInfo();
    expect(info.gpu).toBe(false);
    expect(info.description).toContain("http://localhost:11434");
    expect(info.description).toContain("nomic-embed-text");
  });

  it("strip trailing slash from OLLAMA_BASE_URL", async () => {
    process.env.OLLAMA_BASE_URL = "https://ollama.com/";

    const originalFetch = globalThis.fetch;
    let calledUrl = "";
    globalThis.fetch = mockFetch((url) => {
      calledUrl = url;
      return { model: "nomic-embed-text", embeddings: [[0.1]] };
    }) as any;

    try {
      const llm = new OllamaLLM();
      await llm.embed("test");
      // Should not produce double slashes before /api/embed
      expect(calledUrl).toBe("https://ollama.com/api/embed");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("Authorization header carries the API key", async () => {
    const originalFetch = globalThis.fetch;
    let sentAuth = "";
    globalThis.fetch = mockFetch((_url, init) => {
      sentAuth = String((init as any)?.headers?.Authorization ?? "");
      return { model: "nomic-embed-text", embeddings: [[0.1]] };
    }) as any;

    try {
      const llm = new OllamaLLM();
      await llm.embed("x");
      expect(sentAuth).toBe("Bearer test-key-123");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
