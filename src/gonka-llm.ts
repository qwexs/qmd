/**
 * gonka-llm.ts - Gonka Broker embedding backend for QMD.
 *
 * Gonka Broker exposes BAAI/bge-m3 through an OpenAI-compatible
 * /v1/embeddings endpoint. Gonka currently has no public reranker model, so
 * reranking falls back to cosine similarity over the returned embeddings.
 *
 * Environment variables:
 *   GONKA_API_KEY       - Required. Gonka Broker API key.
 *   GONKA_BASE_URL      - Optional. Defaults to https://proxy.gonkabroker.com/v1.
 *   GONKA_EMBED_MODEL   - Optional. Defaults to BAAI/bge-m3.
 *   GONKA_PROXY_URL     - Optional. HTTP proxy URL.
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
  RerankDocumentResult,
} from "./llm.js";

const DEFAULT_BASE_URL = "https://proxy.gonkabroker.com/v1";
const DEFAULT_EMBED_MODEL = "BAAI/bge-m3";
const MAX_BATCH_SIZE = 100;

type FetchFn = typeof globalThis.fetch;

async function createFetchFn(): Promise<FetchFn> {
  const proxyUrl = process.env.GONKA_PROXY_URL;
  if (!proxyUrl) return globalThis.fetch;

  console.warn("GONKA_PROXY_URL is not supported in this build; using direct fetch");
  return globalThis.fetch;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const left = a[i] ?? 0;
    const right = b[i] ?? 0;
    dot += left * right;
    normA += left * left;
    normB += right * right;
  }
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

export class GonkaLLM implements LLM {
  private apiKey: string;
  private baseUrl: string;
  private embedModel: string;
  private fetchFn: FetchFn | null = null;

  constructor() {
    const apiKey = process.env.GONKA_API_KEY;
    if (!apiKey) throw new Error("GONKA_API_KEY environment variable is required");
    this.apiKey = apiKey;
    this.baseUrl = (process.env.GONKA_BASE_URL || DEFAULT_BASE_URL).replace(/\/$/, "");
    this.embedModel = process.env.GONKA_EMBED_MODEL || DEFAULT_EMBED_MODEL;
  }

  private async getFetch(): Promise<FetchFn> {
    if (!this.fetchFn) this.fetchFn = await createFetchFn();
    return this.fetchFn;
  }

  private async request<T>(endpoint: string, body: Record<string, unknown>): Promise<T> {
    const fetch = await this.getFetch();
    const resp = await fetch(`${this.baseUrl}${endpoint}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`Gonka API error ${resp.status}: ${text}`);
    }
    return resp.json() as Promise<T>;
  }

  async embed(text: string, _options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      return (await this.embedTexts([text]))[0] ?? null;
    } catch (error) {
      console.error("Gonka embedding error:", error);
      return null;
    }
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    const results: (EmbeddingResult | null)[] = new Array(texts.length).fill(null);
    for (let i = 0; i < texts.length; i += MAX_BATCH_SIZE) {
      const batch = texts.slice(i, i + MAX_BATCH_SIZE);
      try {
        const embedded = await this.embedTexts(batch);
        for (let j = 0; j < embedded.length; j++) results[i + j] = embedded[j] ?? null;
      } catch (error) {
        console.warn("Gonka batch embed failed, falling back to individual requests:", error);
        for (let j = 0; j < batch.length; j++) {
          try {
            results[i + j] = (await this.embedTexts([batch[j]!]))[0] ?? null;
          } catch (innerError) {
            console.error(`Gonka embed failed for text ${i + j}:`, innerError);
          }
        }
      }
    }
    return results;
  }

  private async embedTexts(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    type GonkaEmbedResponse = {
      data: Array<{ embedding: number[]; index: number }>;
      model: string;
    };
    const resp = await this.request<GonkaEmbedResponse>("/embeddings", {
      model: this.embedModel,
      input: texts,
    });
    return resp.data
      .sort((a, b) => a.index - b.index)
      .map((item) => ({ embedding: item.embedding, model: resp.model }));
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    _options: RerankOptions = {}
  ): Promise<RerankResult> {
    const embeddings = await this.embedTexts([query, ...documents.map((document) => document.text)]);
    const queryEmbedding = embeddings[0]?.embedding;
    if (!queryEmbedding) throw new Error("Failed to embed query for reranking");

    const results: RerankDocumentResult[] = documents.map((document, index) => ({
      file: document.file,
      score: cosineSimilarity(queryEmbedding, embeddings[index + 1]?.embedding ?? []),
      index,
    }));
    results.sort((a, b) => b.score - a.score);
    return { results, model: this.embedModel };
  }

  async generate(_prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    return null;
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    const results: Queryable[] = [{ type: "vec", text: query }];
    if (options?.includeLexical !== false) results.push({ type: "lex", text: query });
    return results;
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  async tokenize(text: string): Promise<readonly number[]> {
    return Array.from({ length: Math.ceil(text.length / 4) }, (_, index) => index);
  }

  async countTokens(text: string): Promise<number> {
    return Math.ceil(text.length / 4);
  }

  async detokenize(_tokens: readonly number[]): Promise<string> {
    return "";
  }

  async getDeviceInfo(): Promise<{ gpu: false; gpuOffloading: false; gpuDevices: string[]; cpuCores: number }> {
    return { gpu: false, gpuOffloading: false, gpuDevices: [], cpuCores: 0 };
  }

  async dispose(): Promise<void> {}

  async unloadIdleResources(): Promise<void> {}
}
