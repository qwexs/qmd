/**
 * jina-llm.ts - Jina AI API backend for QMD
 *
 * Cloud LLM provider using Jina AI cloud APIs for embeddings and reranking.
 * Designed for environments without local GPU access (CI, containers).
 *
 * Environment variables:
 *   JINA_API_KEY          - Required. Jina AI API key.
 *   JINA_PROXY_URL        - Optional. HTTP proxy URL (uses undici ProxyAgent).
 *   JINA_EMBED_MODEL      - Optional. Embedding model (default: jina-embeddings-v3).
 *   JINA_RERANK_MODEL     - Optional. Rerank model (default: jina-reranker-v2-base-multilingual).
 *   JINA_EMBED_DIMENSIONS - Optional. Embedding dimensions (default: 1024).
 *
 * Implements the v2.5.3 LLM interface plus the additional public surface
 * that store.ts expects on the singleton returned by getDefaultLlamaCpp()
 * (embedModelName / generateModelName / rerankModelName getters, plus
 * tokenize / detokenize / countTokens / embedBatch / getDeviceInfo).
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

// =============================================================================
// Configuration
// =============================================================================

const JINA_API_BASE = "https://api.jina.ai/v1";
const DEFAULT_EMBED_MODEL = "jina-embeddings-v3";
const DEFAULT_RERANK_MODEL = "jina-reranker-v2-base-multilingual";
const DEFAULT_EMBED_DIMENSIONS = 1024;
const MAX_BATCH_SIZE = 100;
const APPROX_CHARS_PER_TOKEN = 4;

// =============================================================================
// Proxy support
// =============================================================================

type FetchInput = Parameters<typeof fetch>[0];
type FetchInit = Parameters<typeof fetch>[1];
type FetchFn = (input: FetchInput, init?: FetchInit) => ReturnType<typeof fetch>;

// undici is an optional runtime dep — only used when JINA_PROXY_URL is set.
// We import it lazily and fall back to the global fetch if it is not installed.
async function createFetchFn(): Promise<FetchFn> {
  const proxyUrl = process.env.JINA_PROXY_URL;
  if (!proxyUrl) return globalThis.fetch;

  try {
    // @ts-expect-error — undici is an optional dependency; only used when proxy URL is set
    const undici = await import("undici");
    const agent = new undici.ProxyAgent(proxyUrl);
    return (input, init) => {
      return undici.fetch(input, { ...(init ?? {}), dispatcher: agent });
    };
  } catch {
    console.warn("undici not available for proxy support, using direct fetch");
    return globalThis.fetch;
  }
}

// =============================================================================
// JinaLLM Implementation
// =============================================================================

export class JinaLLM implements LLM {
  private apiKey: string;
  private embedModelNameValue: string;
  private rerankModelNameValue: string;
  private embedDimensions: number;
  private fetchFn: FetchFn | null = null;

  constructor() {
    const apiKey = process.env.JINA_API_KEY;
    if (!apiKey) {
      throw new Error("JINA_API_KEY environment variable is required");
    }
    this.apiKey = apiKey;
    this.embedModelNameValue = process.env.JINA_EMBED_MODEL || DEFAULT_EMBED_MODEL;
    this.rerankModelNameValue = process.env.JINA_RERANK_MODEL || DEFAULT_RERANK_MODEL;
    this.embedDimensions = parseInt(process.env.JINA_EMBED_DIMENSIONS || "", 10) || DEFAULT_EMBED_DIMENSIONS;
  }

  // Public getters used by store.ts (mirrors LlamaCpp.embedModelName / etc.).
  get embedModelName(): string {
    return this.embedModelNameValue;
  }

  get generateModelName(): string {
    // Jina does not offer a text-generation model; we reuse the rerank model
    // name as a placeholder so callers that log it still get something
    // meaningful. generate() always returns null for Jina.
    return this.rerankModelNameValue;
  }

  get rerankModelName(): string {
    return this.rerankModelNameValue;
  }

  private async getFetch(): Promise<FetchFn> {
    if (!this.fetchFn) {
      this.fetchFn = await createFetchFn();
    }
    return this.fetchFn;
  }

  private async request<T>(endpoint: string, body: Record<string, unknown>): Promise<T> {
    const fetch = await this.getFetch();
    const resp = await fetch(`${JINA_API_BASE}${endpoint}`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`Jina API error ${resp.status}: ${text}`);
    }

    return resp.json() as Promise<T>;
  }

  // ==========================================================================
  // Embeddings
  // ==========================================================================

  async embed(text: string, _options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const results = await this.embedTexts([text]);
      return results[0] ?? null;
    } catch (error) {
      console.error("Jina embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts. Splits into chunks of MAX_BATCH_SIZE.
   * Falls back to individual requests on batch failure.
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const results: (EmbeddingResult | null)[] = new Array(texts.length).fill(null);

    for (let i = 0; i < texts.length; i += MAX_BATCH_SIZE) {
      const batch = texts.slice(i, i + MAX_BATCH_SIZE);
      try {
        const batchResults = await this.embedTexts(batch);
        for (let j = 0; j < batchResults.length; j++) {
          results[i + j] = batchResults[j] ?? null;
        }
      } catch (error) {
        console.warn(`Jina batch embed failed, falling back to individual requests:`, error);
        for (let j = 0; j < batch.length; j++) {
          try {
            const single = await this.embedTexts([batch[j]!]);
            results[i + j] = single[0] ?? null;
          } catch (innerError) {
            console.error(`Jina embed failed for text ${i + j}:`, innerError);
            results[i + j] = null;
          }
        }
      }
    }

    return results;
  }

  private async embedTexts(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    type JinaEmbedResponse = {
      data: Array<{ embedding: number[]; index: number }>;
      model: string;
    };

    const resp = await this.request<JinaEmbedResponse>("/embeddings", {
      model: this.embedModelNameValue,
      task: "text-matching",
      dimensions: this.embedDimensions,
      input: texts,
    });

    const sorted = resp.data.slice().sort((a, b) => a.index - b.index);
    return sorted.map((item) => ({
      embedding: item.embedding,
      model: resp.model,
    }));
  }

  // ==========================================================================
  // Reranking
  // ==========================================================================

  async rerank(
    query: string,
    documents: RerankDocument[],
    _options: RerankOptions = {}
  ): Promise<RerankResult> {
    type JinaRerankResponse = {
      results: Array<{ index: number; relevance_score: number }>;
      model: string;
    };

    const docTexts = documents.map((d) => d.text);

    const resp = await this.request<JinaRerankResponse>("/rerank", {
      model: this.rerankModelNameValue,
      query,
      documents: docTexts,
      top_n: documents.length,
    });

    const results: RerankDocumentResult[] = resp.results.map((r) => ({
      file: documents[r.index]!.file,
      score: r.relevance_score,
      index: r.index,
    }));

    results.sort((a, b) => b.score - a.score);

    return {
      results,
      model: resp.model,
    };
  }

  // ==========================================================================
  // Generation (not supported by Jina)
  // ==========================================================================

  async generate(_prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    return null;
  }

  // ==========================================================================
  // Query Expansion (fallback without LLM)
  // ==========================================================================

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    const results: Queryable[] = [
      { type: "vec", text: query },
      { type: "lex", text: query },
    ];
    if (options?.includeLexical !== false) {
      const lower = query.toLowerCase();
      if (lower !== query) {
        results.push({ type: "lex", text: lower });
      }
    }
    return results;
  }

  // ==========================================================================
  // Model info
  // ==========================================================================

  async modelExists(_model: string): Promise<ModelInfo> {
    return { name: this.embedModelNameValue, exists: true };
  }

  // ==========================================================================
  // Tokenization (approximate, ~4 chars per token)
  // ==========================================================================
  //
  // Cloud providers do not expose their tokenizers. The store uses
  // tokenize/detokenize for input-size truncation before embed() and
  // embedBatch(); approximating at 4 chars per token is good enough for
  // the safety bounds used by store.ts.

  async tokenize(text: string): Promise<readonly number[]> {
    const count = Math.ceil(text.length / APPROX_CHARS_PER_TOKEN);
    return Array.from({ length: count }, (_, i) => i);
  }

  async countTokens(text: string): Promise<number> {
    return Math.ceil(text.length / APPROX_CHARS_PER_TOKEN);
  }

  async detokenize(_tokens: readonly number[]): Promise<string> {
    return "";
  }

  // ==========================================================================
  // Device info (stub for qmd doctor)
  // ==========================================================================

  async getDeviceInfo(_options?: { allowBuild?: boolean }): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram: { total: number; free: number } | null;
    cpuCores: number;
    description: string;
  }> {
    return {
      gpu: false,
      gpuOffloading: false,
      gpuDevices: [],
      vram: null,
      cpuCores: 0,
      description: `Jina AI cloud provider (${JINA_API_BASE}, dim=${this.embedDimensions})`,
    };
  }

  // ==========================================================================
  // Lifecycle
  // ==========================================================================

  async dispose(): Promise<void> {
    // No local resources to clean up
  }

  /**
   * No-op for API-based provider. Compatible with LlamaCpp idle resource management.
   */
  async unloadIdleResources(): Promise<void> {
    // Nothing to unload
  }
}
