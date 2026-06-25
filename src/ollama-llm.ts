/**
 * ollama-llm.ts - Ollama Cloud API backend for QMD
 *
 * Cloud LLM provider using the Ollama Cloud REST API (https://ollama.com) for
 * embeddings and reranking. Designed for environments without local GPU
 * access (CI, containers) and as a low-cost alternative to Jina/OpenAI for
 * embedding-based search.
 *
 * This provider is **search-only** — it does not perform text generation or
 * query expansion (matching JinaLLM, which has the same constraint). Reranking
 * is implemented as cosine similarity over embeddings, the same approach used by
 * OpenAILLM, because Ollama does not expose a native `/api/rerank` endpoint.
 *
 * Environment variables:
 *   OLLAMA_API_KEY          - Required. Ollama API key from ollama.com/settings/keys.
 *   OLLAMA_BASE_URL         - Optional. Base URL (default: https://ollama.com).
 *                             Override with http://localhost:11434 for self-hosted Ollama.
 *   OLLAMA_EMBED_MODEL      - Optional. Embedding model (default: nomic-embed-text).
 *   OLLAMA_EMBED_DIMENSIONS - Optional. Embedding dimensions (default: model default).
 *   OLLAMA_PROXY_URL        - Optional. HTTP proxy URL (uses undici ProxyAgent).
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

const DEFAULT_BASE_URL = "https://ollama.com";
const DEFAULT_EMBED_MODEL = "nomic-embed-text";
const MAX_BATCH_SIZE = 100;
const APPROX_CHARS_PER_TOKEN = 4;

// =============================================================================
// Proxy support
// =============================================================================

type FetchInput = Parameters<typeof fetch>[0];
type FetchInit = Parameters<typeof fetch>[1];
type FetchFn = (input: FetchInput, init?: FetchInit) => ReturnType<typeof fetch>;

// undici is an optional runtime dep — only used when OLLAMA_PROXY_URL is set.
// We import it lazily and fall back to the global fetch if it is not installed.
async function createFetchFn(): Promise<FetchFn> {
  const proxyUrl = process.env.OLLAMA_PROXY_URL;
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
// Cosine Similarity
// =============================================================================

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// =============================================================================
// OllamaLLM Implementation
// =============================================================================

export class OllamaLLM implements LLM {
  private apiKey: string;
  private baseUrl: string;
  private embedModelNameValue: string;
  private embedDimensions: number | null;
  private fetchFn: FetchFn | null = null;
  // Cache the /api/tags response so modelExists() does not hit the network on every call.
  private tagsCache: { names: Set<string>; fetchedAt: number } | null = null;
  private static readonly TAGS_CACHE_TTL_MS = 60_000;

  constructor() {
    const apiKey = process.env.OLLAMA_API_KEY;
    if (!apiKey) {
      throw new Error("OLLAMA_API_KEY environment variable is required");
    }
    this.apiKey = apiKey;
    this.baseUrl = (process.env.OLLAMA_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.embedModelNameValue = process.env.OLLAMA_EMBED_MODEL || DEFAULT_EMBED_MODEL;
    const dim = process.env.OLLAMA_EMBED_DIMENSIONS;
    this.embedDimensions = dim ? parseInt(dim, 10) : null;
    if (this.embedDimensions !== null && (!Number.isFinite(this.embedDimensions) || this.embedDimensions <= 0)) {
      throw new Error(`Invalid OLLAMA_EMBED_DIMENSIONS="${dim}", must be a positive integer`);
    }
  }

  // Public getters used by store.ts (mirrors LlamaCpp.embedModelName / etc.).
  get embedModelName(): string {
    return this.embedModelNameValue;
  }

  get generateModelName(): string {
    // Ollama provider does not perform text generation; reuse the embed model
    // name as a placeholder so callers that log it still get something
    // meaningful. generate() always returns null for OllamaLLM.
    return this.embedModelNameValue;
  }

  get rerankModelName(): string {
    // Reranking is implemented as cosine similarity over embeddings.
    return this.embedModelNameValue;
  }

  private async getFetch(): Promise<FetchFn> {
    if (!this.fetchFn) {
      this.fetchFn = await createFetchFn();
    }
    return this.fetchFn;
  }

  private async request<T>(endpoint: string, body: Record<string, unknown>, method: "GET" | "POST" = "POST"): Promise<T> {
    const fetch = await this.getFetch();
    const init: RequestInit = {
      method,
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
    };
    if (method === "POST") {
      init.body = JSON.stringify(body);
    }
    const resp = await fetch(`${this.baseUrl}${endpoint}`, init);

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`Ollama API error ${resp.status} ${resp.statusText}: ${text}`);
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
      console.error("Ollama embedding error:", error);
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
        console.warn(`Ollama batch embed failed, falling back to individual requests:`, error);
        for (let j = 0; j < batch.length; j++) {
          try {
            const single = await this.embedTexts([batch[j]!]);
            results[i + j] = single[0] ?? null;
          } catch (innerError) {
            console.error(`Ollama embed failed for text ${i + j}:`, innerError);
            results[i + j] = null;
          }
        }
      }
    }

    return results;
  }

  private async embedTexts(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    type OllamaEmbedResponse = {
      model: string;
      embeddings: number[][];
    };

    const body: Record<string, unknown> = {
      model: this.embedModelNameValue,
      input: texts,
    };
    if (this.embedDimensions !== null) {
      body.dimensions = this.embedDimensions;
    }
    // truncate: true — silently drop inputs that exceed the model's context window
    body.truncate = true;

    const resp = await this.request<OllamaEmbedResponse>("/api/embed", body);
    if (!Array.isArray(resp.embeddings)) {
      throw new Error("Ollama /api/embed response missing 'embeddings' array");
    }
    return resp.embeddings.map((embedding) => ({
      embedding,
      model: resp.model,
    }));
  }

  // ==========================================================================
  // Reranking (via cosine similarity over embeddings)
  // ==========================================================================

  async rerank(
    query: string,
    documents: RerankDocument[],
    _options: RerankOptions = {}
  ): Promise<RerankResult> {
    const allTexts = [query, ...documents.map((d) => d.text)];
    const embeddings = await this.embedTexts(allTexts);

    const queryEmbedding = embeddings[0]?.embedding;
    if (!queryEmbedding) {
      throw new Error("Failed to embed query for reranking");
    }

    const results: RerankDocumentResult[] = documents.map((doc, index) => {
      const docEmbedding = embeddings[index + 1]?.embedding;
      const score = docEmbedding ? cosineSimilarity(queryEmbedding, docEmbedding) : 0;
      return {
        file: doc.file,
        score,
        index,
      };
    });

    results.sort((a, b) => b.score - a.score);

    return {
      results,
      model: this.embedModelNameValue,
    };
  }

  // ==========================================================================
  // Generation (not supported — search-only provider)
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
  // Model info — probes /api/tags with caching
  // ==========================================================================

  /**
   * Check whether a model is available on the configured Ollama host.
   * Uses GET /api/tags and matches the requested name against the listed
   * models. The response is cached for TAGS_CACHE_TTL_MS to avoid hammering
   * the API on every embed/rerank call.
   *
   * Matching rules:
   *  - Exact match on `name`.
   *  - `name:tag` form (`nomic-embed-text:latest`) is matched when the
   *    request omits a tag and the host returns the same base with any tag.
   */
  async modelExists(modelName: string): Promise<ModelInfo> {
    try {
      const available = await this.getAvailableModelNames();
      if (available.has(modelName)) {
        return { name: modelName, exists: true };
      }
      // Try base name match (strip tag suffix if present)
      const base = modelName.split(":")[0]!;
      for (const candidate of available) {
        if (candidate.startsWith(`${base}:`)) {
          return { name: modelName, exists: true };
        }
      }
      return { name: modelName, exists: false };
    } catch (error) {
      // Surface as not-found rather than throwing — store.ts treats false as "skip embed"
      console.error("Ollama /api/tags probe failed:", error);
      return { name: modelName, exists: false };
    }
  }

  private async getAvailableModelNames(): Promise<Set<string>> {
    const now = Date.now();
    if (this.tagsCache && now - this.tagsCache.fetchedAt < OllamaLLM.TAGS_CACHE_TTL_MS) {
      return this.tagsCache.names;
    }
    // /api/tags accepts GET; payload is empty.
    type TagsResponse = { models?: Array<{ name: string }> } | Array<{ name: string }>;
    const resp = await this.request<TagsResponse>("/api/tags", {}, "GET");
    const list = Array.isArray(resp) ? resp : (resp.models ?? []);
    const names = new Set<string>(list.map((m) => m.name));
    this.tagsCache = { names, fetchedAt: now };
    return names;
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
    // Cloud tokenizers are not reversible from opaque integer IDs. Returning
    // the empty string is safe because store.ts only uses detokenize as a
    // fallback when tokenizer-based truncation is required; the upstream
    // LlamaCpp.detokenize round-trips real text. For the cloud path the
    // char-approximation path in tokenize() is what callers should use.
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
      description: `Ollama Cloud provider (${this.baseUrl}, embed=${this.embedModelNameValue})`,
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
