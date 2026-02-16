/**
 * openai-llm.ts - OpenAI API backend for QMD
 *
 * Alternative LLM provider using OpenAI-compatible APIs for embeddings and generation.
 * Reranking is implemented via cosine similarity of embeddings.
 *
 * Environment variables:
 *   OPENAI_API_KEY          - Required. OpenAI API key.
 *   OPENAI_BASE_URL         - Optional. Base URL (default: https://api.openai.com/v1).
 *   OPENAI_PROXY_URL        - Optional. HTTP proxy URL (uses undici ProxyAgent).
 *   OPENAI_EMBED_MODEL      - Optional. Embedding model (default: text-embedding-3-small).
 *   OPENAI_GENERATE_MODEL   - Optional. Generation model (default: gpt-4o-mini).
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
} from "./llm";

// =============================================================================
// Configuration
// =============================================================================

const DEFAULT_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_EMBED_MODEL = "text-embedding-3-small";
const DEFAULT_GENERATE_MODEL = "gpt-4o-mini";
const MAX_BATCH_SIZE = 100;

// =============================================================================
// Proxy support
// =============================================================================

type FetchFn = typeof globalThis.fetch;

async function createFetchFn(): Promise<FetchFn> {
  const proxyUrl = process.env.OPENAI_PROXY_URL;
  if (!proxyUrl) return globalThis.fetch;

  try {
    const { ProxyAgent } = await import("undici");
    const agent = new ProxyAgent(proxyUrl);
    return (input: RequestInfo | URL, init?: RequestInit) => {
      return globalThis.fetch(input, { ...init, dispatcher: agent } as any);
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
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// =============================================================================
// OpenAILLM Implementation
// =============================================================================

export class OpenAILLM implements LLM {
  private apiKey: string;
  private baseUrl: string;
  private embedModel: string;
  private generateModel: string;
  private fetchFn: FetchFn | null = null;

  constructor() {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OPENAI_API_KEY environment variable is required");
    }
    this.apiKey = apiKey;
    this.baseUrl = process.env.OPENAI_BASE_URL || DEFAULT_BASE_URL;
    this.embedModel = process.env.OPENAI_EMBED_MODEL || DEFAULT_EMBED_MODEL;
    this.generateModel = process.env.OPENAI_GENERATE_MODEL || DEFAULT_GENERATE_MODEL;
  }

  private async getFetch(): Promise<FetchFn> {
    if (!this.fetchFn) {
      this.fetchFn = await createFetchFn();
    }
    return this.fetchFn;
  }

  private async request<T>(endpoint: string, body: Record<string, unknown>): Promise<T> {
    const fetch = await this.getFetch();
    const resp = await fetch(`${this.baseUrl}${endpoint}`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`OpenAI API error ${resp.status}: ${text}`);
    }

    return resp.json() as Promise<T>;
  }

  // ==========================================================================
  // Embeddings
  // ==========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const results = await this.embedTexts([text]);
      return results[0];
    } catch (error) {
      console.error("OpenAI embedding error:", error);
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
          results[i + j] = batchResults[j];
        }
      } catch (error) {
        console.warn(`OpenAI batch embed failed, falling back to individual requests:`, error);
        // Fallback: embed one by one
        for (let j = 0; j < batch.length; j++) {
          try {
            const single = await this.embedTexts([batch[j]]);
            results[i + j] = single[0];
          } catch (innerError) {
            console.error(`OpenAI embed failed for text ${i + j}:`, innerError);
            results[i + j] = null;
          }
        }
      }
    }

    return results;
  }

  private async embedTexts(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    type OpenAIEmbedResponse = {
      data: Array<{ embedding: number[]; index: number }>;
      model: string;
      usage: { total_tokens: number };
    };

    const resp = await this.request<OpenAIEmbedResponse>("/embeddings", {
      model: this.embedModel,
      input: texts,
    });

    // Sort by index to maintain order
    const sorted = resp.data.sort((a, b) => a.index - b.index);
    return sorted.map((item) => ({
      embedding: item.embedding,
      model: resp.model,
    }));
  }

  // ==========================================================================
  // Reranking (via cosine similarity)
  // ==========================================================================

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    // Embed query and all documents
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

    // Sort by score descending (highest relevance first)
    results.sort((a, b) => b.score - a.score);

    return {
      results,
      model: this.embedModel,
    };
  }

  // ==========================================================================
  // Generation
  // ==========================================================================

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    type OpenAIChatResponse = {
      choices: Array<{ message: { content: string } }>;
      model: string;
    };

    try {
      const resp = await this.request<OpenAIChatResponse>("/chat/completions", {
        model: this.generateModel,
        messages: [{ role: "user", content: prompt }],
        max_tokens: options?.maxTokens ?? 150,
        temperature: options?.temperature ?? 0.7,
      });

      const text = resp.choices?.[0]?.message?.content || "";
      return {
        text,
        model: resp.model,
        done: true,
      };
    } catch (error) {
      console.error("OpenAI generation error:", error);
      return null;
    }
  }

  // ==========================================================================
  // Query Expansion
  // ==========================================================================

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    const prompt = `Expand this search query into variations. Return one per line in format:
type: text
Where type is lex (keyword search), vec (semantic search), or hyde (hypothetical document).
Query: ${query}`;

    const result = await this.generate(prompt, { maxTokens: 150, temperature: 0.7 });
    if (!result?.text) {
      // Fallback
      const fallback: Queryable[] = [
        { type: "vec", text: query },
        { type: "lex", text: query },
      ];
      return options?.includeLexical === false ? fallback.filter((q) => q.type !== "lex") : fallback;
    }

    const lines = result.text.trim().split("\n");
    const queryables: Queryable[] = lines
      .map((line) => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim();
        if (type !== "lex" && type !== "vec" && type !== "hyde") return null;
        const text = line.slice(colonIdx + 1).trim();
        if (!text) return null;
        return { type: type as Queryable["type"], text };
      })
      .filter((q): q is Queryable => q !== null);

    const includeLexical = options?.includeLexical ?? true;
    const filtered = includeLexical ? queryables : queryables.filter((q) => q.type !== "lex");

    if (filtered.length > 0) return filtered;

    // Fallback
    const fallback: Queryable[] = [
      { type: "hyde", text: `Information about ${query}` },
      { type: "lex", text: query },
      { type: "vec", text: query },
    ];
    return includeLexical ? fallback : fallback.filter((q) => q.type !== "lex");
  }

  // ==========================================================================
  // Model info
  // ==========================================================================

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  // ==========================================================================
  // Tokenization (approximate, ~4 chars per token)
  // ==========================================================================

  async tokenize(text: string): Promise<readonly number[]> {
    const count = Math.ceil(text.length / 4);
    return Array.from({ length: count }, (_, i) => i);
  }

  async countTokens(text: string): Promise<number> {
    return Math.ceil(text.length / 4);
  }

  async detokenize(_tokens: readonly number[]): Promise<string> {
    return "";
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
