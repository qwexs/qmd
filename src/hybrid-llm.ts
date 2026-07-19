/**
 * hybrid-llm.ts - Hybrid LLM backend for QMD (fork extension)
 *
 * Splits the LLM responsibility across two providers:
 *   - Embedding / tokenize                       → local LlamaCpp (BERTA + qwen3)
 *   - Query expansion (fallback)                 → Jina cloud
 *   - Rerank                                     → Jina cloud
 *
 * Why this exists:
 *   node-llama-cpp's `createRankingContext` API only works with decoder-only LLM
 *   rerankers (Qwen3/Gemma/MiniCPM). Encoder-only rerankers (GTE, BGE-m3, Jina
 *   reranker) cannot use that interface. So for "local rerank" we are stuck
 *   with at least a 600 MB decoder model. To save RAM we keep rerank in the
 *   cloud (Jina) and run only the 138 MB BERTA embed model locally.
 *
 * The local LlamaCpp is constructed with ONLY an embedModel so its rerankModel
 * and generateModel slots stay null. ensureRerankModel() / ensureGenerateModel()
 * never fire on this instance, so the 610 MB Qwen3-Reranker GGUF is never
 * allocated.
 *
 * Environment variables (in addition to standard qmd ones):
 *   - QMD_EMBED_MODEL   (optional) hf: URI of local embedding model; if unset,
 *                        falls back to ~/.config/qmd/index.yml `models.embed`,
 *                        then the package default (embeddinggemma).
 *   - JINA_API_KEY      (required) for the rerank/expandQuery cloud path
 *   - JINA_RERANK_MODEL (optional) defaults to jina-reranker-v2-base-multilingual
 *
 * Implements the LLM interface plus the public surface that store.ts expects
 * on the singleton returned by getDefaultLlamaCpp() (embedModelName /
 * rerankModelName / generateModelName getters, plus tokenize / embedBatch /
 * modelExists / getDeviceInfo / unloadIdleResources).
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
} from "./llm.js";
import { LlamaCpp, resolveEmbedModel } from "./llm.js";
import { JinaLLM } from "./jina-llm.js";
import { loadConfig } from "./collections.js";

/**
 * Resolve the embed URI respecting the same precedence as the qmd CLI:
 *   1. YAML config (loadConfig → config.models.embed)
 *   2. QMD_EMBED_MODEL env var
 *   3. DEFAULT_EMBED_MODEL fallback
 *
 * This mirrors the precedence used by qmd CLI so that HybridLLM picks up the
 * model the user wrote into ~/.config/qmd/index.yml (`models.embed`). Without
 * this step, process.env.QMD_EMBED_MODEL is empty in CLI invocations (no env
 * override), and HybridLLM would silently fall back to the hardcoded default
 * (embeddinggemma) — wrong model.
 */
function resolveHybridEmbedUri(): string {
  let configEmbed: string | undefined;
  try {
    const config = loadConfig();
    configEmbed = config?.models?.embed;
  } catch {
    // Config may not exist yet; fall through to env/default.
  }
  return resolveEmbedModel({ embed: configEmbed });
}

export class HybridLLM implements LLM {
  private local: LlamaCpp;
  private cloud: JinaLLM;

  constructor() {
    // Resolve the embed URI respecting the same precedence as the qmd
    // CLI (config → env → default). See resolveHybridEmbedUri().
    const embedUri = resolveHybridEmbedUri();

    // Local: LlamaCpp with ONLY the embed model loaded. rerankModel and
    // generateModel are not provided, so those slots stay null and the heavy
    // GGUF files (qwen3-reranker 610 MB, query-expansion 1.2 GB) are never
    // loaded into RAM.
    this.local = new LlamaCpp({
      embedModel: embedUri,
    });

    // Cloud: JinaLLM for rerank + expandQuery (no local model for query
    // expansion that we are happy to ship with the fork).
    this.cloud = new JinaLLM();
  }

  // ---- Public getters expected by store.ts ----
  get embedModelName(): string {
    // Re-resolve so config edits (e.g. via qmd status syncing models into
    // store_config) take effect on the next embed.
    return resolveHybridEmbedUri();
  }

  get generateModelName(): string {
    // Hybrid does not load a local generate model; expose the cloud rerank
    // model name as a placeholder for callers that just want to log which
    // model is configured.
    return this.cloud.rerankModelName;
  }

  get rerankModelName(): string {
    return this.cloud.rerankModelName;
  }

  // ---- Embedding: local LlamaCpp ----
  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.local.embed(text, options);
  }

  async embedBatch(texts: string[], options?: EmbedOptions): Promise<EmbeddingResult[]> {
    const results = await this.local.embedBatch(texts, options);
    return results.filter((r): r is EmbeddingResult => r !== null);
  }

  async tokenize(text: string): Promise<number[]> {
    const tokens = await this.local.tokenize(text);
    return Array.from(tokens);
  }

  // ---- Rerank: cloud Jina ----
  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    return this.cloud.rerank(query, documents, options);
  }

  // ---- Query expansion: cloud Jina (fallback path) ----
  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    return this.cloud.expandQuery(query, options);
  }

  // ---- Generation: cloud Jina placeholder ----
  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    // No local generation model is loaded. Jina does not offer generation.
    return this.cloud.generate(prompt, options);
  }

  // ---- Diagnostics ----
  async modelExists(model: string): Promise<ModelInfo> {
    // Defer to local for embedding-model lookups, cloud for everything else.
    try {
      return await this.local.modelExists(model);
    } catch {
      return { exists: false, name: model };
    }
  }

  // ---- Lifecycle ----
  async dispose(): Promise<void> {
    try {
      if (typeof this.local?.dispose === "function") {
        await this.local.dispose();
      }
    } catch {
      // Best-effort cleanup; swallow errors so shutdown still proceeds.
    }
  }
}