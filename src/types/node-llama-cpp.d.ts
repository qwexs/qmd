// Ambient type declarations for the optional `node-llama-cpp` package.
//
// `node-llama-cpp` is intentionally NOT in `dependencies`, `optionalDependencies`
// or `peerDependencies` in this fork so that `bun install` / `npm install` can
// succeed on a clean Ubuntu / Windows machine without a C++ toolchain — for
// users that only consume the cloud backends (QMD_LLM_PROVIDER=openai or =jina).
//
// The dynamic import is wrapped in `loadNodeLlamaCpp()` in src/llm.ts so the
// missing package only matters at runtime for the local LLM path; missing
// types are replaced here with `any` so `tsc` can still type-check the rest
// of the codebase (including the cloud LLM backends) when the package is not
// physically installed.
//
// If you have the real `node-llama-cpp` package installed, delete this file
// (or just leave it — TypeScript's module resolution prefers the bundled
// `.d.ts` from the installed package over this ambient fallback).

declare module "node-llama-cpp" {
  // Minimal surface used across the codebase. Marked as `any` so we don't
  // duplicate upstream's full type tree here; the real package supplies
  // proper types when present.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const getLlama: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const resolveModelFile: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const LlamaChatSession: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const LlamaLogLevel: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export type Llama = any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export type LlamaModel = any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export type LlamaEmbeddingContext = any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export type Token = any;
}
