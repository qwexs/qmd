/**
 * store-init-worker - one subprocess that opens a store and exits.
 *
 * Spawned N-up by store-concurrency.test.ts to reproduce the cross-connection
 * schema-init race. Each worker busy-waits until a shared wall-clock start time
 * so every process hits initializeDatabase at once, then opens the store. Exits
 * 0 on success, 1 if createStore throws (e.g. "trigger already exists").
 */
import { createStore } from "../../src/store.ts";

const dbPath = process.argv[2];
const startAtMs = Number(process.argv[3] ?? "0");

if (!dbPath) {
  console.error("usage: store-init-worker <dbPath> [startAtMs]");
  process.exit(2);
}

while (Date.now() < startAtMs) {
  // Align all workers to the same start so their DROP/CREATE windows overlap.
}

try {
  const store = createStore(dbPath);
  store.close();
  process.exit(0);
} catch (err) {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
}
