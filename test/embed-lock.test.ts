import { afterEach, describe, expect, test } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import { openDatabase, type Database } from "../src/db.js";
import { acquireEmbedLock } from "../src/embed-lock.js";

const tempDirs: string[] = [];
const databases: Database[] = [];

afterEach(async () => {
  for (const db of databases.splice(0)) db.close();
  await Promise.all(tempDirs.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

async function testIndex(): Promise<{ indexPath: string; db: Database }> {
  const dir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-"));
  tempDirs.push(dir);
  const indexPath = join(dir, "index.sqlite");
  const db = openDatabase(indexPath);
  databases.push(db);
  return { indexPath, db };
}

describe("embed index lock", () => {
  test("serializes callers and releases only its own lock", async () => {
    const { indexPath, db } = await testIndex();
    const first = acquireEmbedLock(db, indexPath, { staleMs: 60_000 });
    expect(first.acquired).toBe(true);

    const second = acquireEmbedLock(db, indexPath, { staleMs: 60_000 });
    expect(second.acquired).toBe(false);
    expect(second.reason).toBe("lock-held");

    first.release();
    const third = acquireEmbedLock(db, indexPath, { staleMs: 60_000 });
    expect(third.acquired).toBe(true);
    third.release();
  });

  test("recovers a lock immediately when its local process is dead", async () => {
    const { indexPath, db } = await testIndex();
    db.exec(`
      CREATE TABLE runtime_locks (
        name TEXT PRIMARY KEY, token TEXT NOT NULL, pid INTEGER NOT NULL,
        hostname TEXT NOT NULL, started_at TEXT NOT NULL, index_path TEXT NOT NULL
      )
    `);
    db.prepare(`
      INSERT INTO runtime_locks (name, token, pid, hostname, started_at, index_path)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run("embed", "abandoned", 999_999_999, hostname(), new Date().toISOString(), indexPath);

    const acquired = acquireEmbedLock(db, indexPath, { staleMs: 60_000 });
    expect(acquired.acquired).toBe(true);
    expect(acquired.recoveredStale).toBe(true);
    const owner = db.prepare(`SELECT pid FROM runtime_locks WHERE name = ?`).get<{ pid: number }>("embed");
    expect(owner?.pid).toBe(process.pid);
    acquired.release();
  });

  test("allows exactly one winner when many callers recover the same stale owner", async () => {
    const { indexPath, db } = await testIndex();
    const dead = acquireEmbedLock(db, indexPath, { staleMs: 60_000, pid: 999_999_999 });
    expect(dead.acquired).toBe(true);

    const contenders = Array.from({ length: 20 }, () => {
      const connection = openDatabase(indexPath);
      databases.push(connection);
      return acquireEmbedLock(connection, indexPath, { staleMs: 60_000 });
    });

    expect(contenders.filter((lock) => lock.acquired)).toHaveLength(1);
    for (const lock of contenders) lock.release();
  });
});
