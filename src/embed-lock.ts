import { randomUUID } from "node:crypto";
import { hostname } from "node:os";
import type { Database } from "./db.js";

export type EmbedLockOwner = {
  token: string;
  pid: number;
  hostname: string;
  startedAt: string;
  indexPath: string;
};

export type EmbedLock = {
  acquired: boolean;
  owner: EmbedLockOwner | null;
  reason: "acquired" | "lock-held";
  recoveredStale: boolean;
  release: () => void;
};

type AcquireEmbedLockOptions = {
  staleMs: number;
  now?: Date;
  pid?: number;
  hostname?: string;
};

type LockRow = {
  token: string;
  pid: number;
  hostname: string;
  started_at: string;
  index_path: string;
};

const LOCK_NAME = "embed";

function rowToOwner(row: LockRow | undefined): EmbedLockOwner | null {
  if (!row) return null;
  return {
    token: row.token,
    pid: row.pid,
    hostname: row.hostname,
    startedAt: row.started_at,
    indexPath: row.index_path,
  };
}

function processIsAlive(pid: number): boolean {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

function ownerIsActive(owner: EmbedLockOwner, localHostname: string, nowMs: number, staleMs: number): boolean {
  // A local PID is authoritative and lets a crashed process recover on the
  // next invocation without waiting for the lease to expire. Remote owners
  // cannot be probed, so their lease age is authoritative.
  if (owner.hostname === localHostname) return processIsAlive(owner.pid);
  const startedAt = Date.parse(owner.startedAt);
  if (!Number.isFinite(startedAt)) return false;
  return Math.max(0, nowMs - startedAt) <= staleMs;
}

function ensureLockTable(db: Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS runtime_locks (
      name TEXT PRIMARY KEY,
      token TEXT NOT NULL,
      pid INTEGER NOT NULL,
      hostname TEXT NOT NULL,
      started_at TEXT NOT NULL,
      index_path TEXT NOT NULL
    )
  `);
}

/**
 * Acquire an atomic, index-scoped embed lock.
 *
 * The lock is a compare-and-swap row inside the physical SQLite index. SQLite
 * serializes competing transactions, which avoids the stale-file replacement
 * race where two recoverers can both believe they acquired the same lock.
 * Live local PIDs are never stolen; dead local owners recover immediately and
 * remote owners recover after the configured lease duration.
 */
export function acquireEmbedLock(db: Database, indexPath: string, options: AcquireEmbedLockOptions): EmbedLock {
  ensureLockTable(db);
  const now = options.now ?? new Date();
  const localPid = options.pid ?? process.pid;
  const localHostname = options.hostname ?? hostname();
  const staleMs = Math.max(1, options.staleMs);
  const token = randomUUID();
  const candidate: EmbedLockOwner = {
    token,
    pid: localPid,
    hostname: localHostname,
    startedAt: now.toISOString(),
    indexPath,
  };

  let result: { acquired: boolean; recoveredStale: boolean; owner: EmbedLockOwner | null } | null = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    // Each mutation is its own atomic statement. Avoid a deferred read
    // transaction here: several stale-lock contenders can otherwise all read
    // the old row and then fail with SQLITE_BUSY_SNAPSHOT while upgrading.
    const inserted = db.prepare(`
      INSERT OR IGNORE INTO runtime_locks (name, token, pid, hostname, started_at, index_path)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(LOCK_NAME, token, localPid, localHostname, candidate.startedAt, indexPath);
    if (inserted.changes === 1) {
      result = { acquired: true, recoveredStale: false, owner: candidate };
      break;
    }

    const row = db.prepare(`
      SELECT token, pid, hostname, started_at, index_path
      FROM runtime_locks
      WHERE name = ?
    `).get<LockRow>(LOCK_NAME);
    const observed = rowToOwner(row);
    if (!observed) continue; // The previous owner released between statements.

    if (ownerIsActive(observed, localHostname, now.getTime(), staleMs)) {
      result = { acquired: false, recoveredStale: false, owner: observed };
      break;
    }

    const updated = db.prepare(`
      UPDATE runtime_locks
      SET token = ?, pid = ?, hostname = ?, started_at = ?, index_path = ?
      WHERE name = ? AND token = ?
    `).run(token, localPid, localHostname, candidate.startedAt, indexPath, LOCK_NAME, observed.token);
    if (updated.changes === 1) {
      result = { acquired: true, recoveredStale: true, owner: candidate };
      break;
    }
  }

  if (!result) {
    const winner = db.prepare(`
      SELECT token, pid, hostname, started_at, index_path
      FROM runtime_locks
      WHERE name = ?
    `).get<LockRow>(LOCK_NAME);
    result = { acquired: false, recoveredStale: false, owner: rowToOwner(winner) };
  }
  if (!result.acquired) {
    return {
      acquired: false,
      owner: result.owner,
      reason: "lock-held",
      recoveredStale: result.recoveredStale,
      release: () => {},
    };
  }

  return {
    acquired: true,
    owner: candidate,
    reason: "acquired",
    recoveredStale: result.recoveredStale,
    release: () => {
      try {
        db.prepare(`DELETE FROM runtime_locks WHERE name = ? AND token = ?`).run(LOCK_NAME, token);
      } catch {
        // Do not mask an embed failure during finally. A crashed/closed owner
        // is recovered by PID or lease on the next invocation.
      }
    },
  };
}
