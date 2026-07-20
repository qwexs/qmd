#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { chmodSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(fileURLToPath(new URL("..", import.meta.url)));

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    // Passing an absolute executable path through cmd.exe breaks when Node is
    // installed under "C:\\Program Files". spawnSync handles it directly on
    // every supported platform.
    shell: false,
    ...options,
  });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

run(process.execPath, [join(root, "node_modules", "typescript", "bin", "tsc"), "-p", "tsconfig.build.json"]);

const cliPath = join(root, "dist", "cli", "qmd.js");
const tmpPath = `${cliPath}.tmp`;
const built = readFileSync(cliPath, "utf8");
const withoutExistingShebang = built.startsWith("#!") ? built.slice(built.indexOf("\n") + 1) : built;
writeFileSync(tmpPath, `#!/usr/bin/env node\n${withoutExistingShebang}`);
renameSync(tmpPath, cliPath);
chmodSync(cliPath, 0o755);
