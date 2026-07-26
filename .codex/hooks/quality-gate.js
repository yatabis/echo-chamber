import { createHash } from 'node:crypto';
import { execFile, spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import {
  lstat,
  mkdir,
  readFile,
  readlink,
  rm,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);
const MAX_COMMAND_OUTPUT_LENGTH = 8_000;
const STATE_DIRECTORY = join(tmpdir(), 'echo-chamber-codex-quality-gate');
const QUALITY_CHECKS = [
  {
    label: 'lint',
    command: 'pnpm',
    args: ['lint:check'],
  },
  {
    label: 'typecheck',
    command: 'pnpm',
    args: ['typecheck'],
  },
  {
    label: 'format',
    command: 'pnpm',
    args: ['format:check'],
  },
  {
    label: 'tests',
    command: 'pnpm',
    args: ['test:run'],
  },
];

/**
 * Reads and parses the JSON object provided to a command hook on stdin.
 *
 * @returns {Promise<Record<string, unknown>>} Parsed hook input.
 */
async function readHookInput() {
  const chunks = [];

  for await (const chunk of process.stdin) {
    chunks.push(chunk);
  }

  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

/**
 * Runs Git and returns its stdout without converting path bytes to text.
 *
 * @param {string[]} args Git arguments.
 * @param {string} cwd Working directory inside the repository.
 * @returns {Promise<Buffer>} Raw stdout.
 */
async function runGit(args, cwd) {
  const { stdout } = await execFileAsync('git', args, {
    cwd,
    encoding: 'buffer',
    maxBuffer: 64 * 1024 * 1024,
  });

  return stdout;
}

/**
 * Resolves the repository root for a hook invocation.
 *
 * @param {string} cwd Hook working directory.
 * @returns {Promise<string>} Absolute repository root.
 */
async function resolveRepositoryRoot(cwd) {
  const stdout = await runGit(['rev-parse', '--show-toplevel'], cwd);

  return stdout.toString('utf8').trim();
}

/**
 * Adds untracked file contents to a worktree fingerprint.
 *
 * Git diffs already cover tracked content, while untracked files require
 * explicit hashing so in-place edits do not escape change detection.
 *
 * @param {import('node:crypto').Hash} hash Fingerprint hash.
 * @param {Buffer} pathsOutput NUL-delimited untracked paths from Git.
 * @param {string} repositoryRoot Absolute repository root.
 * @returns {Promise<void>}
 */
async function hashUntrackedFiles(hash, pathsOutput, repositoryRoot) {
  const paths = pathsOutput
    .toString('utf8')
    .split('\0')
    .filter((path) => path.length > 0);

  for (const path of paths) {
    const absolutePath = join(repositoryRoot, path);
    const stats = await lstat(absolutePath);

    hash.update('\0path\0');
    hash.update(path);
    hash.update(`\0mode\0${stats.mode}`);

    if (stats.isSymbolicLink()) {
      hash.update('\0link\0');
      hash.update(await readlink(absolutePath));
    } else if (stats.isFile()) {
      hash.update('\0file\0');

      for await (const chunk of createReadStream(absolutePath)) {
        hash.update(chunk);
      }
    }
  }
}

/**
 * Computes a digest of all non-ignored tracked, staged, and untracked changes.
 *
 * @param {string} repositoryRoot Absolute repository root.
 * @returns {Promise<string>} SHA-256 worktree fingerprint.
 */
async function createWorktreeFingerprint(repositoryRoot) {
  const [status, unstagedDiff, stagedDiff, untrackedPaths] = await Promise.all([
    runGit(
      ['status', '--porcelain=v1', '-z', '--untracked-files=all'],
      repositoryRoot
    ),
    runGit(['diff', '--binary', '--no-ext-diff'], repositoryRoot),
    runGit(['diff', '--cached', '--binary', '--no-ext-diff'], repositoryRoot),
    runGit(
      ['ls-files', '--others', '--exclude-standard', '-z'],
      repositoryRoot
    ),
  ]);
  const hash = createHash('sha256');

  hash.update(status);
  hash.update(unstagedDiff);
  hash.update(stagedDiff);
  await hashUntrackedFiles(hash, untrackedPaths, repositoryRoot);

  return hash.digest('hex');
}

/**
 * Builds a collision-resistant path for one Codex turn's baseline state.
 *
 * @param {Record<string, unknown>} input Hook input.
 * @param {string} repositoryRoot Absolute repository root.
 * @returns {string} Baseline state path.
 */
function createStatePath(input, repositoryRoot) {
  const identity = [
    repositoryRoot,
    String(input.session_id ?? 'unknown-session'),
    String(input.turn_id ?? 'unknown-turn'),
  ].join('\0');
  const filename = `${createHash('sha256').update(identity).digest('hex')}.json`;

  return join(STATE_DIRECTORY, filename);
}

/**
 * Appends output while retaining only the tail needed for failure feedback.
 *
 * @param {string} current Existing output tail.
 * @param {Buffer} chunk New command output.
 * @returns {string} Updated bounded output tail.
 */
function appendCommandOutput(current, chunk) {
  return `${current}${chunk.toString('utf8')}`.slice(
    -MAX_COMMAND_OUTPUT_LENGTH
  );
}

/**
 * Runs one repository quality command and captures bounded diagnostic output.
 *
 * @param {{label: string, command: string, args: string[]}} check Check command.
 * @param {string} repositoryRoot Absolute repository root.
 * @returns {Promise<{code: number, output: string}>} Exit code and output tail.
 */
async function runQualityCheck(check, repositoryRoot) {
  return new Promise((resolve) => {
    const child = spawn(check.command, check.args, {
      cwd: repositoryRoot,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let output = '';

    child.stdout.on('data', (chunk) => {
      output = appendCommandOutput(output, chunk);
    });
    child.stderr.on('data', (chunk) => {
      output = appendCommandOutput(output, chunk);
    });
    child.on('error', (error) => {
      resolve({
        code: 1,
        output: `${output}\n${error.message}`.trim(),
      });
    });
    child.on('close', (code) => {
      resolve({
        code: code ?? 1,
        output: output.trim(),
      });
    });
  });
}

/**
 * Writes a valid JSON response for hook events that require JSON stdout.
 *
 * @param {Record<string, unknown>} response Hook response.
 * @returns {void}
 */
function writeHookResponse(response) {
  process.stdout.write(`${JSON.stringify(response)}\n`);
}

/**
 * Records the worktree state before Codex begins handling a user prompt.
 *
 * @param {Record<string, unknown>} input Hook input.
 * @param {string} repositoryRoot Absolute repository root.
 * @returns {Promise<void>}
 */
async function recordBaseline(input, repositoryRoot) {
  const statePath = createStatePath(input, repositoryRoot);
  const fingerprint = await createWorktreeFingerprint(repositoryRoot);

  await mkdir(dirname(statePath), { recursive: true });
  await writeFile(statePath, JSON.stringify({ fingerprint }), 'utf8');
  writeHookResponse({});
}

/**
 * Loads a previously recorded worktree fingerprint, if one exists.
 *
 * @param {string} statePath Baseline state path.
 * @returns {Promise<string | null>} Saved fingerprint.
 */
async function readBaseline(statePath) {
  try {
    const state = JSON.parse(await readFile(statePath, 'utf8'));

    return typeof state.fingerprint === 'string' ? state.fingerprint : null;
  } catch (error) {
    if (error && typeof error === 'object' && error.code === 'ENOENT') {
      return null;
    }

    throw error;
  }
}

/**
 * Runs the quality gate when the worktree changed during the current turn.
 *
 * A missing baseline fails safe by running the checks. This covers resumed
 * sessions and transient baseline-recording failures without silently
 * bypassing the gate.
 *
 * @param {Record<string, unknown>} input Hook input.
 * @param {string} repositoryRoot Absolute repository root.
 * @returns {Promise<void>}
 */
async function enforceQualityGate(input, repositoryRoot) {
  const statePath = createStatePath(input, repositoryRoot);
  const [baseline, current] = await Promise.all([
    readBaseline(statePath),
    createWorktreeFingerprint(repositoryRoot),
  ]);

  await rm(statePath, { force: true });

  if (baseline === current) {
    writeHookResponse({});
    return;
  }

  for (const check of QUALITY_CHECKS) {
    const result = await runQualityCheck(check, repositoryRoot);

    if (result.code !== 0) {
      const diagnostic =
        result.output.length > 0
          ? result.output
          : `${check.command} ${check.args.join(' ')} exited with code ${result.code}.`;

      writeHookResponse({
        decision: 'block',
        reason: [
          `Repository quality gate failed during ${check.label}.`,
          diagnostic,
          'Resolve every error, rerun the relevant checks, and do not finish the task until the gate passes.',
        ].join('\n\n'),
      });
      return;
    }
  }

  writeHookResponse({});
}

/**
 * Dispatches the current Codex hook event.
 *
 * @returns {Promise<void>}
 */
async function main() {
  const input = await readHookInput();
  const cwd = typeof input.cwd === 'string' ? input.cwd : process.cwd();
  const repositoryRoot = await resolveRepositoryRoot(cwd);

  if (input.hook_event_name === 'UserPromptSubmit') {
    await recordBaseline(input, repositoryRoot);
    return;
  }

  if (input.hook_event_name === 'Stop') {
    await enforceQualityGate(input, repositoryRoot);
    return;
  }

  writeHookResponse({});
}

main().catch((error) => {
  process.stderr.write(
    `Codex quality gate hook failed: ${
      error instanceof Error ? error.stack : String(error)
    }\n`
  );
  process.exitCode = 1;
});
