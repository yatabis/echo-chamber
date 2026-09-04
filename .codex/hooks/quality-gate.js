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
const NATIVE_WORKSPACE_PATH = 'native/echo-inference';
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
const NATIVE_QUALITY_CHECKS = [
  {
    label: 'native format',
    command: 'cargo',
    args: ['fmt', '--all', '--', '--check'],
    cwd: NATIVE_WORKSPACE_PATH,
  },
  {
    label: 'native state tests',
    command: 'cargo',
    args: ['test', '-p', 'echo-inference-state', '--all-features'],
    cwd: NATIVE_WORKSPACE_PATH,
  },
];
const NATIVE_MLX_QUALITY_CHECKS = [
  {
    label: 'native clippy',
    command: 'cargo',
    args: [
      'clippy',
      '--workspace',
      '--all-targets',
      '--all-features',
      '--',
      '-D',
      'warnings',
    ],
    cwd: NATIVE_WORKSPACE_PATH,
  },
  {
    label: 'native tests',
    command: 'cargo',
    args: ['test', '--workspace', '--all-features', '--', '--test-threads=1'],
    cwd: NATIVE_WORKSPACE_PATH,
  },
];
const MLX_ENVIRONMENT_VARIABLES = [
  'MLX_C_INCLUDE_DIR',
  'MLX_C_LIB_DIR',
  'MLX_LIB_DIR',
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
 * Computes a digest of HEAD plus non-ignored tracked, staged, and untracked
 * changes. An optional path scope supports independently detecting Native
 * workspace changes.
 *
 * @param {string} repositoryRoot Absolute repository root.
 * @param {string[]} [pathspecs] Optional Git pathspec scope.
 * @returns {Promise<string>} SHA-256 worktree fingerprint.
 */
async function createWorktreeFingerprint(repositoryRoot, pathspecs = []) {
  const pathArguments = pathspecs.length === 0 ? [] : ['--', ...pathspecs];
  const headArguments =
    pathspecs.length === 0
      ? ['rev-parse', 'HEAD']
      : ['ls-tree', '-d', 'HEAD', '--', ...pathspecs];
  const [head, status, unstagedDiff, stagedDiff, untrackedPaths] =
    await Promise.all([
      runGit(headArguments, repositoryRoot),
      runGit(
        [
          'status',
          '--porcelain=v1',
          '-z',
          '--untracked-files=all',
          ...pathArguments,
        ],
        repositoryRoot
      ),
      runGit(
        ['diff', '--binary', '--no-ext-diff', ...pathArguments],
        repositoryRoot
      ),
      runGit(
        ['diff', '--cached', '--binary', '--no-ext-diff', ...pathArguments],
        repositoryRoot
      ),
      runGit(
        ['ls-files', '--others', '--exclude-standard', '-z', ...pathArguments],
        repositoryRoot
      ),
    ]);
  const hash = createHash('sha256');

  hash.update(head);
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
 * @param {{label: string, command: string, args: string[], cwd?: string}} check Check command.
 * @param {string} repositoryRoot Absolute repository root.
 * @returns {Promise<{code: number, output: string}>} Exit code and output tail.
 */
async function runQualityCheck(check, repositoryRoot) {
  return new Promise((resolve) => {
    const child = spawn(check.command, check.args, {
      cwd: join(repositoryRoot, check.cwd ?? ''),
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
  const [fingerprint, nativeFingerprint] = await Promise.all([
    createWorktreeFingerprint(repositoryRoot),
    createWorktreeFingerprint(repositoryRoot, [NATIVE_WORKSPACE_PATH]),
  ]);

  await mkdir(dirname(statePath), { recursive: true });
  await writeFile(
    statePath,
    JSON.stringify({ fingerprint, nativeFingerprint }),
    'utf8'
  );
  writeHookResponse({});
}

/**
 * Loads a previously recorded worktree fingerprint, if one exists.
 *
 * @param {string} statePath Baseline state path.
 * @returns {Promise<{fingerprint: string, nativeFingerprint: string | null} | null>} Saved fingerprints.
 */
async function readBaseline(statePath) {
  try {
    const state = JSON.parse(await readFile(statePath, 'utf8'));

    return typeof state.fingerprint === 'string'
      ? {
          fingerprint: state.fingerprint,
          nativeFingerprint:
            typeof state.nativeFingerprint === 'string'
              ? state.nativeFingerprint
              : null,
        }
      : null;
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
  const [baseline, current, currentNative] = await Promise.all([
    readBaseline(statePath),
    createWorktreeFingerprint(repositoryRoot),
    createWorktreeFingerprint(repositoryRoot, [NATIVE_WORKSPACE_PATH]),
  ]);

  await rm(statePath, { force: true });

  if (baseline?.fingerprint === current) {
    writeHookResponse({});
    return;
  }

  const nativeChanged =
    baseline?.nativeFingerprint === null ||
    baseline === null ||
    baseline.nativeFingerprint !== currentNative;
  const mlxEnvironmentAvailable = MLX_ENVIRONMENT_VARIABLES.every((name) =>
    process.env[name]?.trim()
  );
  const checks = [
    ...QUALITY_CHECKS,
    ...(nativeChanged ? NATIVE_QUALITY_CHECKS : []),
    ...(nativeChanged && mlxEnvironmentAvailable
      ? NATIVE_MLX_QUALITY_CHECKS
      : []),
  ];

  for (const check of checks) {
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
