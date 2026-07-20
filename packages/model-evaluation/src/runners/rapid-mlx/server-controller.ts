import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  createWriteStream,
  existsSync,
  mkdtempSync,
  rmSync,
  symlinkSync,
} from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { basename, delimiter, join, sep } from 'node:path';

import type { LocalEvaluationTarget } from '../../qwen36-eat-readiness/types';
import type { ChildProcess } from 'node:child_process';

interface ServerControllerOptions {
  target: LocalEvaluationTarget;
  rapidMlxBin: string;
  rapidMlxWorkingDirectory: string;
  port: number;
  logPath: string;
  kvCacheDtype: 'int4' | 'int8' | 'bf16';
  prefixCacheMode: 'enabled' | 'disabled';
}

export interface RunningLocalModelServer {
  baseURL: string;
  startupElapsedMs: number;
  temporaryModelPath: string;
  temporaryPrefixCachePath: string;
  stop(): Promise<{ exitCode: number | null; signalCode: string | null }>;
  cleanup(): void;
}

async function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

function prefixCachePathForModelPath(modelPath: string): string {
  const safeName =
    modelPath
      .replaceAll('/', '--')
      .replaceAll('\\', '--')
      .replaceAll('..', '--')
      .replace(/^\.+/, '') || 'default';
  const digest = createHash('sha256')
    .update(modelPath)
    .digest('hex')
    .slice(0, 8);
  return join(
    homedir(),
    '.cache',
    'rapid-mlx',
    'prefix_cache',
    `${safeName}--${digest}`
  );
}

function assertSafeTemporaryPath(path: string, parent: string): void {
  const normalizedParent = parent.endsWith(sep) ? parent : `${parent}${sep}`;
  if (!path.startsWith(normalizedParent) || path === parent) {
    throw new Error(`Refusing to remove non-temporary path: ${path}`);
  }
}

async function waitUntilReady(
  child: ChildProcess,
  readyURL: string,
  timeoutMs: number
): Promise<void> {
  const deadline = performance.now() + timeoutMs;
  while (performance.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(
        `Rapid-MLX exited before readiness with code ${child.exitCode}`
      );
    }

    try {
      // Readiness polling is sequential and stops immediately on success.
      // eslint-disable-next-line no-await-in-loop
      const response = await fetch(readyURL);
      if (response.ok) {
        return;
      }
    } catch {
      // Connection refusal is expected while the model is loading.
    }

    // Polling must remain sequential to avoid piling up health requests.
    // eslint-disable-next-line no-await-in-loop
    await delay(500);
  }

  throw new Error(`Rapid-MLX did not become ready within ${timeoutMs}ms`);
}

async function waitForExit(
  child: ChildProcess,
  timeoutMs: number
): Promise<boolean> {
  if (child.exitCode !== null || child.signalCode !== null) {
    return true;
  }

  return await Promise.race([
    new Promise<boolean>((resolve) => {
      child.once('exit', () => {
        resolve(true);
      });
    }),
    delay(timeoutMs).then(() => false),
  ]);
}

/** 評価専用モデルパスでRapid-MLXを起動する。 */
// Startup and bounded shutdown are kept together so cleanup owns one process lifecycle.
// eslint-disable-next-line max-lines-per-function
export async function startLocalModelServer(
  options: ServerControllerOptions
): Promise<RunningLocalModelServer> {
  if (!existsSync(options.target.modelPath)) {
    throw new Error(`Model path does not exist: ${options.target.modelPath}`);
  }
  if (!existsSync(options.rapidMlxBin)) {
    throw new Error(`Rapid-MLX binary does not exist: ${options.rapidMlxBin}`);
  }

  const temporaryRoot = mkdtempSync(join(tmpdir(), 'echo-qwen36-eval-'));
  const temporaryModelPath = join(
    temporaryRoot,
    basename(options.target.modelPath)
  );
  symlinkSync(options.target.modelPath, temporaryModelPath, 'dir');
  const temporaryPrefixCachePath =
    prefixCachePathForModelPath(temporaryModelPath);
  const logStream = createWriteStream(options.logPath, { flags: 'w' });
  const startedAt = performance.now();
  const child = spawn(
    options.rapidMlxBin,
    [
      'serve',
      temporaryModelPath,
      '--served-model-name',
      options.target.servedModelName,
      '--host',
      '127.0.0.1',
      '--port',
      String(options.port),
      '--log-level',
      'INFO',
      '--max-num-seqs',
      '1',
      '--max-concurrent-requests',
      '1',
      '--prefill-step-size',
      '2048',
      options.prefixCacheMode === 'enabled'
        ? '--enable-prefix-cache'
        : '--disable-prefix-cache',
      '--kv-cache-dtype',
      options.kvCacheDtype,
      '--kv-disk-checkpoint-interval',
      '0',
      '--max-tokens',
      '3072',
      '--enable-auto-tool-choice',
      '--tool-call-parser',
      'qwen3_coder_xml',
      '--reasoning-parser',
      'qwen3',
      '--no-spec-decode',
      '--gpu-memory-utilization',
      '0.75',
      '--pflash',
      'off',
      '--text-only',
      '--timeout',
      '1800',
    ],
    {
      cwd: options.rapidMlxWorkingDirectory,
      env: {
        ...process.env,
        // The evaluation binary lives in an isolated venv, but the subject
        // under test is the Rapid-MLX working tree supplied by the caller.
        // Put that tree first so the console-script does not silently import
        // the venv's older wheel copy.
        PYTHONPATH: [options.rapidMlxWorkingDirectory, process.env.PYTHONPATH]
          .filter((path): path is string => path !== undefined && path !== '')
          .join(delimiter),
        RAPID_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET: '0.01',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    }
  );
  child.stdout.pipe(logStream, { end: false });
  child.stderr.pipe(logStream, { end: false });

  const readyURL = `http://127.0.0.1:${options.port}/health/ready`;
  try {
    await waitUntilReady(child, readyURL, 10 * 60_000);
  } catch (error) {
    child.kill('SIGTERM');
    await waitForExit(child, 10_000);
    logStream.end();
    const prefixRoot = join(homedir(), '.cache', 'rapid-mlx', 'prefix_cache');
    assertSafeTemporaryPath(temporaryRoot, tmpdir());
    assertSafeTemporaryPath(temporaryPrefixCachePath, prefixRoot);
    rmSync(temporaryRoot, { recursive: true, force: true });
    rmSync(temporaryPrefixCachePath, { recursive: true, force: true });
    throw error;
  }

  let stopped = false;
  return {
    baseURL: `http://127.0.0.1:${options.port}/v1`,
    startupElapsedMs: Math.round(performance.now() - startedAt),
    temporaryModelPath,
    temporaryPrefixCachePath,
    async stop(): Promise<{
      exitCode: number | null;
      signalCode: string | null;
    }> {
      if (stopped) {
        return { exitCode: child.exitCode, signalCode: child.signalCode };
      }

      stopped = true;
      child.kill('SIGINT');
      if (!(await waitForExit(child, 30_000))) {
        child.kill('SIGTERM');
      }
      if (!(await waitForExit(child, 10_000))) {
        child.kill('SIGKILL');
        await waitForExit(child, 5_000);
      }
      logStream.end();
      return { exitCode: child.exitCode, signalCode: child.signalCode };
    },
    cleanup(): void {
      const prefixRoot = join(homedir(), '.cache', 'rapid-mlx', 'prefix_cache');
      assertSafeTemporaryPath(temporaryRoot, tmpdir());
      assertSafeTemporaryPath(temporaryPrefixCachePath, prefixRoot);
      rmSync(temporaryRoot, { recursive: true, force: true });
      rmSync(temporaryPrefixCachePath, { recursive: true, force: true });
    },
  };
}
