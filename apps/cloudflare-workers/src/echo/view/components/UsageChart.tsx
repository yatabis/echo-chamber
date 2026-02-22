import type { Usage } from '@echo-chamber/core';

import type { FC } from 'hono/jsx';

interface UsageData {
  date: Date;
  usage: Usage | undefined;
}

const UsageChartBar: FC<{
  date: Date;
  usage: Usage | undefined;
  maxTokens: number;
}> = async ({ date, usage, maxTokens }) => {
  const label = `${date.getMonth() + 1}/${date.getDate()}`;

  if (!usage) {
    return (
      <div className="bar" key={label}>
        <div className="bar-x mono">{label}</div>
        <div
          className="bar-rect"
          style={{ height: '1px', backgroundColor: '#f0f0f0' }}
        />
        <div className="bar-y mono">0</div>
      </div>
    );
  }

  const {
    cached_input_tokens,
    uncached_input_tokens,
    output_tokens,
    reasoning_tokens,
    total_tokens,
    total_cost,
  } = usage;
  const finalOutputTokens = output_tokens - reasoning_tokens;

  const totalHeight = (total_tokens / maxTokens) * 100;

  // スタック計算（下から上へ）
  const cachedInputPct = (cached_input_tokens / total_tokens) * 100;
  const uncachedInputPct = (uncached_input_tokens / total_tokens) * 100;
  const reasoningPct = (reasoning_tokens / total_tokens) * 100;
  const finalOutputPct = (finalOutputTokens / total_tokens) * 100;

  const tooltip = [
    `${label}: 合計 ${total_tokens.toLocaleString()} tokens`,
    `• キャッシュ入力: ${cached_input_tokens.toLocaleString()}`,
    `• 通常入力: ${uncached_input_tokens.toLocaleString()}`,
    `• 推論: ${reasoning_tokens.toLocaleString()}`,
    `• 出力: ${output_tokens.toLocaleString()}`,
    `• コスト: $${total_cost.toFixed(4)}`,
  ].join('\n');

  return (
    <div className="bar" key={label} title={tooltip}>
      <div className="bar-x mono">{label}</div>
      <div
        className="bar-stack"
        style={{ height: `${Math.max(totalHeight, 1)}%` }}
      >
        <div
          className="bar-segment"
          style={{
            height: `${cachedInputPct}%`,
            backgroundColor: '#52D9A3',
          }}
        />
        <div
          className="bar-segment"
          style={{
            height: `${uncachedInputPct}%`,
            backgroundColor: '#4A7FDB',
          }}
        />
        <div
          className="bar-segment"
          style={{
            height: `${reasoningPct}%`,
            backgroundColor: '#9B6DD1',
          }}
        />
        <div
          className="bar-segment"
          style={{
            height: `${finalOutputPct}%`,
            backgroundColor: '#E67A40',
          }}
        />
      </div>
      <div className="bar-y mono">{usage.total_tokens.toLocaleString()}</div>
    </div>
  );
};

export const UsageChart: FC<{ data: UsageData[] }> = async ({ data }) => {
  const maxTokens = Math.max(
    ...data.map(({ usage }) => (usage ? usage.total_tokens : 0))
  );

  return (
    <div className="chart" role="img">
      <div className="chart-bars">
        {data.map(async ({ date, usage }) => (
          <UsageChartBar date={date} usage={usage} maxTokens={maxTokens} />
        ))}
      </div>

      <div className="usage-legend">
        <div className="legend-item">
          <div
            className="legend-color"
            style={{ backgroundColor: '#52D9A3' }}
          ></div>
          <span>キャッシュ入力</span>
        </div>
        <div className="legend-item">
          <div
            className="legend-color"
            style={{ backgroundColor: '#4A7FDB' }}
          ></div>
          <span>通常入力</span>
        </div>
        <div className="legend-item">
          <div
            className="legend-color"
            style={{ backgroundColor: '#9B6DD1' }}
          ></div>
          <span>推論</span>
        </div>
        <div className="legend-item">
          <div
            className="legend-color"
            style={{ backgroundColor: '#E67A40' }}
          ></div>
          <span>出力</span>
        </div>
      </div>
    </div>
  );
};
