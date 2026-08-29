/** Cloudflare platform 上限へ達する前に application が停止する既定値。 */
export const DEFAULT_EXTERNAL_REQUEST_BUDGET = 40;

/** 外部 request 予算の読み取り専用 snapshot。 */
export interface ExternalRequestBudgetSnapshot {
  limit: number;
  used: number;
  remaining: number;
}

/** 外部 request を追加すると共有上限を超える場合の terminal error。 */
export class ExternalRequestBudgetError extends Error {
  readonly code = 'external_request_budget_exceeded' as const;

  constructor(
    readonly limit: number,
    readonly used: number,
    readonly requested: number
  ) {
    super(
      `External request budget exceeded: requested ${requested} with ${used}/${limit} already used`
    );
    this.name = 'ExternalRequestBudgetError';
  }
}

/**
 * alarm / manual run 全体で共有する、同期的な外部 request admission gate。
 *
 * provider へ送信する直前に reserve することで、並列 request でも上限判定と
 * 消費を同じ JavaScript turn 内で確定する。既定40件は Cloudflare Free の
 * 50 external subrequest 上限に対して、session lifecycle と失敗通知のため
 * 10件を残す。
 */
export class ExternalRequestBudget {
  private used = 0;

  constructor(readonly limit = DEFAULT_EXTERNAL_REQUEST_BUDGET) {
    if (!Number.isSafeInteger(limit) || limit <= 0) {
      throw new Error(
        'External request budget limit must be a positive integer'
      );
    }
  }

  /** request を送信前に予約し、上限超過なら消費せず拒否する。 */
  reserve(count = 1): void {
    if (!Number.isSafeInteger(count) || count <= 0) {
      throw new Error('External request count must be a positive integer');
    }
    if (this.used + count > this.limit) {
      throw new ExternalRequestBudgetError(this.limit, this.used, count);
    }

    this.used += count;
  }

  /** observability と test 用の immutable snapshot を返す。 */
  snapshot(): ExternalRequestBudgetSnapshot {
    return {
      limit: this.limit,
      used: this.used,
      remaining: this.limit - this.used,
    };
  }
}
