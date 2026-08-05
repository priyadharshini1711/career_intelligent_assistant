/** Small shared formatting helpers. */

import type { FitReport } from './types'

export type Band = 'strong' | 'good' | 'partial' | 'weak'

/**
 * Score bands. The thresholds mirror `_verdict` in `backend/app/analysis/fit.py`
 * -- duplicated deliberately rather than shipped in the API response, because
 * the backend owns the words ("Strong match") and the frontend owns the colour.
 */
export function band(score: number): Band {
  if (score >= 78) return 'strong'
  if (score >= 62) return 'good'
  if (score >= 45) return 'partial'
  return 'weak'
}

export function bandColor(value: Band): string {
  return {
    strong: 'var(--strong)',
    good: 'var(--accent)',
    partial: 'var(--partial)',
    weak: 'var(--weak)',
  }[value]
}

export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

export function requiredGaps(report: FitReport): number {
  return report.missing_skills.filter((gap) => gap.importance === 'required').length
}
