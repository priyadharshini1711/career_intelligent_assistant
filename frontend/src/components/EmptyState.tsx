import type { SessionState } from '../types'

interface Props {
  state: SessionState | null
  busy: boolean
  onLoadSamples: () => void
}

export default function EmptyState({ state, busy, onLoadSamples }: Props) {
  const hasResume = Boolean(state?.resume)
  const hasJobs = Boolean(state?.jobs.length)

  return (
    <div className="empty">
      <div className="empty-inner">
        <h2>Let’s see where you stand</h2>
        <p>
          Upload your resume and the roles you’re considering. You’ll get a scored breakdown of
          where you fit, where the gaps are, and a chat that answers from your documents only.
        </p>

        <ol className="empty-steps">
          <li>
            <span className={`step-num${hasResume ? ' done' : ''}`}>{hasResume ? '✓' : '1'}</span>
            <span>
              <strong style={{ color: 'var(--text)' }}>Add your resume.</strong> PDF, DOCX, TXT or
              Markdown. It stays in memory for this session and is never written to disk.
            </span>
          </li>
          <li>
            <span className={`step-num${hasJobs ? ' done' : ''}`}>{hasJobs ? '✓' : '2'}</span>
            <span>
              <strong style={{ color: 'var(--text)' }}>Add job descriptions.</strong> Up to ten, so
              you can compare roles against each other rather than one at a time.
            </span>
          </li>
          <li>
            <span className="step-num">3</span>
            <span>
              <strong style={{ color: 'var(--text)' }}>Ask.</strong> “What am I missing?”, “How do I
              align?”, “What will they probe in the interview?”
            </span>
          </li>
        </ol>

        {!hasResume && !hasJobs && (
          <button className="btn btn-primary" onClick={onLoadSamples} disabled={busy}>
            {busy ? 'Loading…' : 'Try it with sample documents'}
          </button>
        )}
      </div>
    </div>
  )
}
