import type { ReadyState, SessionState } from '../types'

interface Props {
  ready: ReadyState | null
  state: SessionState | null
  onReset: () => void
}

export default function Header({ ready, state, onReset }: Props) {
  const degraded = ready ? ready.status !== 'ok' : false
  const model = state?.llm_model ?? (ready?.llm.model as string | undefined) ?? 'loading…'
  const provider = state?.llm_provider ?? (ready?.llm.provider as string | undefined) ?? ''

  return (
    <header className="header">
      <div className="brand">
        <span className="brand-mark">C</span>
        <span>
          Career Intelligence Assistant
          <small>Resume ↔ job description analysis, grounded in your files</small>
        </span>
      </div>

      <div className="header-spacer" />

      <span
        className={`provider-badge${degraded ? ' degraded' : ''}`}
        title={
          degraded
            ? (ready?.note ??
              'Running on the offline stub model — set an API key for real generated answers.')
            : `Generation: ${provider} · Embeddings: ${ready?.embedding_model ?? 'local'}`
        }
      >
        <span className="dot" />
        {provider === 'stub' ? 'Offline stub model' : model}
      </span>

      {state && (state.resume || state.jobs.length > 0) && (
        <button className="btn btn-ghost" onClick={onReset}>
          Start over
        </button>
      )}
    </header>
  )
}
