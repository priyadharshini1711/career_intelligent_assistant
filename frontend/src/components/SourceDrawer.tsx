import { useEffect } from 'react'

import type { Citation } from '../types'

interface Props {
  citation: Citation
  onClose: () => void
}

export default function SourceDrawer({ citation, onClose }: Props) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Source extract">
        <div className="drawer-head">
          <div style={{ flex: 1 }}>
            <h3>
              <span className={`kind-tag ${citation.document_kind}`}>{citation.document_kind}</span>{' '}
              {citation.document_title}
            </h3>
            <div className="sub">
              {citation.section} · cited as {citation.marker}
            </div>
          </div>
          <button className="btn-subtle" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="drawer-body">
          <div className="snippet">{citation.snippet}</div>
          <dl className="drawer-meta">
            <dt>Retrieval score</dt>
            <dd>{citation.score.toFixed(4)}</dd>
            <dt>Chunk</dt>
            <dd className="mono">{citation.chunk_id}</dd>
          </dl>
          <p className="faint" style={{ fontSize: 12, marginTop: 16 }}>
            This is the exact text the model was given for this claim. If the answer says
            something this extract does not support, that is a hallucination worth reporting.
          </p>
        </div>
      </aside>
    </>
  )
}
