import { useEffect, useRef, useState } from 'react'

import type { Citation, DocumentSummary, Turn } from '../types'
import AnswerText from './AnswerText'
import TraceInspector from './TraceInspector'

interface Props {
  turns: Turn[]
  jobs: DocumentSummary[]
  selectedJobId: string | null
  scope: 'one' | 'all'
  busy: boolean
  onScopeChange: (scope: 'one' | 'all') => void
  onSend: (question: string) => void
  onCite: (citation: Citation) => void
}

const STARTERS = [
  'What skills am I missing for this role?',
  'How does my experience align with this job?',
  'Which of these roles fits me best?',
  'What should I prepare for an interview here?',
]

export default function ChatPanel({
  turns,
  jobs,
  selectedJobId,
  scope,
  busy,
  onScopeChange,
  onSend,
  onCite,
}: Props) {
  const [draft, setDraft] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [turns])

  const submit = (text: string) => {
    const question = text.trim()
    if (!question || busy) return
    onSend(question)
    setDraft('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const selectedJob = jobs.find((job) => job.id === selectedJobId)
  const scopeLabel =
    scope === 'all'
      ? `Comparing all ${jobs.length} postings`
      : `Asking about ${selectedJob?.title ?? 'the selected role'}`

  return (
    <div className="chat">
      <div className="chat-scroll" ref={scrollRef}>
        <div className="chat-context">
          {scopeLabel}
          {jobs.length > 1 && (
            <>
              {' · '}
              <button
                className="btn-subtle"
                style={{ padding: 0, fontSize: 12 }}
                onClick={() => onScopeChange(scope === 'all' ? 'one' : 'all')}
              >
                {scope === 'all' ? 'focus on selected' : 'compare all'}
              </button>
            </>
          )}
        </div>

        {turns.length === 0 && (
          <div className="msg assistant">
            <div className="msg-body">
              <p>
                Ask me anything about how your resume lines up with these postings. Every answer
                cites the exact extracts it came from — click a citation to read the source.
              </p>
              <div className="suggestions">
                {STARTERS.map((starter) => (
                  <button key={starter} className="suggestion" onClick={() => submit(starter)}>
                    {starter}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {turns.map((turn) =>
          turn.role === 'user' ? (
            <div className="msg user" key={turn.id}>
              {turn.text}
            </div>
          ) : (
            <div className="msg assistant" key={turn.id}>
              <div className="msg-body">
                {turn.pending ? (
                  <div className="typing" aria-label="Thinking">
                    <span />
                    <span />
                    <span />
                  </div>
                ) : (
                  <>
                    {turn.grounded === false && (
                      <div className="warn-banner">
                        <span>⚠</span>
                        <span>
                          Most of this answer isn’t tied to a specific extract. Treat it as a
                          suggestion and check it against the sources below.
                        </span>
                      </div>
                    )}

                    <AnswerText
                      text={turn.text}
                      citations={turn.citations ?? []}
                      onCite={onCite}
                    />

                    {turn.citations && turn.citations.length > 0 && (
                      <div className="msg-footer">
                        {turn.citations.map((citation) => (
                          <button
                            key={citation.marker}
                            className="source-chip"
                            onClick={() => onCite(citation)}
                          >
                            <span className="marker">{citation.marker}</span>
                            {citation.document_kind === 'resume' ? 'Resume' : citation.document_title}
                            <span className="faint">· {citation.section}</span>
                          </button>
                        ))}
                      </div>
                    )}

                    {turn.suggestions && turn.suggestions.length > 0 && (
                      <div className="suggestions">
                        {turn.suggestions.map((suggestion) => (
                          <button
                            key={suggestion}
                            className="suggestion"
                            onClick={() => submit(suggestion)}
                          >
                            {suggestion}
                          </button>
                        ))}
                      </div>
                    )}

                    {turn.trace && <TraceInspector trace={turn.trace} />}
                  </>
                )}
              </div>
            </div>
          ),
        )}
      </div>

      <div className="composer">
        <div className="composer-box">
          <textarea
            ref={textareaRef}
            rows={1}
            value={draft}
            placeholder="Ask about fit, gaps, alignment, or interview prep…"
            onChange={(event) => {
              setDraft(event.target.value)
              const el = event.target
              el.style.height = 'auto'
              el.style.height = `${Math.min(el.scrollHeight, 160)}px`
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                submit(draft)
              }
            }}
          />
          <button
            className="btn btn-primary"
            onClick={() => submit(draft)}
            disabled={busy || !draft.trim()}
          >
            {busy ? 'Thinking…' : 'Ask'}
          </button>
        </div>
        <p className="composer-hint">
          <span>
            <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line
          </span>
          <span>Answers are grounded in your uploaded documents only.</span>
        </p>
      </div>
    </div>
  )
}
