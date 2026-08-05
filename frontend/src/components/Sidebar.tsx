import { useRef, useState } from 'react'

import { band, requiredGaps } from '../format'
import type { DocumentSummary, FitReport } from '../types'

interface DropZoneProps {
  label: string
  hint: string
  multiple?: boolean
  busy?: boolean
  onFiles: (files: File[]) => void
}

function DropZone({ label, hint, multiple, busy, onFiles }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const handle = (files: FileList | null) => {
    if (!files?.length) return
    onFiles(Array.from(files))
    // Reset so re-picking the same file still fires a change event.
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <label
      className={`dropzone${dragging ? ' dragging' : ''}`}
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        handle(event.dataTransfer.files)
      }}
    >
      <strong>{busy ? 'Processing…' : label}</strong>
      <span>{hint}</span>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md"
        multiple={multiple}
        disabled={busy}
        onChange={(event) => handle(event.target.files)}
      />
    </label>
  )
}

interface Props {
  resume: DocumentSummary | null
  jobs: DocumentSummary[]
  fits: Record<string, FitReport>
  selectedJobId: string | null
  busy: boolean
  onSelectJob: (jobId: string) => void
  onUploadResume: (files: File[]) => void
  onUploadJobs: (files: File[]) => void
  onRemove: (documentId: string) => void
  onLoadSamples: () => void
}

export default function Sidebar({
  resume,
  jobs,
  fits,
  selectedJobId,
  busy,
  onSelectJob,
  onUploadResume,
  onUploadJobs,
  onRemove,
  onLoadSamples,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-section">
        <h3>Your resume</h3>
        {resume ? (
          <div className="doc-card selected">
            <div className="doc-card-body">
              <div className="doc-card-title">{resume.filename}</div>
              <div className="doc-card-meta">
                {resume.word_count.toLocaleString()} words · {resume.chunk_count} chunks
              </div>
            </div>
            <button
              className="doc-remove"
              onClick={() => onRemove(resume.id)}
              aria-label="Remove resume"
              title="Remove resume"
            >
              ×
            </button>
          </div>
        ) : (
          <DropZone
            label="Upload your resume"
            hint="PDF, DOCX, TXT or MD"
            busy={busy}
            onFiles={onUploadResume}
          />
        )}
        {resume && (
          <DropZone label="Replace resume" hint="Uploading again swaps it" busy={busy} onFiles={onUploadResume} />
        )}
      </div>

      <div className="sidebar-section">
        <h3>Job descriptions {jobs.length > 0 && `(${jobs.length})`}</h3>

        {jobs.map((job) => {
          const report = fits[job.id]
          return (
            <div
              key={job.id}
              className={`doc-card${job.id === selectedJobId ? ' selected' : ''}`}
              onClick={() => onSelectJob(job.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  onSelectJob(job.id)
                }
              }}
            >
              <div className="doc-card-body">
                <div className="doc-card-title">{job.title}</div>
                <div className="doc-card-meta">
                  {report
                    ? `${requiredGaps(report)} required gap${requiredGaps(report) === 1 ? '' : 's'}`
                    : `${job.chunk_count} chunks`}
                </div>
              </div>
              {report && (
                <span className={`score-pill ${band(report.overall_score)}`}>
                  {Math.round(report.overall_score)}
                </span>
              )}
              <button
                className="doc-remove"
                onClick={(event) => {
                  event.stopPropagation()
                  onRemove(job.id)
                }}
                aria-label={`Remove ${job.title}`}
                title="Remove"
              >
                ×
              </button>
            </div>
          )
        })}

        <DropZone
          label={jobs.length ? 'Add another posting' : 'Upload job descriptions'}
          hint="Several at once is fine"
          multiple
          busy={busy}
          onFiles={onUploadJobs}
        />
      </div>

      {!resume && !jobs.length && (
        <button className="btn btn-ghost" onClick={onLoadSamples} disabled={busy}>
          Load sample documents
        </button>
      )}
    </aside>
  )
}
