/** Mirrors the Pydantic models in `backend/app/schemas.py`. */

export type DocumentKind = 'resume' | 'job'

export interface DocumentSummary {
  id: string
  kind: DocumentKind
  title: string
  filename: string
  word_count: number
  chunk_count: number
  created_at: string
}

export interface SessionState {
  session_id: string
  resume: DocumentSummary | null
  jobs: DocumentSummary[]
  llm_provider: string
  llm_model: string
  ready: boolean
}

export interface UploadResponse {
  session_id: string
  uploaded: DocumentSummary[]
  skipped: { filename: string; code: string; reason: string }[]
  state: SessionState
}

export interface Citation {
  marker: string
  chunk_id: string
  document_id: string
  document_kind: DocumentKind
  document_title: string
  section: string
  snippet: string
  score: number
}

export interface TraceStage {
  name: string
  duration_ms: number
  attributes: Record<string, unknown>
}

export interface Trace {
  request_id: string
  total_ms: number
  stages: TraceStage[]
  attributes: Record<string, unknown>
}

export interface ChatResponse {
  answer: string
  citations: Citation[]
  suggestions: string[]
  grounded: boolean
  refused: boolean
  trace: Trace | null
}

export interface SkillEvidence {
  skill: string
  resume_snippet: string | null
  similarity: number
}

export interface SkillGap {
  skill: string
  importance: string
  reason: string
}

export interface FitComponent {
  name: string
  score: number
  weight: number
  explanation: string
}

export interface FitReport {
  job_id: string
  job_title: string
  overall_score: number
  verdict: string
  components: FitComponent[]
  matched_skills: SkillEvidence[]
  missing_skills: SkillGap[]
  partial_skills: SkillEvidence[]
  generated_at: string
}

export interface ReadyState {
  status: string
  llm: Record<string, unknown>
  configured_provider: string
  embedding_model: string
  embedding_dimension: number
  note?: string
}

/** A chat turn as the UI holds it. */
export interface Turn {
  id: string
  role: 'user' | 'assistant'
  text: string
  citations?: Citation[]
  suggestions?: string[]
  grounded?: boolean
  refused?: boolean
  trace?: Trace | null
  pending?: boolean
  error?: boolean
}

export class ApiError extends Error {
  code: string
  status: number
  details: Record<string, unknown>

  constructor(message: string, code: string, status: number, details: Record<string, unknown> = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.details = details
  }
}
