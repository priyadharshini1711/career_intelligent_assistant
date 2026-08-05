/**
 * API client.
 *
 * Two things it centralises: the session header (every call carries it, and
 * the server's response can rotate it), and error shape. The backend returns
 * `{code, message, details}` for every failure, so unwrapping it in one place
 * means components can show the server's own message instead of inventing
 * "Something went wrong".
 */

import { ApiError } from './types'
import type {
  ChatResponse,
  FitReport,
  ReadyState,
  SessionState,
  UploadResponse,
} from './types'

const SESSION_KEY = 'cia.session_id'

export function getSessionId(): string | null {
  return localStorage.getItem(SESSION_KEY)
}

export function setSessionId(id: string): void {
  localStorage.setItem(SESSION_KEY, id)
}

export function clearSessionId(): void {
  localStorage.removeItem(SESSION_KEY)
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const sessionId = getSessionId()
  return sessionId ? { ...extra, 'X-Session-Id': sessionId } : extra
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T
  }

  let code = 'http_error'
  let message = `Request failed (${response.status})`
  let details: Record<string, unknown> = {}

  try {
    const body = await response.json()
    code = body.code ?? code
    message = body.message ?? message
    details = body.details ?? {}
  } catch {
    // Non-JSON error body (a proxy timeout, say). Keep the generic message.
  }

  throw new ApiError(message, code, response.status, details)
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { ...init, headers: headers(init.headers as never) })
  return unwrap<T>(response)
}

/** Persist the session id the server assigned or confirmed. */
function remember(result: UploadResponse): UploadResponse {
  setSessionId(result.session_id)
  return result
}

export const api = {
  async ready(): Promise<ReadyState> {
    // /ready answers 503 when degraded, which is information, not a failure.
    const response = await fetch('/api/system/ready')
    return (await response.json()) as ReadyState
  },

  async state(): Promise<SessionState> {
    return request<SessionState>('/api/documents')
  },

  async uploadResume(file: File): Promise<UploadResponse> {
    const form = new FormData()
    form.append('file', file)
    return remember(
      await request<UploadResponse>('/api/documents/resume', { method: 'POST', body: form }),
    )
  },

  async uploadJobs(files: File[]): Promise<UploadResponse> {
    const form = new FormData()
    files.forEach((file) => form.append('files', file))
    return remember(
      await request<UploadResponse>('/api/documents/jobs', { method: 'POST', body: form }),
    )
  },

  async loadSamples(): Promise<UploadResponse> {
    return remember(await request<UploadResponse>('/api/documents/samples', { method: 'POST' }))
  },

  async deleteDocument(documentId: string): Promise<SessionState> {
    return request<SessionState>(`/api/documents/${documentId}`, { method: 'DELETE' })
  },

  async documentText(documentId: string): Promise<{ id: string; title: string; text: string }> {
    return request(`/api/documents/${documentId}/text`)
  },

  async ask(question: string, jobId: string | null): Promise<ChatResponse> {
    return request<ChatResponse>('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, job_id: jobId, include_trace: true }),
    })
  },

  async clearHistory(): Promise<void> {
    await request('/api/chat/history', { method: 'DELETE' })
  },

  async fit(jobId: string): Promise<FitReport> {
    return request<FitReport>(`/api/analysis/fit/${jobId}`)
  },

  async allFits(): Promise<FitReport[]> {
    return request<FitReport[]>('/api/analysis/fit')
  },
}
