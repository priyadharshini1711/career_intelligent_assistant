/**
 * Application shell and the only place that holds server state.
 *
 * No state library. There is one server, one session, and about six pieces of
 * state; `useState` plus a couple of helpers is less code than the setup for
 * Redux or Zustand would be, and the data flow stays readable top to bottom.
 * If this grew a second route or optimistic updates, that calculus changes.
 */

import { useCallback, useEffect, useState } from 'react'

import { api, clearSessionId, getSessionId } from './api'
import ChatPanel from './components/ChatPanel'
import EmptyState from './components/EmptyState'
import FitPanel from './components/FitPanel'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import SourceDrawer from './components/SourceDrawer'
import Toasts, { type Toast } from './components/Toasts'
import { ApiError } from './types'
import type { Citation, FitReport, ReadyState, SessionState, Turn } from './types'

type Tab = 'fit' | 'chat'

let turnCounter = 0
const nextId = () => `t${++turnCounter}`

export default function App() {
  const [ready, setReady] = useState<ReadyState | null>(null)
  const [state, setState] = useState<SessionState | null>(null)
  const [fits, setFits] = useState<Record<string, FitReport>>({})
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('fit')
  const [scope, setScope] = useState<'one' | 'all'>('one')
  const [turns, setTurns] = useState<Turn[]>([])
  const [drawer, setDrawer] = useState<Citation | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [busy, setBusy] = useState(false)
  const [asking, setAsking] = useState(false)
  const [booting, setBooting] = useState(true)

  const toast = useCallback((tone: Toast['tone'], title: string, detail?: string) => {
    const id = nextId()
    setToasts((current) => [...current, { id, tone, title, detail }])
    window.setTimeout(() => setToasts((current) => current.filter((t) => t.id !== id)), 7000)
  }, [])

  const reportError = useCallback(
    (error: unknown, fallback: string) => {
      if (error instanceof ApiError) {
        toast('error', error.message)
      } else {
        toast('error', fallback, error instanceof Error ? error.message : undefined)
      }
    },
    [toast],
  )

  /** Refresh fit reports for the current documents. */
  const refreshFits = useCallback(
    async (next: SessionState) => {
      if (!next.ready) {
        setFits({})
        return
      }
      try {
        const reports = await api.allFits()
        setFits(Object.fromEntries(reports.map((report) => [report.job_id, report])))
      } catch (error) {
        reportError(error, 'Could not compute the fit reports.')
      }
    },
    [reportError],
  )

  const applyState = useCallback(
    (next: SessionState) => {
      setState(next)
      setSelectedJobId((current) => {
        if (current && next.jobs.some((job) => job.id === current)) return current
        return next.jobs[0]?.id ?? null
      })
      void refreshFits(next)
    },
    [refreshFits],
  )

  // Boot: read provider health, and restore the session if the browser has one.
  useEffect(() => {
    let cancelled = false

    const boot = async () => {
      try {
        const health = await api.ready()
        if (!cancelled) setReady(health)
      } catch {
        // /ready failing is not fatal; the badge just shows "loading".
      }

      if (getSessionId()) {
        try {
          const restored = await api.state()
          if (!cancelled) applyState(restored)
        } catch (error) {
          // An expired session on the server is the normal case here, not an
          // error worth shouting about -- drop it and start clean.
          if (error instanceof ApiError && error.status === 404) clearSessionId()
        }
      }
      if (!cancelled) setBooting(false)
    }

    void boot()
    return () => {
      cancelled = true
    }
  }, [applyState])

  const withBusy = async (work: () => Promise<void>) => {
    setBusy(true)
    try {
      await work()
    } finally {
      setBusy(false)
    }
  }

  const handleUploadResume = (files: File[]) =>
    withBusy(async () => {
      try {
        const result = await api.uploadResume(files[0])
        applyState(result.state)
        toast('info', 'Resume added', `${result.uploaded[0]?.chunk_count ?? 0} chunks indexed.`)
      } catch (error) {
        reportError(error, 'Could not read that resume.')
      }
    })

  const handleUploadJobs = (files: File[]) =>
    withBusy(async () => {
      try {
        const result = await api.uploadJobs(files)
        applyState(result.state)
        if (result.uploaded.length) {
          toast('info', `Added ${result.uploaded.length} job description${result.uploaded.length === 1 ? '' : 's'}`)
        }
        // Partial failures are reported per file rather than failing the batch.
        result.skipped.forEach((skip) => toast('error', `Skipped ${skip.filename}`, skip.reason))
      } catch (error) {
        reportError(error, 'Could not read those job descriptions.')
      }
    })

  const handleLoadSamples = () =>
    withBusy(async () => {
      try {
        const result = await api.loadSamples()
        applyState(result.state)
        toast('info', 'Sample documents loaded', 'One resume and three job descriptions.')
      } catch (error) {
        reportError(error, 'Could not load the sample documents.')
      }
    })

  const handleRemove = (documentId: string) =>
    withBusy(async () => {
      try {
        applyState(await api.deleteDocument(documentId))
      } catch (error) {
        reportError(error, 'Could not remove that document.')
      }
    })

  const handleReset = () => {
    clearSessionId()
    setState(null)
    setFits({})
    setTurns([])
    setSelectedJobId(null)
    setTab('fit')
    toast('info', 'Session cleared', 'Your documents were only ever held in memory.')
  }

  const handleSend = async (question: string) => {
    const pendingId = nextId()
    setTab('chat')
    setTurns((current) => [
      ...current,
      { id: nextId(), role: 'user', text: question },
      { id: pendingId, role: 'assistant', text: '', pending: true },
    ])
    setAsking(true)

    try {
      const response = await api.ask(question, scope === 'all' ? null : selectedJobId)
      setTurns((current) =>
        current.map((turn) =>
          turn.id === pendingId
            ? {
                ...turn,
                pending: false,
                text: response.answer,
                citations: response.citations,
                suggestions: response.suggestions,
                grounded: response.grounded,
                refused: response.refused,
                trace: response.trace,
              }
            : turn,
        ),
      )
    } catch (error) {
      const message =
        error instanceof ApiError
          ? error.message
          : 'Something went wrong reaching the assistant. Please try again.'
      setTurns((current) =>
        current.map((turn) =>
          turn.id === pendingId ? { ...turn, pending: false, error: true, text: message } : turn,
        ),
      )
      reportError(error, 'The assistant could not answer.')
    } finally {
      setAsking(false)
    }
  }

  const askAbout = (question: string) => {
    setScope('one')
    void handleSend(question)
  }

  if (booting) {
    return (
      <div className="loading-page">
        <div>
          <div className="spinner" />
          Starting up…
        </div>
      </div>
    )
  }

  const selectedReport = selectedJobId ? fits[selectedJobId] : undefined
  const canAsk = Boolean(state?.ready)

  return (
    <div className="app">
      <Header ready={ready} state={state} onReset={handleReset} />

      <div className="app-body">
        <Sidebar
          resume={state?.resume ?? null}
          jobs={state?.jobs ?? []}
          fits={fits}
          selectedJobId={selectedJobId}
          busy={busy}
          onSelectJob={(jobId) => {
            setSelectedJobId(jobId)
            setScope('one')
          }}
          onUploadResume={handleUploadResume}
          onUploadJobs={handleUploadJobs}
          onRemove={handleRemove}
          onLoadSamples={handleLoadSamples}
        />

        <main className="main">
          {!canAsk ? (
            <EmptyState state={state} busy={busy} onLoadSamples={handleLoadSamples} />
          ) : (
            <>
              <nav className="tabs">
                <button
                  className={`tab${tab === 'fit' ? ' active' : ''}`}
                  onClick={() => setTab('fit')}
                >
                  Fit breakdown
                </button>
                <button
                  className={`tab${tab === 'chat' ? ' active' : ''}`}
                  onClick={() => setTab('chat')}
                >
                  Ask a question
                </button>
              </nav>

              <div className="tab-panel">
                {tab === 'fit' ? (
                  selectedReport ? (
                    <FitPanel report={selectedReport} onAskAbout={askAbout} />
                  ) : (
                    <div className="loading-page">
                      <div>
                        <div className="spinner" />
                        Scoring your fit…
                      </div>
                    </div>
                  )
                ) : (
                  <ChatPanel
                    turns={turns}
                    jobs={state?.jobs ?? []}
                    selectedJobId={selectedJobId}
                    scope={scope}
                    busy={asking}
                    onScopeChange={setScope}
                    onSend={handleSend}
                    onCite={setDrawer}
                  />
                )}
              </div>
            </>
          )}
        </main>
      </div>

      {drawer && <SourceDrawer citation={drawer} onClose={() => setDrawer(null)} />}
      <Toasts toasts={toasts} onDismiss={(id) => setToasts((c) => c.filter((t) => t.id !== id))} />
    </div>
  )
}
