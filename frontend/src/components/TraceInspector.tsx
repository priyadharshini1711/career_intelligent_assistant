/**
 * "How this answer was built."
 *
 * Exposing the retrieval trace in the product, not just the logs, is a
 * deliberate choice. A RAG answer is only as trustworthy as its evidence, and
 * the honest way to earn that trust is to show which chunks were retrieved,
 * from where, and how strongly each channel scored them. It doubles as the
 * debugging surface that caught most of the retrieval bugs in this project.
 */

import { formatMs } from '../format'
import type { Trace } from '../types'

interface Source {
  chunk_id: string
  kind: string
  document: string
  section: string
  score: number
  dense: number
  lexical: number
}

export default function TraceInspector({ trace }: { trace: Trace }) {
  const stages = trace.stages ?? []
  const slowest = Math.max(...stages.map((stage) => stage.duration_ms), 1)

  const retrieve = stages.find((stage) => stage.name === 'retrieve')
  const sources = (retrieve?.attributes.sources as Source[] | undefined) ?? []

  const context = stages.find((stage) => stage.name === 'build_context')?.attributes ?? {}
  const generate = stages.find((stage) => stage.name === 'generate')?.attributes ?? {}
  const guardrail = stages.find((stage) => stage.name === 'guardrail_output')?.attributes ?? {}

  const grounding = guardrail.grounding_ratio as number | undefined
  const invalid = (guardrail.invalid_markers as string[] | undefined) ?? []

  return (
    <details className="trace">
      <summary>
        How this answer was built · {formatMs(trace.total_ms)} · {sources.length} chunks retrieved
      </summary>

      <div className="trace-body">
        <div className="stage-bars">
          {stages.map((stage) => (
            <div className="stage-bar" key={stage.name}>
              <span>{stage.name.replace(/_/g, ' ')}</span>
              <div className="track">
                <div className="fill" style={{ width: `${(stage.duration_ms / slowest) * 100}%` }} />
              </div>
              <span className="ms">{formatMs(stage.duration_ms)}</span>
            </div>
          ))}
        </div>

        {sources.length > 0 && (
          <table className="trace-table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Section</th>
                <th style={{ textAlign: 'right' }}>Dense</th>
                <th style={{ textAlign: 'right' }}>BM25</th>
              </tr>
            </thead>
            <tbody>
              {sources.map((source) => (
                <tr key={source.chunk_id}>
                  <td>
                    <span className={`kind-tag ${source.kind}`}>{source.kind}</span>{' '}
                    {source.document}
                  </td>
                  <td>{source.section}</td>
                  <td className="num">{source.dense.toFixed(3)}</td>
                  <td className="num">{source.lexical.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <dl className="drawer-meta">
          <dt>Intent</dt>
          <dd>{String(trace.attributes.intent ?? '—')}</dd>

          <dt>Context sent</dt>
          <dd>
            {String(context.context_words ?? '—')} words across{' '}
            {String(context.chunks_in_prompt ?? '—')} chunks
            {Number(context.dropped_chunks ?? 0) > 0 &&
              ` (${context.dropped_chunks} dropped to fit the budget)`}
          </dd>

          <dt>Model</dt>
          <dd>
            {String(generate.provider ?? '—')} · {String(generate.model ?? '—')}
            {generate.input_tokens != null &&
              ` · ${generate.input_tokens} in / ${generate.output_tokens ?? '—'} out`}
          </dd>

          {grounding !== undefined && (
            <>
              <dt>Grounding</dt>
              <dd>
                {Math.round(grounding * 100)}% of substantial sentences carry a citation
                {invalid.length > 0 && ` · ${invalid.length} fabricated citation(s) stripped`}
              </dd>
            </>
          )}

          <dt>Request</dt>
          <dd className="mono">{trace.request_id}</dd>
        </dl>
      </div>
    </details>
  )
}
