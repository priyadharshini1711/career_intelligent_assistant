/**
 * Renders a model answer.
 *
 * A deliberately small markdown subset -- headings, bullets, bold, inline
 * code, italics -- plus the one thing that actually matters here: turning
 * `[R1]` and `[J2]` markers into clickable citation chips that open the source
 * they came from.
 *
 * Why hand-rolled rather than react-markdown: that pulls in remark and a
 * plugin chain (~40KB) to render six constructs, and the citation behaviour
 * would need a custom plugin on top anyway. Rendering into React elements also
 * means no `dangerouslySetInnerHTML`, so model output can never inject markup
 * -- worth having when the text is generated from documents a stranger uploaded.
 */

import { Fragment, type ReactNode } from 'react'

import type { Citation } from '../types'

const CITATION = /\[([RJ]\d+)\]/g

// Bold, inline code, and asterisk italics. Underscore italics (`_like this_`)
// are deliberately unsupported: this domain is full of snake_case identifiers
// (`LLM_PROVIDER`, `GEMINI_API_KEY`), and treating their underscores as
// emphasis delimiters shredded the text. Asterisks have no such collision.
// `**` must precede `*` in the alternation or bold matches as two italics.
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*)/g

interface Props {
  text: string
  citations: Citation[]
  onCite: (citation: Citation) => void
}

/** Split a line into inline spans, then wire up citation markers. */
function renderInline(
  line: string,
  citations: Citation[],
  onCite: (citation: Citation) => void,
  keyPrefix: string,
): ReactNode[] {
  const byMarker = new Map(citations.map((citation) => [citation.marker, citation]))

  return line.split(INLINE).map((part, index) => {
    const key = `${keyPrefix}-${index}`

    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return <strong key={key}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return <code key={key}>{part.slice(1, -1)}</code>
    }
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      return <em key={key}>{part.slice(1, -1)}</em>
    }

    // Remaining plain text: pull out citation markers.
    const pieces: ReactNode[] = []
    let cursor = 0
    for (const match of part.matchAll(CITATION)) {
      const marker = match[1]
      const citation = byMarker.get(marker)
      if (match.index === undefined) continue

      if (match.index > cursor) pieces.push(part.slice(cursor, match.index))
      cursor = match.index + match[0].length

      if (citation) {
        pieces.push(
          <button
            key={`${key}-${marker}-${match.index}`}
            className="cite"
            onClick={() => onCite(citation)}
            title={`${citation.document_title} — ${citation.section}`}
          >
            {marker}
          </button>,
        )
      } else {
        // Guardrails strip unsupplied markers server-side, so this should not
        // happen. If it ever does, show the text rather than swallow it.
        pieces.push(match[0])
      }
    }
    if (cursor < part.length) pieces.push(part.slice(cursor))

    return <Fragment key={key}>{pieces}</Fragment>
  })
}

export default function AnswerText({ text, citations, onCite }: Props) {
  const blocks: ReactNode[] = []
  let bullets: string[] = []

  const flushBullets = () => {
    if (!bullets.length) return
    const items = bullets
    blocks.push(
      <ul key={`ul-${blocks.length}`}>
        {items.map((item, index) => (
          <li key={index}>{renderInline(item, citations, onCite, `li-${blocks.length}-${index}`)}</li>
        ))}
      </ul>,
    )
    bullets = []
  }

  text.split('\n').forEach((rawLine, index) => {
    const line = rawLine.trimEnd()

    if (!line.trim()) {
      flushBullets()
      return
    }

    const bullet = line.match(/^\s*[-*•]\s+(.*)$/)
    if (bullet) {
      bullets.push(bullet[1])
      return
    }

    flushBullets()

    const heading = line.match(/^(#{1,4})\s+(.*)$/)
    if (heading) {
      blocks.push(
        <h4 key={`h-${index}`}>{renderInline(heading[2], citations, onCite, `h-${index}`)}</h4>,
      )
      return
    }

    // A whole line wrapped in ** reads as a heading in practice; models use it
    // that way constantly and rendering it as a paragraph loses the structure.
    const boldLine = line.match(/^\*\*(.+)\*\*:?$/)
    if (boldLine) {
      blocks.push(
        <h4 key={`b-${index}`}>{renderInline(boldLine[1], citations, onCite, `b-${index}`)}</h4>,
      )
      return
    }

    blocks.push(<p key={`p-${index}`}>{renderInline(line, citations, onCite, `p-${index}`)}</p>)
  })

  flushBullets()

  return <>{blocks}</>
}
