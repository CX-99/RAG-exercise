import type { MouseEvent, ReactNode } from 'react'
import Markdown from 'react-markdown'
import { linkCitations } from './citations'

/**
 * Elements a response may render. Images are deliberately left out: a prompt-injected
 * source document could make the model emit an image whose URL leaks data as soon as the
 * browser loads it. Links are allowed only so citation markers can become chips; any
 * other link is shown as plain text (see {@link Link}). Disallowed elements are
 * unwrapped, so their text still shows.
 */
const ALLOWED_ELEMENTS = [
  'p',
  'br',
  'strong',
  'em',
  'ul',
  'ol',
  'li',
  'code',
  'pre',
  'blockquote',
  'a',
]

const SOURCE_ANCHOR = /^#source-(\d+)$/

/** Open and scroll to the source card a citation chip points at. */
function revealSource(event: MouseEvent<HTMLAnchorElement>, id: string) {
  const card = document.getElementById(id)
  if (card instanceof HTMLDetailsElement) {
    event.preventDefault()
    card.open = true
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }
}

/** A citation chip for internal source links; any other link renders as plain text. */
function Link({ href, children }: { href?: string; children?: ReactNode }) {
  const match = href ? SOURCE_ANCHOR.exec(href) : null
  if (!match) return <span>{children}</span>
  const id = `source-${match[1]}`
  return (
    <a
      href={`#${id}`}
      className="citation-chip"
      aria-label={`Source ${match[1]}`}
      onClick={(event) => revealSource(event, id)}
    >
      {match[1]}
    </a>
  )
}

/**
 * Render response text as a safe subset of Markdown, with citation markers as chips.
 *
 * Models format answers with Markdown such as `**bold**` and lists. react-markdown builds
 * React elements rather than injecting HTML, and `skipHtml` drops any raw HTML.
 */
export function AnswerText({ text }: { text: string }) {
  return (
    <div className="answer-text">
      <Markdown
        allowedElements={ALLOWED_ELEMENTS}
        unwrapDisallowed
        skipHtml
        components={{ a: ({ href, children }) => <Link href={href}>{children}</Link> }}
      >
        {linkCitations(text)}
      </Markdown>
    </div>
  )
}
