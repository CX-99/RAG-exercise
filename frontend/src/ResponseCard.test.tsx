import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ResponseCard } from './ResponseCard'
import type { AnswerKind, AskResponse, Citation } from './types'

const citation: Citation = {
  doc_id: 'annual-leave',
  title: 'Annual Leave Policy',
  section: '3. Annual Leave > 3.2 Carry-over',
  snippet: 'Employees may carry over up to 5 unused annual leave days.',
  score: 0.9,
}

function render(kind: AnswerKind, answer: string, citations: Citation[] = []) {
  const result: AskResponse = { answer, kind, citations, request_id: 'req-1234abcd' }
  return renderToStaticMarkup(<ResponseCard result={result} onAsk={() => {}} />)
}

describe('ResponseCard', () => {
  it('labels a policy answer, links its chip to a collapsible source card', () => {
    const html = render('answer', 'You can carry over 5 days [1].', [citation])
    expect(html).toContain('From company policies · 1 source')
    expect(html).toContain('href="#source-1"')
    expect(html).toContain('<details id="source-1">')
    expect(html).toContain('Annual Leave Policy')
    expect(html).toContain('3. Annual Leave &gt; 3.2 Carry-over')
  })

  it('never shows the request ID', () => {
    const html = render('answer', 'You can carry over 5 days [1].', [citation])
    expect(html).not.toContain('req-1234abcd')
    expect(html).not.toContain('Request ID')
  })

  it('counts several sources', () => {
    const html = render('answer', 'A rule [1] and another [2].', [citation, citation])
    expect(html).toContain('From company policies · 2 sources')
    expect(html).toContain('<details id="source-2">')
  })

  it('labels general knowledge without repeating the label in the text', () => {
    const html = render(
      'general',
      'General knowledge (not from the company policies): The capital of France is Paris.',
    )
    expect(html).toContain('General knowledge, not from company policies')
    expect(html).toContain('<p>The capital of France is Paris.</p>')
    expect(html).not.toContain('Sources')
  })

  it('shows small talk without a badge or sources', () => {
    const html = render('chat', 'Hello! Try asking about annual leave.')
    expect(html).not.toContain('class="badge')
    expect(html).not.toContain('Sources')
  })

  it('turns a decline into guidance with example questions', () => {
    const html = render('not_found', 'I could not find the answer in the provided policy documents.')
    expect(html).toContain('Not covered by the policies')
    expect(html).toContain('The policy documents don&#x27;t answer this.')
    expect(html).toContain('class="example"')
    expect(html).not.toContain('Sources')
  })

  it('asks for a narrower question when the answer was too long', () => {
    const html = render('too_long', 'This question needs a longer answer than I can give.')
    expect(html).toContain('Answer too long')
    expect(html).toContain('Ask about one part of a policy instead')
    expect(html).toContain('class="example"')
    expect(html).not.toContain('Sources')
  })
})
