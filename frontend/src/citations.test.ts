import { describe, expect, it } from 'vitest'
import { linkCitations } from './citations'

describe('linkCitations', () => {
  it('links single and grouped markers to source anchors', () => {
    expect(linkCitations('A [1]. B [2, 3].')).toBe(
      'A [1](#source-1). B [2](#source-2)[3](#source-3).',
    )
  })

  it('leaves existing Markdown links and non-numeric brackets alone', () => {
    expect(linkCitations('[1](https://example.com) and [a label]')).toBe(
      '[1](https://example.com) and [a label]',
    )
  })
})
