// Matches [n] and [n, m] citation markers that aren't already Markdown links.
const CITATION_GROUP = /\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g

/** Rewrite `[n]` and `[n, m]` markers as links to the matching source cards. */
export function linkCitations(text: string): string {
  return text.replace(CITATION_GROUP, (_, group: string) =>
    group
      .split(',')
      .map((number) => `[${number.trim()}](#source-${number.trim()})`)
      .join(''),
  )
}
