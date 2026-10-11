// Mirrors backend/app/schemas.py — keep the two in sync.

/** Longest accepted question after trimming; must match the backend's `MAX_QUESTION_CHARS`. */
export const MAX_QUESTION_CHARS = 2000

/** A source passage that a response cites. */
export interface Citation {
  /** Corpus file stem, e.g. `annual-leave`. */
  doc_id: string
  title: string
  /** Heading path, e.g. `3. Annual Leave > 3.2 Carry-over`. */
  section: string
  /** Raw section text, shown so readers can check the response against the source. */
  snippet: string
  /** Dense cosine similarity to the question; not a probability. */
  score: number
}

/**
 * What a response is, so the UI can label it without parsing its text: generated from
 * cited sources, labelled general knowledge, small talk, a decline, or a reply too long
 * to give in full (it hit the token limit, so it isn't shown).
 */
export type AnswerKind = 'answer' | 'general' | 'chat' | 'not_found' | 'too_long'

/** Body of a successful `POST /api/ask` response. */
export interface AskResponse {
  /** Response text with `[n]` markers that refer to `citations[n - 1]`. */
  answer: string
  kind: AnswerKind
  citations: Citation[]
  /** Also written to the server log, so a reported problem can be traced. */
  request_id: string
}
