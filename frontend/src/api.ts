import { MAX_QUESTION_CHARS, type AskResponse } from './types'

/** An error with a message that is safe to show to the user. */
export class ApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/**
 * Send a question to the backend's `POST /api/ask`.
 *
 * @param question - The trimmed question text.
 * @returns The answer, its citations, and the request ID.
 * @throws {@link ApiError} with a user-safe message if the request fails.
 */
export async function askQuestion(question: string): Promise<AskResponse> {
  let response: Response
  try {
    response = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
  } catch {
    throw new ApiError('Could not reach the server. Check your connection and try again.')
  }

  if (!response.ok) {
    throw new ApiError(describeFailure(response.status), response.status)
  }
  return (await response.json()) as AskResponse
}

/** Map an HTTP status to a short message that is safe to show to the user. */
function describeFailure(status: number): string {
  if (status === 422) {
    return `Please enter a question of up to ${MAX_QUESTION_CHARS} characters.`
  }
  if (status >= 500) {
    return `The server is unavailable or hit an error (${status}). Please try again shortly.`
  }
  return `Request failed (${status}).`
}
