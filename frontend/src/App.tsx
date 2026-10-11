import { useState, type KeyboardEvent, type SubmitEvent } from 'react'
import { ApiError, askQuestion } from './api'
import { ExampleQuestions, ResponseCard } from './ResponseCard'
import { MAX_QUESTION_CHARS, type AskResponse } from './types'

/** State of the current question, as a union so each state carries only its own data. */
type RequestState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; result: AskResponse }
  | { status: 'error'; message: string }

/** The question form and the response view. */
export default function App() {
  const [question, setQuestion] = useState('')
  const [state, setState] = useState<RequestState>({ status: 'idle' })

  const isLoading = state.status === 'loading'
  const canSubmit = question.trim().length > 0 && !isLoading

  async function ask(text: string) {
    const trimmed = text.trim()
    if (!trimmed || isLoading) return
    setQuestion(trimmed)
    setState({ status: 'loading' })
    try {
      const result = await askQuestion(trimmed)
      setState({ status: 'success', result })
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : 'Something went wrong. Please try again.'
      setState({ status: 'error', message })
    }
  }

  function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault()
    void ask(question)
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.currentTarget.form?.requestSubmit()
    }
  }

  return (
    <main className="app">
      <header>
        <h1>Policy Assistant</h1>
        <p className="muted">
          An AI chatbot for the company's policies. Answers cite the sections they rely on.
        </p>
      </header>

      <form className="ask-form" onSubmit={handleSubmit}>
        <label htmlFor="question">Your question</label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          maxLength={MAX_QUESTION_CHARS}
          rows={3}
          placeholder="e.g. How many days of annual leave do new employees get?"
        />
        <div className="form-footer">
          <span className="muted">
            {question.length}/{MAX_QUESTION_CHARS} · Ctrl+Enter to ask
          </span>
          <button type="submit" disabled={!canSubmit}>
            {isLoading ? 'Asking…' : 'Ask'}
          </button>
        </div>
      </form>

      {state.status === 'idle' && (
        <section className="intro" aria-label="Getting started">
          <p className="muted">Try one of these:</p>
          <ExampleQuestions onAsk={ask} />
        </section>
      )}

      {/* Always rendered, so screen readers announce the result when it appears. */}
      <div aria-live="polite">
        {state.status === 'loading' && (
          <p className="status" role="status">
            <span className="spinner" aria-hidden="true" />
            Searching the policies…
          </p>
        )}
        {state.status === 'error' && (
          <p role="alert" className="error">
            {state.message}
          </p>
        )}
        {state.status === 'success' && <ResponseCard result={state.result} onAsk={ask} />}
      </div>
    </main>
  )
}
