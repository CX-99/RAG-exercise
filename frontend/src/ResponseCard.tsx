import { AnswerText } from './AnswerText'
import { EXAMPLE_QUESTIONS } from './examples'
import type { AnswerKind, AskResponse, Citation } from './types'

// Mirrors GENERAL_KNOWLEDGE_PREFIX in backend/app/rag/prompts.py.
const GENERAL_PREFIX = 'General knowledge (not from the company policies):'

/** Badge text and tone for each kind of response; small talk gets no badge. */
const BADGES: Record<AnswerKind, { label: string; tone: 'policy' | 'general' | 'neutral' } | null> =
  {
    answer: { label: 'From company policies', tone: 'policy' },
    general: { label: 'General knowledge, not from company policies', tone: 'general' },
    chat: null,
    not_found: { label: 'Not covered by the policies', tone: 'neutral' },
    too_long: { label: 'Answer too long', tone: 'neutral' },
  }

/** Shown with example questions in place of the text when there is no answer to give. */
const GUIDANCE: Partial<Record<AnswerKind, string>> = {
  not_found:
    "The policy documents don't answer this. Try rephrasing, or ask about annual leave, " +
    'travel and expenses, remote work, information security, or the code of conduct.',
  too_long:
    'This question needs a longer answer than the assistant can give in one reply. ' +
    'Ask about one part of a policy instead, for example:',
}

/** Clickable example questions; each asks the question straight away. */
export function ExampleQuestions({
  onAsk,
  disabled = false,
}: {
  onAsk: (question: string) => void
  disabled?: boolean
}) {
  return (
    <ul className="examples" aria-label="Example questions">
      {EXAMPLE_QUESTIONS.map((question) => (
        <li key={question}>
          <button
            type="button"
            className="example"
            disabled={disabled}
            onClick={() => onAsk(question)}
          >
            {question}
          </button>
        </li>
      ))}
    </ul>
  )
}

/** The cited sources as collapsible cards; citation chips in the answer open them. */
function Sources({ citations }: { citations: Citation[] }) {
  return (
    <section className="sources" aria-label="Sources">
      <h3>Sources</h3>
      <ol>
        {citations.map((citation, index) => (
          <li key={`${citation.doc_id}-${index}`}>
            <details id={`source-${index + 1}`}>
              <summary>
                <span className="source-number">{index + 1}</span>
                <span className="source-label">
                  <span className="source-title">{citation.title}</span>
                  <span className="source-section">{citation.section}</span>
                </span>
              </summary>
              <blockquote>{citation.snippet}</blockquote>
            </details>
          </li>
        ))}
      </ol>
    </section>
  )
}

/**
 * One response, labelled by its kind. The request ID stays in the API response and the
 * server log for tracing, but isn't shown.
 */
export function ResponseCard({
  result,
  onAsk,
}: {
  result: AskResponse
  onAsk: (question: string) => void
}) {
  const badge = BADGES[result.kind]
  const guidance = GUIDANCE[result.kind]
  const sources = result.citations.length
  // The badge already says so; don't repeat the label as the answer's first words.
  const text =
    result.kind === 'general' && result.answer.startsWith(GENERAL_PREFIX)
      ? result.answer.slice(GENERAL_PREFIX.length).trim()
      : result.answer

  return (
    <article className={`response response-${result.kind}`}>
      {badge && (
        <p className={`badge badge-${badge.tone}`}>
          {badge.label}
          {sources > 0 && ` · ${sources} ${sources === 1 ? 'source' : 'sources'}`}
        </p>
      )}
      {guidance ? (
        <>
          <p>{guidance}</p>
          <ExampleQuestions onAsk={onAsk} />
        </>
      ) : (
        <AnswerText text={text} />
      )}
      {sources > 0 && <Sources citations={result.citations} />}
    </article>
  )
}
