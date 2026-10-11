# Overview

The details behind the [README](../README.md): how the assistant behaves, its API and settings, how it is tested and evaluated, and its known issues. The [architecture](architecture.md) and [decision records](decisions/) explain the design.

## How it answers

Every message is first answered from the five policy sections that best match it. If those don't answer it, a second call without sources handles the rest:

| Message | Response | `kind` |
| --- | --- | --- |
| Covered by the policies ("What's the nightly hotel limit?") | An answer from the documents, citing numbered sources | `answer` |
| General knowledge, not about the company ("What's the capital of France?") | A brief answer labelled "General knowledge (not from the company policies):", with no sources | `general` |
| About the company but not in the documents ("How many sick days do I get?") | "I could not find the answer in the provided policy documents." The assistant never guesses company rules. | `not_found` |
| Small talk ("hi") | A short reply with an example question | `chat` |
| Needs a reply longer than the 2,000-token limit | A request to ask about one part of a policy, with example questions | `too_long` |

In the UI:
- **Badges:** each response gets a badge for its kind: "From company policies", with a source count, "General knowledge, not from company policies", "Not covered by the policies", or "Answer too long". Small talk has no badge.
- **Citations:** markers render as numbered chips. Clicking one opens the matching collapsible source card, which shows the policy, the section and the original text.
- **Example questions:** they appear before the first question, after a decline, and after an answer that was too long.
- **Safe Markdown:** answers render as a safe subset of Markdown (paragraphs, emphasis, lists, code). Links become plain text and images are dropped, so a URL in a document can't make the browser fetch it. Raw HTML is dropped too.

An answered policy question costs about $0.00006 and takes 1–10 seconds. Messages the documents don't answer make two calls, so they cost and take about twice as long.

## Policy documents

The corpus is five text-based PDFs in `data/corpus/`. Each PDF is converted to Markdown, with headings recovered from font size, then split by section. Citations therefore name the policy and section, such as "3. Annual Leave > 3.2 Carry-over".

The policies are fictional, so the PDFs are generated from `data/source/*.md`. To regenerate them, run `uv run python scripts/build_corpus_pdfs.py` from `backend/`; the output is byte-identical.

To use your own documents:
- Put text-based PDFs, or `.md`/`.txt` files, in `data/corpus/`, then restart the backend; the index is built in memory once per process.
- Scanned PDFs are rejected, because they need OCR.

## Configuration

Settings come from environment variables, which take precedence over `.env` in the repository root or in `backend/`. [.env.example](../.env.example) lists them all.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | none (required) | Used for both embeddings and answers; the backend won't start without it |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `openai/gpt-oss-20b` | Answer model |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `CORPUS_DIR` | `data/corpus` | Policy documents; a relative path resolves from the repository root |
| `RETRIEVAL_TOP_K` | 5 | Sections sent to the model per question |
| `LLM_MAX_TOKENS` | 2000 | Completion-token limit per reply, including the model's hidden reasoning |
| `LLM_TIMEOUT_SECONDS` | 30 | Timeout per attempt; with two retries a call can take longer |
| `LOG_LEVEL` | INFO | Backend log level |

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health` | Liveness status and version |
| POST | `/api/ask` | Answer, kind, citations and request ID |

**Request:** a `question` of 1–2000 characters after trimming; anything else returns 422.

**Response:**
- `answer`: the text, with `[n]` markers that refer to `citations[n - 1]`;
- `kind`: `answer`, `general`, `chat`, `not_found` or `too_long`, so clients can label a response without parsing its text;
- `citations`: each has `doc_id`, `title`, `section`, `snippet` and `score`. The score is the dense cosine similarity, not a probability;
- `request_id`.

**Errors:**
- **503:** if the embedding or answer model fails or times out, or the index can't be built, the API returns a generic 503. Requests waiting on a failed index build fail with it. The failure isn't cached, so a later request retries the build.
- **500:** unexpected errors.

**Logging:** each request writes one log line with:
- the request ID, retrieval and generation times, and chunk count;
- the route (`grounded`, `general` or `conversation`), the refusal reason (`model_not_found`, `uncited` or `truncated`), and token counts, including those of a cut-off reply;
- for errors, the error type and provider status.

The question text and provider response bodies are never written to the log. Unexpected exceptions are re-raised, and the server logs their traceback, which can include the exception message.

## Tests and checks

From `backend/`:

~~~powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
npx pyright@1.1.408   # type check; needs Node.js, which the frontend uses anyway
~~~

From `frontend/`:

~~~powershell
npm test
npm run lint
npm run build
~~~

All 131 backend tests and 16 frontend tests pass, and Pyright reports no type errors.
- **Mocked, no key needed:** backend tests use fakes, and `httpx2`'s `MockTransport` in place of the API, with a fake key. (`httpx2` is the HTTP library the OpenAI SDK is built on.) Outside network access is blocked, and `.env` is never read, so tests cost nothing.
- **Frontend tests** (Vitest) check Markdown rendering and the citation chips, and that links, images and raw HTML are never rendered.
- **Warnings are errors,** except `PytestCacheWarning`. Being unable to write pytest's optional cache shouldn't fail otherwise passing tests across Windows accounts.

## Evaluation

The evaluator runs the full live pipeline on labelled questions and saves every row, including the model's raw replies. It makes real API calls. From `backend/`, with a key in `.env`:

~~~powershell
uv run python -m evals.run_retrieval_eval --output evals/llm_results_hosted.json --max-cost-usd 0.03
uv run python -m evals.run_retrieval_eval --questions evals/retrieval_questions_heldout.jsonl --output evals/llm_results_heldout_hosted.json --max-cost-usd 0.02
~~~

**Budget guard:**
- It fetches the models' listed prices from OpenRouter.
- It reserves the cost of indexing the corpus first, then each question's worst case: both possible calls, with retries, at the full token limit.
- It skips any question that would exceed the cap. The worst case grows with `LLM_MAX_TOKENS`, which is why the caps above are well over the actual spend of about $0.002 per run.
- A provider failure is recorded in its row and the run continues.

**Questions:**
- **Calibration:** 20 questions, 15 answerable and 5 not.
- **Held-out:** 13 reviewer-written questions, 7 answerable paraphrases and 6 near misses the policies don't cover.

**Metrics:**
- **Retrieval hit:** needs both the right document and the full section path.
- **MRR:** truncated at 5.
- **Answer-contains:** checks for a labelled phrase after normalizing case and typography (Unicode spaces, dash and quote variants, Markdown markers). Refusals count as misses.

Results on 2026-10-11 with the hosted models, measured with the earlier 500-token limit:

| Metric | Calibration (20 questions) | Held-out (13 questions) |
| --- | --- | --- |
| Hit@1 | 100% (15/15) | 100% (7/7) |
| MRR@5 | 1.0000 | 1.0000 |
| Expected section cited | 100% (15/15) | 100% (7/7) |
| Answer-contains (normalized phrase) | 86.67% (13/15) | 85.71% (6/7) |
| Unanswerable refusal rate | 100% (5/5) | 100% (6/6) |
| Answerable false-refusal rate | 0% (0/15) | 0% (0/7) |

- **Phrase misses:** all three (q06, q14, h12) are correct answers worded differently from the label, judged from the raw replies.
- **Declines:** no unanswerable question got a general-knowledge answer.
- **Not a blind estimate:** the held-out questions were seen in earlier rounds, so measuring quality needs fresh questions.
- **Limit raised since:** no reply in either run hit the 500-token limit, and a higher limit doesn't change a reply that finished within it. The results should therefore still hold at 2,000, but they were not re-run.

[Calibration report](../backend/evals/llm_results_hosted.json) and [held-out report](../backend/evals/llm_results_heldout_hosted.json). The [decision record](decisions/0003-hosted-models-only.md#results) has the details.

**History:**
- Earlier results are kept in `backend/evals/` and described in [decision 0002](decisions/0002-retrieval-and-generation.md#evaluation-protocol-and-results): the removed extractive mode (`retrieval_results*.json`), and two live rounds with the local embedding model (`llm_results.json`, `llm_results_r2.json` and their held-out files).
- [`evaluation_protocol.json`](../backend/evals/evaluation_protocol.json) records every change made after the first held-out run, with file hashes.
- All live testing cost $0.0085 in total, measured on the key's usage record.

## Known issues

- **Phrase labels understate answers:** generated answers paraphrase, so a correct answer can miss a long labelled phrase. Short key-fact labels or human review would measure answers better.
- **Long answers:** a reply that needs more than 2,000 tokens, such as a summary of several whole policies, gets `too_long` instead of a partial answer. A long answer can also take 10–20 seconds. [Details](decisions/0003-hosted-models-only.md#answer-length-limit).
- **Fallback latency:** messages the documents don't answer make two calls, grounded and then fallback. One greeting took 12 seconds.
- **Cold start:** the backend embeds the corpus through the API on the first question after each start. That cost about $0.00013, and the first answer took 7.3 seconds.
- **Provider routing varies:** OpenRouter can route each call to a different provider, so cost and latency vary; one test saw a fourfold cost range. Pinning a provider would make evaluations more reproducible.
- **Citation brackets:** numeric-looking brackets such as `[2024-2025]` may be removed or read as citations. [Details](decisions/0002-retrieval-and-generation.md#citation-brackets).
- **Title parsing:** an unusual Markdown title (closing hashes, a tab, a heading inside a code fence) can give a wrong title; the bundled corpus is unaffected. [Details](decisions/0002-retrieval-and-generation.md#title-parsing).
- **PDF extraction limits:** the converter doesn't handle scans, tables or multi-column layouts. Headings are found by font size only, and citations name sections, not pages. [Details](decisions/0002-retrieval-and-generation.md#pdf-extraction-limits).
- **Single turn only:** each question is answered on its own, so a follow-up like "And for part-time staff?" lacks context and is usually declined. [Details](decisions/0002-retrieval-and-generation.md#conversation-memory).
- **Prompt injection:** the prompt tells the model to treat sources as data, but that is wording, not an enforced defence. It hasn't been tested adversarially.

## Design questions

1. **Why these choices?** A small corpus suits a simple in-memory hybrid index. Hosted models give real answers with no model downloads, at a tiny cost per question. [Assumptions](architecture.md#use-case-and-assumptions), [retrieval](decisions/0002-retrieval-and-generation.md#hybrid-retrieval-with-rrf) and [hosted models](decisions/0003-hosted-models-only.md).
2. **What could go wrong?** The model could answer from the wrong section, mislabel a company question as general knowledge, or fail; the provider can time out. The [failure table](architecture.md#failure-modes-and-edge-cases) covers each case and how it is caught.
3. **How do we know it works?** Mocked tests check contracts and error paths. The live evaluation measures retrieval, citations, refusals and answer content separately, and keeps raw replies so failures can be diagnosed. [Evaluation and limits](architecture.md#evaluation-and-limits).
4. **What are the trade-offs?** Quality, latency, cost, reliability and complexity. [Trade-offs](architecture.md#trade-offs).
5. **How would it evolve?** Persist versioned embeddings when restarts or corpus size make re-embedding costly. Add a reranker or support checks when measured errors justify them. [Production triggers](architecture.md#poc-versus-production).
6. **What would production need?** Authentication, document access control, budgets and rate limits, managed secrets, privacy-aware monitoring, larger evaluations, and controlled document updates. [PoC versus production](architecture.md#poc-versus-production).
7. **How do the components work together?** React sends the question through the Vite proxy to FastAPI. The pipeline embeds it, retrieves sections, generates a cited answer or falls back, then validates the citations. [Request flow](architecture.md#request-flow).

## Requirements checklist

| Assessment requirement | Where it is met |
| --- | --- |
| Justify key decisions, trade-offs, assumptions and alternatives | [Decision records](decisions/) and [assumptions](architecture.md#use-case-and-assumptions) |
| Working web interface, from question to grounded response | The [README](../README.md) run steps |
| Reliable, maintainable and measurable beyond a prototype | [Tests and checks](#tests-and-checks), [Evaluation](#evaluation), and error handling in the [API](#api) |
| What could go wrong: edge cases and failure modes | [Failure table](architecture.md#failure-modes-and-edge-cases) and [Known issues](#known-issues) |
| How to know it works, measure it and improve it | [Evaluation](#evaluation) and [evaluation and limits](architecture.md#evaluation-and-limits) |
| Trade-offs: quality, latency, cost, reliability, usability, complexity | [Trade-offs](architecture.md#trade-offs) |
| Evolution with scale, usage, data or requirements | [PoC versus production](architecture.md#poc-versus-production) |
| Production versus proof of concept | [PoC versus production](architecture.md#poc-versus-production) |
| How components work together | [Current state and request flow](architecture.md#current-state) |
| Responsible use of the provided credits | Every live run was budget-capped; all testing cost $0.0085 |
| README, `.env.example`, no committed secrets | The [README](../README.md) and [.env.example](../.env.example); `.env` is gitignored |

## Project layout

- `backend/app/rag/`: PDF conversion, chunking, hosted embeddings, BM25 and RRF, generation, citations, and request metrics.
- `backend/app/api/`, `schemas.py`, `config.py`: HTTP routes, the API contract and settings.
- `backend/tests/`: mocked tests. `backend/evals/`: labelled questions, the live evaluator and saved reports.
- `backend/scripts/build_corpus_pdfs.py`: generates the policy PDFs.
- `data/corpus/`: the five policy PDFs. `data/source/`: the Markdown they are generated from.
- `frontend/`: the React/Vite chat page.
- `docs/`: this overview, the [architecture](architecture.md) and the [decision records](decisions/).

Out of scope for this PoC: Docker, authentication, a vector database, a persistent index, and conversation memory.
