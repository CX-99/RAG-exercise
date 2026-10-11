# Architecture

## Use case and assumptions

The assistant answers employees' questions about five fictional company policies. A useful answer cites the policy and section it came from, so a reader can check it against the original text. It must never guess a company rule the documents don't contain.

| Assumption | Effect on the design |
| --- | --- |
| A small English corpus of text-based PDFs | Convert each PDF to Markdown, recovering headings from font size, so section paths survive. Index in memory with flat cosine search plus BM25. Scanned PDFs, tables, multi-column layouts and other languages are unsupported or unevaluated. |
| API credits are limited | Use small, cheap hosted models (`openai/text-embedding-3-small`, `openai/gpt-oss-20b`). Cap each reply at 2,000 tokens, enough for a whole-policy summary. Budget-guard every live evaluation, and keep tests mocked so they cost nothing. |
| A hosted API is acceptable for this data | Questions and policy text go to OpenRouter for embedding and generation. Real, sensitive policies would need a provider agreement or self-hosted models. |
| Policies are fictional and can be inspected directly | Bundle the corpus with labelled questions, so reviewers can trace each expected phrase to its section. Results don't establish accuracy on real policies or conflicting versions. |
| One local user with no shared service | Use a Vite development proxy and one in-memory pipeline per backend process. There is no authentication and no stored conversation or index. These are scope assumptions, not a production access model. |
| Users ask complete, self-contained questions | Each request is answered on its own, with no conversation history or user memory. A follow-up without its subject is retrieved as written and usually declined. |
| Corpus edits are infrequent and a restart is acceptable | Load and embed documents lazily on the first request, then reuse that snapshot. There is no file watcher, incremental update or effective-date handling. |
| Questions and provider errors may contain sensitive text | Application logs record types and statuses, never question text or provider response bodies. SDK debug logging is disabled. Unexpected exceptions still reach the server's traceback log. |

## Current state

A working PoC: an in-memory hybrid index over hosted embeddings, grounded generation with validated citations, and a source-free fallback for messages the documents don't answer.

~~~text
Browser → Vite proxy → FastAPI /api/ask → cached PolicyPipeline
                                           |
                 embed question (OpenRouter) → cosine + BM25 → RRF → top 5 sections
                                           |
                     grounded call: question + numbered sources (OpenRouter)
                                           |
                       normalize 【n】/［n］, validate and renumber [n]
                                           |
                     cut off by the token limit? ── yes → kind=too_long
                                           |
                     valid citation? ── yes → kind=answer, cited sources
                                           |
                                           no
                                           |
                     fallback call: question only, no sources (OpenRouter)
                         small talk          → kind=chat
                         general knowledge   → kind=general, labelled
                         company question    → kind=not_found
~~~

| Component | Responsibility |
| --- | --- |
| `frontend/` | Question form with example questions, loading and error states, a badge per response kind, safe Markdown with citation chips, and collapsible source cards |
| `app/api/` | Validation, request IDs, safe request logs and error mapping |
| `app/schemas.py` | Request and response contract |
| `app/rag/pdf.py`, `documents.py`, `chunking.py` | PDF-to-Markdown conversion, file loading, titles, heading paths and section chunks |
| `app/rag/embeddings.py` | `Embedder` protocol and the OpenRouter embedding client |
| `app/rag/retrieval.py` | Normalized cosine, BM25 and RRF fusion |
| `app/rag/prompts.py`, `llm.py` | Grounded and fallback prompts, and the OpenRouter chat client |
| `app/rag/errors.py` | `LLMError`, the safe error shared by embeddings and generation |
| `app/rag/citations.py` | Bracket normalization, citation groups and ranges, renumbering and the uncited check |
| `app/rag/pipeline.py`, `telemetry.py` | Lazy cached build, orchestration and request-local metrics |
| `app/config.py`, `main.py` | Settings validation at startup and the build-failure HTTP mapping |
| `evals/` | Live, budget-guarded evaluation of retrieval, citations, refusals and answer content |

### Request flow

1. **Validate** the question: 1–2000 characters after trimming. Invalid input returns 422.
2. **Get the pipeline.** If it is already built, return it immediately. Otherwise acquire the build lock, check again, and build in a worker thread: load and chunk the documents, then embed all chunks in one API call. A failed build raises `PipelineUnavailable`, which `main.py` maps to a generic 503, logging only the type. Requests that were waiting on that build fail with it, rather than each starting another slow build. The failure isn't cached, so a later request retries.
3. **Start the request.** Create a request ID and request-local metrics, then call `PolicyPipeline.answer(question)`.
4. **Retrieve** in a worker thread: embed the question, rank chunks by cosine and BM25, fuse with RRF (k=60), and keep the top 5.
5. **Grounded call:** send the question and numbered sources. The system prompt requires source-only claims with `[n]` citations, the fixed not-found reply when the sources don't answer, and treating source text as data.
6. **Validate citations:**
   - Normalize lenticular and fullwidth brackets to `[n]`.
   - Expand groups and ranges, drop invalid references, and renumber by first appearance.
   - Return only the cited sources.
   - A not-found reply gives refusal reason `model_not_found`; a reply with no valid citation gives `uncited`.
7. **Fallback:** a declined or uncited reply goes to `converse` with no sources. It returns one of:
   - a short small-talk reply (`route=conversation`, `kind=chat`);
   - a general-knowledge answer starting with the fixed label (`route=general`, `kind=general`);
   - the not-found reply for any question about the company (`kind=not_found`, keeping the grounded refusal reason).

   Any `[n]` in a fallback reply is removed, because no sources were shown.
8. **Return** `answer`, `kind`, `citations` and `request_id`. A reply cut off by the token limit, in either call, is never shown: the response is `kind=too_long` with refusal reason `truncated`, and its tokens are still logged. Embedding or generation errors and timeouts map to a generic 503. Missing token counts don't discard a complete answer. Unexpected exceptions propagate as 500.
9. **Log** one line: timings, chunk count, route, refusal reason, token usage, and any error type and provider status. The question text is never logged.

### Test and configuration seams

- **Injection points:** `RagPipeline`, `Embedder`, `Retriever` and `LLMClient` are protocols, so each can be replaced in tests. API tests override `get_pipeline`; build tests replace only its builder.
- **Mocked tests:**
  - Tests never read `.env`; they set a fake key, and outside network access is blocked.
  - The Windows event loop's internal loopback socket pair is still allowed.
  - Embedding and chat clients are tested against `httpx2`'s `MockTransport`, covering request shape, row order, row-count mismatches, provider errors, timeouts and truncation.
- **Settings:** a missing or blank `OPENROUTER_API_KEY` stops app creation; each client repeats the check.
- **Hot path:** cached requests skip the build lock and thread hop. Corpus changes need a restart.
- **Secrets:** the Vite proxy keeps the key on the backend. SDK debug logging is disabled to avoid logging prompts.

## Evaluation and limits

Results on 2026-10-11 with the hosted models, one budget-guarded run per set:

| Metric | Calibration (20 questions) | Held-out (13 questions) |
| --- | --- | --- |
| Hit@1 (document and full section path) | 100% (15/15) | 100% (7/7) |
| MRR@5 | 1.0000 | 1.0000 |
| Expected section cited | 100% (15/15) | 100% (7/7) |
| Answer-contains (normalized phrase) | 86.67% (13/15) | 85.71% (6/7) |
| Unanswerable refusal rate | 100% (5/5) | 100% (6/6) |
| Answerable false-refusal rate | 0% (0/15) | 0% (0/7) |

**What the results show:**
- **Paraphrases, not errors:** the three phrase misses (q06, q14, h12) are correct answers worded differently from the label. Every answerable question was answered with the expected citation.
- **Declines held:** every unanswerable question got the not-found reply, never a general-knowledge answer.

**Limits:**
- **Not blind:** the held-out questions were exposed in earlier rounds, which diagnosed and fixed failures. These results confirm the current design works on them, not how it does on new questions.
- **Small sample:** each set ran once, and OpenRouter can route each call to a different provider.
- **Phrase labels:** the answer-contains rate checks for a labelled phrase after normalizing case and typography, and counts refusals as misses. It understates correct paraphrased answers and doesn't prove an answer is complete.
- **Not measured:** prompt-injection resistance, production latency and cost at volume.

**History:** earlier results are kept as labelled history in [decision 0002](decisions/0002-retrieval-and-generation.md#evaluation-protocol-and-results): the removed extractive mode and two live rounds with a local embedding model. The [evaluation protocol](../backend/evals/evaluation_protocol.json) records each change made after the first held-out run, with file hashes. The change to hosted models is recorded there under `hosted_models_only`.

**To improve the next version:**
- Collect fresh, independently labelled paraphrases, near misses and document edge cases.
- Tune on a development set and keep a held-out set unseen.
- Use short key-fact labels or human review of claim support instead of long phrases.
- Measure retrieval, citations, refusals and answer content separately.

## Trade-offs

- **Quality and usability:** generated answers synthesize across a section and read naturally, and section citations with source cards let readers check them. Valid citation syntax doesn't prove each claim is supported.
- **Latency and cost:**
  - An answered policy question costs about $0.00006 and takes 1–10 seconds.
  - Messages the documents don't answer make a second call, which roughly doubles their cost and latency; one greeting took 12 seconds.
  - Each backend start re-embeds the corpus (about $0.00013), and the first answer took 7.3 seconds.
  - No production latency target or load test exists.
- **Reliability:** both models depend on one gateway and the network; there is no offline mode. Generic 503s, retries and uncached build failures limit the impact of a provider outage, but don't remove it.
- **Complexity and scale:** flat search and a per-process cache are easy to inspect. But search is linear, each worker holds its own copy, and updates need a restart. Persistence, ANN search or a vector service would add operations and versioning work, and should follow measured bottlenecks.

## Failure modes and edge cases

| Failure | Current behaviour | How it is detected or mitigated |
| --- | --- | --- |
| Question the documents don't answer, but a related section exists | The grounded call must return the not-found reply; the fallback must then decline any company question. | All 11 unanswerable evaluation questions were declined. A wrong answer would still show its cited source for checking. |
| Company question answered from general knowledge | The fallback prompt forbids answering company rules from general knowledge. The label is recognised even if the model formats it (bold, other case). | A labelled answer shows the "General knowledge" badge. An unlabelled one would show as small talk: no badge, but no sources either. None of the 11 unanswerable company questions received either. |
| Small talk such as "hi" | The grounded call declines, then the fallback replies briefly with an example question and no citations. | The log records `route=conversation`. It takes two calls. |
| General question outside the company | The fallback answers it, labelled "General knowledge (not from the company policies):", with no citations. | The log records `route=general`; the UI shows a separate badge. |
| Wrong section ranks first | All top 5 sections go to the model, which can cite any of them. | Hit@1 and MRR are measured separately from citations; both were perfect in the latest run. |
| Model cites as `【1】` | Lenticular and fullwidth brackets are normalized to `[n]` before parsing. | Citation tests cover it; an earlier round lost three correct answers before the fix. |
| Answer without a valid citation | Treated as uncited and sent to the fallback, so an ungrounded policy claim isn't shown as an answer. | The log records `refusal_reason=uncited`. |
| Prompt injection inside a document | The system prompt tells the model to treat sources as data; nothing enforces it. | The UI never renders links or images from a response, so injected Markdown can't make the browser fetch a URL. Adversarial evaluation and claim-support checks are future work. |
| Embedding or generation provider fails or times out | Each call has a 30-second timeout per attempt and two retries. Errors and timeouts return a generic 503. | Mocked tests cover each path. The log records the error type and provider status. |
| Reply needs more than the token limit ("return me the annual leave policy details" at the old 500-token limit) | `finish_reason=length` gives `kind=too_long`, asking for a narrower question, instead of a cut-off answer. The model's hidden reasoning counts toward the limit. | The log records `refusal_reason=truncated` with the tokens used. The limit was raised from 500 to 2,000 after this was seen live. |
| Embedding API returns the wrong number of rows | Raised as an error rather than pairing chunks with the wrong vectors. | A mocked test covers it. |
| API key missing or blank | Settings validation stops app creation with a clear message. | Configuration and client tests cover missing and blank keys. |
| Index build fails (provider down, unreadable document) | The first request returns a generic 503; the failure isn't cached, so a later request retries. | The build log records the exception type. Liveness alone doesn't confirm readiness. |
| Concurrent first requests | A lock and a second cache check make sure the index is built once, in a worker thread. If that build fails, the requests waiting on it fail with it instead of rebuilding one after another. | Concurrency tests check one build, including when it fails; warm requests bypass the lock. |
| Corpus edited while the server runs | The built pipeline keeps using its in-memory snapshot. | Restart after updates. There is no change detection. |
| Scanned or unreadable PDF | The loader raises an error, so the index build fails with a generic 503 and logs the type. | Tests cover both cases. OCR is future work. |
| PDF with tables, columns or body-size bold headings | Text order or section boundaries can be wrong; nothing detects it. | The bundled PDFs avoid these layouts. A parity test checks their chunks match the Markdown sources exactly. |
| Follow-up without its subject ("And for part-time staff?") | Answered as a standalone question and usually declined. | Documented single-turn decision. History-aware query rewriting is the planned fix. |
| Duplicate document stems | `leave.md` and `leave.txt` share a `doc_id`, so loading raises an error. | Loader tests cover it; the build failure maps to 503. |
| Oversized paragraph | The 1,200-character chunk target is soft; a large paragraph stays whole. | Chunker tests cover it. Token-aware limits are future work. |
| Sensitive text in logs | Application log fields hold types and statuses only. Unexpected exceptions are re-raised, and the server logs their traceback with the message. | Tests cover the structured fields, not server redaction. Shared use needs redaction, access limits and retention rules. |
| Unusual Markdown title syntax | The loader's title rule can disagree with the chunker's heading rules. | [Known issue](decisions/0002-retrieval-and-generation.md#title-parsing); the bundled corpus is unaffected. |
| Numeric prose in brackets | `[2024-2025]` may be removed, or a clause number read as a citation. | [Known issue](decisions/0002-retrieval-and-generation.md#citation-brackets); the source cards allow checking. |
| Model formats with Markdown or Unicode spacing | The UI renders a safe Markdown subset; the evaluator normalizes typography before matching. | Frontend and evaluator tests cover the forms the model produced. |

## PoC versus production

These changes are proposals, not implemented capabilities.

| Current PoC choice | What production would need | What would trigger the change |
| --- | --- | --- |
| Lazy, per-process in-memory index, re-embedded at each start | Build immutable indexes outside request handling. Persist embeddings keyed by model, document and chunker version, with atomic rollout and rollback. | Measured cold-start delay, re-embedding cost, memory duplication, frequent updates or corpus growth. |
| Flat cosine + BM25/RRF with the model deciding answerability | Evaluate a reranker and a calibrated support check; verify claims against cited evidence. Consider ANN search only after measuring search cost. | Unsupported answers or false refusals above agreed targets, or search becoming a bottleneck. |
| Hosted models through one gateway, with no pinned provider | Pin providers or use a direct provider contract. Add fallbacks, circuit breakers and data-processing agreements. Consider self-hosted models for sensitive policies. | Uptime targets, reproducibility needs, or confidential documents. |
| Mocked tests and small saved live evaluations | Run regression checks in CI. Curate larger, independently labelled held-out sets with tuning kept separate. Evaluate groundedness and injection resistance in budgeted runs. | Model, prompt or retrieval changes, new policy areas, or external users. |
| Per-request timings, refusal reasons and token counts | Add end-to-end traces, latency percentiles, readiness metrics, alerts, privacy-aware log handling and priced cost accounting. | Service-level objectives, paid traffic, or incident response. |
| No authentication, document access control, rate limits or spending limits | Authenticate users and apply source permissions before retrieval. Set rate and concurrency limits, and enforce per-user and total budgets. | Shared access or sensitive policies. |
| Local `.env` configuration | Use managed secrets with rotation and least privilege. Redact sensitive errors and monitor key usage. | Deployment beyond one workstation. |
| Local Vite and FastAPI development servers | Serve a built frontend behind a production ingress with TLS. Run a supervised backend with readiness checks, bounded concurrency and tested rollout. | Shared availability or uptime requirements. |
| Generated text-based PDFs converted with font-size rules, without versions or effective dates | Layout-aware extraction with OCR, for example Docling or a document-intelligence service. Add table structure, page references, stable IDs, provenance, effective dates and an approval process. | Real policies, scans or complex layouts, conflicting revisions, or more formats. |
| Single-turn questions with no memory | History-aware query rewriting over the last few turns, kept client-side, with multi-turn evaluation. Long-term memory would need consent, retention limits and per-user access control. | Users routinely ask follow-ups, or a chat workflow is required. |
| No feedback capture | Collect opt-in feedback with suitable consent and retention, and turn recurring failures into new evaluation cases. | Repeated user confusion or unsupported answers, or a need to measure usefulness. |

## Planned pipeline evolution

- Collect fresh evaluation questions with short key-fact labels, and keep a held-out set unseen.
- Evaluate a reranker and claim-level support checks; valid citation syntax alone doesn't prove support.
- Pin the OpenRouter provider for reproducible evaluations.
- Add OCR, table parsing and page references to PDF ingestion, and keep document versions and effective dates.
- Persist versioned embeddings, or adopt ANN or a vector store, only when cold starts, memory or search cost warrant it.

## Decision records

- [0001: Application stack and repository layout](decisions/0001-stack.md)
- [0002: Retrieval and generation](decisions/0002-retrieval-and-generation.md) (superseded in part by 0003)
- [0003: Hosted models only](decisions/0003-hosted-models-only.md)
