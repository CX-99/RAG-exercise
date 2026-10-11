# Policy Assistant

A working RAG proof of concept for five fictional company policies. It retrieves sections and, by default, quotes relevant sentences with citations. Optional OpenRouter generation is behind the same interface.

> **Status:** Working PoC with an offline extractive mode and a live-LLM mode, both evaluated below on the same 33 questions. The relevance threshold was calibrated on 20 in-sample questions; topical similarity does not establish whether a question can be answered.

## Run locally

Prerequisites: Python 3.12+, uv, and Node.js 22+. Use `uv`, or `python -m uv` if uv isn't on PATH.

~~~powershell
# Terminal 1, from the repository root
cd backend
uv sync --locked
uv run uvicorn app.main:app --reload

# Terminal 2, from the repository root
cd frontend
npm install
npm run dev
~~~

The UI is at http://localhost:5173 and API docs at http://localhost:8000/docs. Vite proxies /api to the backend. Default LLM_PROVIDER=extractive requires no API key. The embedding model downloads on first use unless cached. Restart the backend after editing the corpus; the index is in memory and built once per process.

### Run with a real LLM (recommended for assessment)

Extractive mode quotes passages; generated answers need an OpenRouter key. Copy [.env.example](.env.example) to `.env`, set `OPENROUTER_API_KEY`, then start the backend in OpenRouter mode (the frontend is unchanged):

~~~powershell
cd backend
$env:LLM_PROVIDER = "openrouter"
uv run uvicorn app.main:app --reload
~~~

In this mode the assistant tries to answer everything:

| Message | Response |
| --- | --- |
| Covered by the policies ("What's the nightly hotel limit?") | Answer from the documents, citing numbered sources |
| General knowledge, not about the company ("What's the capital of France?") | Starts "General knowledge (not from the company policies):", no citations |
| About the company but not in the documents ("How many sick days do I get?") | "I could not find the answer…": it never guesses company rules |
| Small talk ("hi") | A short reply with an example question |

Every message is first answered from its top 5 policy sections. Only if those don't answer it does a second, source-free call handle small talk, general knowledge or the decline. In live use an answered policy question cost about $0.00006 and took 1–10 seconds; a fallback message makes two calls. `$env:LLM_PROVIDER` lasts only for that terminal session.

### Policy documents

The corpus is five text-based PDFs in `data/corpus/`. They are converted to Markdown, with headings recovered from font size, then chunked by section, so citations name the policy and section, such as "3. Annual Leave > 3.2 Carry-over". The policies are fictional, so the PDFs are generated from `data/source/*.md`. To regenerate them, run `python -m uv run python scripts/build_corpus_pdfs.py` from backend/; the output is byte-identical. To use your own documents, put text-based PDFs (or `.md`/`.txt` files) in `data/corpus/` and restart the backend. Scanned PDFs are rejected because they need OCR. [Design and limits](docs/decisions/0002-retrieval-and-generation.md#pdf-ingestion).

## Configuration

Environment variables override optional repo-root/backend dotenv files. [.env.example](.env.example) lists defaults. Relative CORPUS_DIR paths resolve from the repository root. OpenRouter requires a nonblank key at application startup; it is not used in extractive mode.

| Variable | Default | Purpose |
| --- | --- | --- |
| LLM_PROVIDER | extractive | Local quotation preview, or opt-in openrouter generation |
| EMBEDDING_MODEL | BAAI/bge-small-en-v1.5 | Local fastembed model |
| CORPUS_DIR | data/corpus | Directory of policy documents: PDF, Markdown or plain text |
| RETRIEVAL_TOP_K | 5 | RRF-fused evidence chunks |
| MIN_RELEVANCE | 0.735 | Cost-saving topical-similarity pre-filter |
| LLM_MAX_TOKENS | 500 | OpenRouter completion-token budget |
| LLM_TIMEOUT_SECONDS | 30 | SDK timeout per attempt; two retries can extend total duration |
| OPENROUTER_API_KEY | none | Required only for openrouter |
| OPENROUTER_BASE_URL | https://openrouter.ai/api/v1 | Optional generation endpoint |
| LLM_MODEL | openai/gpt-oss-20b | Optional generation model |
| LOG_LEVEL | INFO | Backend logs; SDK prompt debug logging remains disabled |

The threshold is near the midpoint between the highest unsupported and lowest supported calibration scores. It measures similarity, not answerability: a related but unsupported question can pass, and a paraphrase can be refused. OpenRouter's prompted LLM refusal is a separate answerability step. In a six-question live smoke test it refused both near misses that passed the pre-filter; that is evidence, not a measurement. A cross-encoder with a calibrated answerability decision is deferred because it needs another model download.

## Answer and citation behaviour

Extractive mode scores sentences in the top chunk by unique question-content-word overlap, using a fixed stopword list. It selects up to two highest-scoring sentences, breaks ties by earlier position, and presents them in document order. Zero overlap falls back to the first two sentences.

Output begins `Most relevant passage:`, includes [1], and states: "LLM generation is disabled; this text is quoted, not generated." This is evidence to inspect, not a claim that the passage answers the question. It cannot synthesize across sections or reliably distinguish closely related policy topics.

Only cited sources are returned. Numeric groups and hyphen/en-dash ranges are expanded, invalid numbers dropped, and references renumbered by first appearance. A fixed not-found reply or an output with no valid citations returns no citations. Syntactic citation validity does not prove semantic support.

The UI renders responses as a safe Markdown subset: paragraphs, emphasis, lists and code. Links and images are shown as plain text, so Markdown injected through a source document can't make the browser fetch a URL. Raw HTML is dropped.

Each response is labelled by its kind:
- "From company policies" or "Quoted from company policies", with a source count;
- "General knowledge, not from company policies";
- "Not covered by the policies", with example questions to try;
- small talk, which has no label.

Citation markers render as numbered chips; clicking one opens the matching collapsible source card. Example questions appear before the first question. The request ID isn't shown in the UI; it stays in the API response and the server log for tracing.

## Known issues

These issues remain unresolved in the frozen evaluation baseline; the bundled corpus is unaffected by the title-parsing edge cases.

- **Sentence selection:** the stray token "s" from "What's" skews scoring, so h11 omits SGD 220. [Details](docs/decisions/0002-retrieval-and-generation.md#sentence-selection).
- **Title parsing:** a separate title rule mishandles closing hashes, tabs, and headings inside code fences, producing incorrect titles or losing heading text. [Details](docs/decisions/0002-retrieval-and-generation.md#title-parsing).
- **Citation brackets (OpenRouter output):** numeric-looking brackets may disappear or become citations, and renumbered ranges need not be ascending. [Details](docs/decisions/0002-retrieval-and-generation.md#citation-brackets).
- **Stitched quotes:** selected sentences can be non-adjacent but are joined without an ellipsis, hiding omitted context. [Details](docs/decisions/0002-retrieval-and-generation.md#stitched-quotes).
- **PDF extraction limits:** scanned PDFs, tables and multi-column layouts aren't handled, headings are found by font size only, and citations name sections rather than pages. [Details](docs/decisions/0002-retrieval-and-generation.md#pdf-extraction-limits).
- **Single-turn only:** each question is answered on its own, so a follow-up like "And for part-time staff?" lacks context and is usually declined. Long-term memory is deliberately excluded. [Details](docs/decisions/0002-retrieval-and-generation.md#conversation-memory).
- **Fallback latency (OpenRouter):** small talk, general-knowledge and declined questions make two calls, grounded then fallback, so they take longer; one greeting took 12 seconds. [Details](docs/decisions/0002-retrieval-and-generation.md#answering-beyond-the-documents).

Four issues found during live testing are fixed:
- [phrase matching on LLM output](docs/decisions/0002-retrieval-and-generation.md#phrase-matching-on-llm-output);
- [Markdown display](docs/decisions/0002-retrieval-and-generation.md#markdown-in-llm-output);
- [`【1】` citations](docs/decisions/0002-retrieval-and-generation.md#lenticular-bracket-citations);
- the [relevance threshold in LLM mode](docs/decisions/0002-retrieval-and-generation.md#relevance-threshold-in-llm-mode).

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | /api/health | Liveness status and version |
| POST | /api/ask | Answer, citations, and request_id |

Requests contain a question of 1–2000 characters after trimming. Responses contain `answer`, `kind`, `citations`, and `request_id`. `kind` is one of `answer`, `quote`, `general`, `chat` or `not_found`, so clients can label a response without parsing its text; each citation has `doc_id`, `title`, `section`, `snippet`, and `score`. The score is dense cosine, not a probability.

Invalid input returns 422. In extractive mode, a message whose best evidence is below the threshold is declined. In OpenRouter mode every message is first answered from its sources. If that yields no cited answer, a source-free fallback may answer small talk or labelled general knowledge, or decline. A decline returns "I could not find the answer in the provided policy documents." Construction failures and generation errors/timeouts return generic 503 responses. Truncated OpenRouter completions also return 503; incomplete usage counts are omitted without discarding an otherwise complete answer. Unexpected errors remain 500 and are logged at ERROR.

One application log per answered request records request_id, retrieval/generation milliseconds, chunk count, route (grounded, general, or conversation), refusal, refusal_reason (low_relevance, model_not_found, or uncited), and available token counts. For errors, application log lines record only types and statuses, without question text or provider response bodies. Unexpected exceptions are re-raised; the server logs their traceback, including the exception message, which may contain sensitive text. Cold index construction is outside query timings.

## Tests and checks

From backend/:

~~~powershell
python -m uv run pytest
python -m uv run ruff check .
python -m uv run ruff format --check .
~~~

From frontend/:

~~~powershell
npm run lint
npm test
npm run build
~~~

Backend tests use deterministic fakes and SDK MockTransport, disable dotenv reads, and require no network or model downloads. Frontend tests (Vitest) check Markdown rendering, and that links, images and raw HTML are never rendered. Warnings are errors except PytestCacheWarning: inability to write pytest's optional cache must not make otherwise passing tests fail across Windows accounts.

## Evaluation

Both evaluations run locally with the cached embedding model; no hosted LLM is used. From backend/:

~~~powershell
$env:LLM_PROVIDER = "extractive"
$env:HF_HUB_OFFLINE = "1"
$env:FASTEMBED_CACHE_PATH = Join-Path $PWD "../.cache/fastembed"
python -m uv run python -m evals.run_retrieval_eval --output evals/retrieval_results.json
python -m uv run python -m evals.run_retrieval_eval --questions evals/retrieval_questions_heldout.jsonl --output evals/retrieval_results_heldout.json
~~~

Calibrate only on retrieval_questions.jsonl. Freeze all settings and sentence-selection rules before running the held-out set. The 20 calibration questions have 15 supported and 5 unsupported labels; the 13 reviewer-authored held-out questions have 7 supported paraphrases and 6 unsupported near misses.

A retrieval hit requires both document and full section path. MRR is truncated at k. Refusal metrics use their respective supported/unsupported denominators. Extractive answer-contains rate checks for the expected phrase in the post-processed answer, after normalizing case and typography (Unicode spaces, dash and quote variants, Markdown markers, whitespace); refusals count as misses, so its denominator is all answerable questions. Phrase presence is only a content probe, not full semantic correctness.

| Metric | Calibration (20 questions) | Held-out (13 questions) |
| --- | --- | --- |
| Hit@1 | 100% (15/15) | 85.71% (6/7) |
| Hit@5 | 100% (15/15) | 100% (7/7) |
| MRR@5 | 1.0000 | 0.9286 |
| Unanswerable refusal rate | 100% (5/5) | 33.33% (2/6) |
| Answerable false-refusal rate | 0% (0/15) | 42.86% (3/7) |
| Extractive answer-contains rate | 100% (15/15) | 42.86% (3/7) |

[Calibration detail](backend/evals/retrieval_results.json), [held-out detail](backend/evals/retrieval_results_heldout.json), and [HTTP smoke responses](backend/evals/smoke_results.json) contain the underlying evidence. [Decision 0002](docs/decisions/0002-retrieval-and-generation.md) describes calibration and limitations.

### Live-LLM evaluation

The same 33 questions ran through the full live pipeline under a budget guard. The guard reserves each question's worst-case cost at listed prices, covering both possible calls, and skips any question that could exceed the cap. A provider failure is recorded in its row and the run continues. From backend/, with a key in `.env`:

~~~powershell
python -m uv run python -m evals.run_retrieval_eval --provider openrouter --output evals/llm_results_r2.json --max-cost-usd 0.009
python -m uv run python -m evals.run_retrieval_eval --provider openrouter --questions evals/retrieval_questions_heldout.jsonl --output evals/llm_results_heldout_r2.json --max-cost-usd 0.006
~~~

Results with gpt-oss-20b on 2026-10-10. Round 1 had the threshold cutoff and no `【1】` support; round 2 is the current code.

| Metric | Calibration round 1 | Calibration round 2 | Held-out round 1 | Held-out round 2 |
| --- | --- | --- | --- | --- |
| Expected section cited | 15/15 | 15/15 | 1/7 | **7/7** |
| Answer-contains (normalized phrase) | 12/15 | 13/15 | 1/7 | **6/7** |
| Unanswerable refusal rate | 5/5 | 5/5 | 6/6 | 6/6 |
| Answerable false-refusal rate | 0/15 | 0/15 | 6/7 | **0/7** |

In round 2, every phrase miss is a correct paraphrase: q06, q14 and h12, checked against the raw replies saved in each row. No unanswerable question received a general-knowledge answer.

The fixes were prompted by round-1 held-out failures, so held-out round 2 is not a blind measurement. Treat it as evidence that the diagnosed causes were fixed, and use fresh questions to estimate quality.

Round 2 cost about $0.003, including one attempt lost to a provider error before failures were recorded per row. All live testing so far cost $0.0055, measured on the key's usage record. [Round 2 calibration](backend/evals/llm_results_r2.json), [round 2 held-out](backend/evals/llm_results_heldout_r2.json), [round 1 calibration](backend/evals/llm_results.json), [round 1 held-out](backend/evals/llm_results_heldout.json).

## Verification and remaining limitations

137 offline backend tests and 15 frontend tests pass. Backend lint/format and frontend lint/build pass. HTTP smoke content checks pass for carry-over and dental coverage, but **fail for the hotel question**: the selected quotation omits SGD 220. The saved smoke report marks this failure explicitly.

The protocol records one full held-out run with frozen settings and selection rules; the report itself has no run ID or timestamp. The hotel question (h11) was also exercised separately as a required smoke case before that run, so it was not blind; no tuning followed its failure. Of the 43 frozen files, 32 now differ in content. First, the 19 Python source files received comments and docstrings only: their syntax trees with docstrings removed are identical to the pre-edit snapshot. Four intended changes followed:
- the evaluator's phrase matching now normalizes typography;
- low-relevance messages get a conversation reply, and the evaluator gained a live-LLM mode;
- OpenRouter mode answers beyond the documents and accepts `【1】` citations;
- the corpus became PDFs, with new dependencies in the lockfile;
- the fictional company name was removed from each policy's first sentence.

None changes a saved extractive decision. Recomputing from the saved rows gives the same numbers, and the PDF corpus reproduces every chunk exactly. After the name change, every metric and per-question decision is unchanged; one similarity score moved by 0.01, still far from the threshold. The labels are byte-identical to the snapshot. Other later changes are documentation, evidence, frontend comments, Markdown rendering, and the UI heading from "Answer" to "Response". [Evidence limits](docs/decisions/0002-retrieval-and-generation.md#evaluation-protocol-and-results) distinguish the post-run hash snapshot from filesystem modification times.

Held-out results show that strong retrieval ranking does not ensure answerability or correct quoted content. See decision 0002 for the missed cases and evaluation provenance.

A budget-capped live OpenRouter smoke test (6 questions, $0.000287 spent against a $0.01 cap) answered all 3 supported questions correctly with the right citations, including the hotel cap that extractive mode misses, and refused all 3 unsupported ones: two by the model and one by the pre-filter. It is a smoke test on already-exposed questions, not a measurement. [Smoke evidence](backend/evals/live_llm_smoke_results.json) and [findings](docs/decisions/0002-retrieval-and-generation.md#live-llm-smoke-test).

## Design questions

1. **Why these architectural, product, and implementation choices?** A small local corpus suits an inspectable in-memory hybrid index; the extractive default avoids paid generation and shows evidence with citations. [Assumptions](docs/architecture.md#use-case-and-assumptions) and [retrieval trade-offs](docs/decisions/0002-retrieval-and-generation.md#hybrid-retrieval-with-rrf) explain the choices.
2. **What could go wrong, including edge cases and failure modes?** Similarity can accept unsupported questions or refuse supported paraphrases, and a correct section can still yield an incomplete quote. The [failure table](docs/architecture.md#failure-modes-and-edge-cases) covers these and operational failures.
3. **How would we know it works, measure it, and improve it?** Offline tests check contracts and errors; saved evaluations separately measure retrieval, refusal, and answer content, in both extractive and live-LLM modes, and keep raw model replies so failures can be diagnosed. [Evaluation and limits](docs/architecture.md#evaluation-and-limits) explains the blind spots and how new validation data would guide improvements.
4. **What are the quality, latency, cost, reliability, usability, and complexity trade-offs?** Local quotation avoids hosted-generation cost and network dependence but lacks synthesis and reliable answerability; in-memory search is simple but rebuilds on restart and scales linearly. [Trade-offs](docs/architecture.md#trade-offs) also cover cold-start latency and source inspection.
5. **How would the design evolve with scale, usage, data, or requirements?** Persist versioned indexes when startup or memory becomes costly, and improve retrieval, ingestion, and support checks when measured errors or new document types justify them. [Production triggers](docs/architecture.md#poc-versus-production) tie each change to a need.
6. **What would production need beyond this PoC?** Shared deployment needs access controls, budgets, managed secrets, privacy-aware monitoring, stronger evaluation, and controlled document updates. The [PoC versus production table](docs/architecture.md#poc-versus-production) separates these proposals from current capabilities.
7. **How do the components work together?** React sends a question through the Vite proxy to FastAPI; the pipeline retrieves evidence, applies the pre-filter, quotes or generates, then validates citation references. [Components and request flow](docs/architecture.md#current-state) show the interfaces and error paths.

## Requirements checklist

| Assessment requirement | Where it is met |
| --- | --- |
| Justify key decisions, trade-offs, assumptions and alternatives | [Decision records](docs/decisions/) and [assumptions](docs/architecture.md#use-case-and-assumptions) |
| Working web interface, from question to grounded response | [Run locally](#run-locally) and [Run with a real LLM](#run-with-a-real-llm-recommended-for-assessment) |
| Reliable, maintainable and measurable beyond a prototype | [Tests and checks](#tests-and-checks), [Evaluation](#evaluation), and error handling in [API](#api) |
| What could go wrong: edge cases and failure modes | [Failure table](docs/architecture.md#failure-modes-and-edge-cases) and [Known issues](#known-issues) |
| How to know it works, measure it and improve it | [Evaluation](#evaluation) and [evaluation and limits](docs/architecture.md#evaluation-and-limits) |
| Trade-offs: quality, latency, cost, reliability, usability, complexity | [Trade-offs](docs/architecture.md#trade-offs) |
| Evolution with scale, usage, data or requirements | [PoC versus production](docs/architecture.md#poc-versus-production) |
| Production versus proof of concept | [PoC versus production](docs/architecture.md#poc-versus-production) |
| How components work together | [Current state and request flow](docs/architecture.md#current-state) |
| Responsible use of the provided credits | All live runs were budget-capped; total spend was about $0.006 |
| README, `.env.example`, no committed secrets | This file and [.env.example](.env.example); `.env` is gitignored |

## Project layout and design

- backend/app/rag/: structured ingestion, embeddings, BM25/RRF, generation, citations, and request metrics.
- backend/app/api/, schemas.py, config.py: HTTP behaviour, contracts, and settings.
- backend/tests/ and evals/: offline regression tests and evaluation.
- data/corpus/: five fictional policies with overlapping topics, as PDFs.
- data/source/: the Markdown the PDFs are generated from (backend/scripts/build_corpus_pdfs.py).
- frontend/: React/Vite question form and section citations.
- docs/: [architecture](docs/architecture.md) and [decision records](docs/decisions/).

No Docker, auth, vector database service, persistent index, or extra UI workflow is included. Future work should follow measured errors: answerability checks, held-out coverage, a reranker, PDF/table parsing, document versioning, and persistence when corpus size warrants it.
