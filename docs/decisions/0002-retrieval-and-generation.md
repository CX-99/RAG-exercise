# 0002: Retrieval and generation

- **Status:** Superseded in part by [0003: Hosted models only](0003-hosted-models-only.md)
- **Date:** 2026-10-09

> **Read with 0003.** The app now uses hosted OpenRouter models for both embeddings and answers. The local embedding model (fastembed), the extractive quotation mode, `LLM_PROVIDER` and the `MIN_RELEVANCE` pre-filter were removed. Sections that describe them are kept as history and marked below. Chunking, PDF ingestion, hybrid retrieval with RRF, citation handling and answering beyond the documents still apply.

## Context

The PoC needs inspectable policy evidence and measurable trade-offs between quality, latency, and cost. Its five fictional policies deliberately overlap. Infrastructure and production deployment are outside scope.

## Structure-aware chunks

Markdown headings define sections and ancestry. The chunker omits its first recognized H1 as the title heading; subsequent H1s are top-level sections. The document loader extracts the title separately, with the inconsistencies listed under [title parsing](#title-parsing). The chunker requires whitespace before closing hash marks, preserving names such as C#, and ignores headings inside code fences; the title loader does not share those rules. Text is indexed as title — section path plus body; citation snippets preserve the raw chunk body. Heading-free text uses Introduction, missing loader titles use the filename stem, and duplicate stems are rejected.

The 1,200-character target is a soft limit at paragraph boundaries; oversized individual paragraphs stay intact. Alternatives were fixed windows (simple but sever qualifications), overlapping token windows (more continuity but duplicate evidence and prompt cost), and whole documents (less indexing but noisier, larger prompts). Repeated headings provide context without repeated body text.

At larger scale, evaluate token limits/overlap, and extend [PDF ingestion](#pdf-ingestion) with OCR, tables, multi-column reading order, and page references. Document versions and stable identities should precede incremental ingestion.

## PDF ingestion

The corpus is five text-based PDFs in `data/corpus/`, the format real policies usually arrive in. The policies are fictional, so `backend/scripts/build_corpus_pdfs.py` renders them from the Markdown kept in `data/source/`: headings in larger bold Helvetica, body text at 10 pt. Rendering uses reportlab's invariant mode, so regenerating gives byte-identical files.

**Approach.** `app/rag/pdf.py` converts each PDF to Markdown before chunking. This is the usual RAG ingestion pattern: the chunker, section paths and citations then work the same for every format.
- pdfplumber reports each character's font size. The size covering the most characters is body text.
- Each larger size, at least 1.15 times body size, becomes a heading level, largest first.
- Body lines are joined into paragraphs. A gap well above the line spacing inside paragraphs (its lower quartile) starts a new paragraph.
- A PDF that can't be parsed, or has no text layer, fails the index build loudly rather than being skipped.

**Alternatives considered:**
- **PyMuPDF / pymupdf4llm:** better layout handling, but AGPL-licensed.
- **Docling or Unstructured:** layout models and OCR, but large model downloads.
- **pypdf:** BSD-licensed, but it gives no per-character font data, so headings would be lost.
- **A hosted document-intelligence service:** strongest on scans and tables, but adds cost, network dependence and data sharing.

**Verification:**
- Converting the five PDFs back gives exactly the 36 chunks the Markdown sources give, with the same sections and text.
- Both extractive evaluations on the PDF corpus match the Markdown results exactly: every aggregate, every row and every similarity score, with a difference of 0. The saved live-LLM results therefore also apply.
- Tests cover the round trip, a heading separated from its body by a page break, a paragraph flowing across pages, scanned and unreadable PDFs, and corpus parity.

The [PDF extraction limits](#pdf-extraction-limits) are listed under known issues.

## Hybrid retrieval with RRF

> **Superseded by [0003](0003-hosted-models-only.md).** Dense vectors now come from `openai/text-embedding-3-small` through OpenRouter, not from a local model. BM25 and RRF are unchanged.

Fastembed BAAI/bge-small-en-v1.5 produces local vectors, normalized for NumPy cosine search. BM25 uses lowercase word tokens over the same prefixed chunks. RRF sums 1/(60 + rank), beginning at one; the default returns five results with their dense cosine scores. Corpus order breaks ties. Nonpositive BM25 scores supply no lexical evidence and are omitted from that ranking.

Dense-only search handles paraphrases but can miss exact terms; sparse-only search is cheap but vocabulary-sensitive. Weighted score addition needs scale calibration. RRF combines rankings cheaply without normalizing incompatible scores, but can promote a lexical match with weak semantic relevance.

A cross-encoder reranker could improve question–passage assessment, with additional latency and a new local model. It is deferred under the current no-new-model constraint. A calibrated support/answerability decision would still need evaluation; reranking alone is not proof of entailment.

## In-memory index and construction

Chunks, vectors, and BM25 are built lazily once per process. Initialized requests return the pipeline directly, without a lock or worker-thread hop. First requests use a lock with a second cache check; construction and retrieval execute through asyncio.to_thread. The local embedding adapter serialized model use; the hosted client in [0003](0003-hosted-models-only.md) needs no lock.

Construction errors become a domain exception carrying only the original exception type. The HTTP layer returns a generic 503 and logs that type; failed builds are not cached, permitting retry. Settings rejects missing OpenRouter credentials at app creation, before a request or index build.

A persistent local index, ANN library, or vector service could reduce startup/search costs at scale but adds invalidation, model-version compatibility, tuning, and operations. Flat search is transparent and linear; each worker owns a copy, and corpus edits require a restart. Query timings exclude cold dependency construction. At larger scale, build immutable versioned indexes outside requests and persist embeddings keyed by model/document/chunker version.

## Similarity as a cost-saving pre-filter

> **Superseded by [0003](0003-hosted-models-only.md).** The pre-filter and `MIN_RELEVANCE` were removed; the model's own refusal decides answerability. Kept as history.

The highest dense score among returned top-k chunks is compared with MIN_RELEVANCE. Extractive mode cannot judge whether a passage answers a question. So with an empty result or a score below the threshold it quotes nothing, declines, and records low_relevance. OpenRouter mode no longer applies the threshold: every message is first answered from its sources and the model decides, as described in [answering beyond the documents](#answering-beyond-the-documents). Equality passes. The first RRF result need not have the highest dense score.

This measures topical similarity, not whether the evidence answers the question. Always generating spends more latency/tokens; thresholding RRF lacks absolute relevance; an answerability classifier needs more labels and modelling. The pre-filter is cheap but accepts some near misses and refuses some paraphrases. In OpenRouter mode the prompted LLM must separately refuse unsupported questions. Extractive mode has no semantic answerability step and presents its selection explicitly as a quotation. A cross-encoder with a calibrated answerability decision is deferred because it needs another model download.

MIN_RELEVANCE remains 0.735 based only on calibration. Supported calibration scores span 0.747207–0.902533; unsupported scores span 0.586381–0.722298. The midpoint is 0.734752, leaving about 0.012 margin either side when rounded to 0.735. At 0.50 no calibration negatives are refused; at 0.70, 3/5 are refused; at 0.735, 5/5 are refused with 0/15 false refusals; at 0.75, one supported question is also refused. A narrow in-sample gap is not evidence of general answerability.

## Local quotation and optional generation

> **Superseded by [0003](0003-hosted-models-only.md).** Extractive quotation was removed and generation is the only mode. The generation settings below still apply, except two: the token limit is now 2,000, and a cut-off reply returns `kind=too_long` instead of a 503 ([answer length limit](0003-hosted-models-only.md#answer-length-limit)).

ExtractiveLLM splits the top chunk into sentences, strips source-authored citation groups, and scores each sentence by unique content-word overlap with the question. It reuses retrieval.tokenize and a fixed generic stopword list. It selects up to two highest-scoring sentences, prefers earlier sentences on ties, then emits them in document order. All-zero scores fall back to the first two. There is no stemming, synonym expansion, or paraphrase model for selection.

Output starts "Most relevant passage:" with [1], followed by "LLM generation is disabled; this text is quoted, not generated." This is inspectable evidence, not a generated assertion that the passage answers the question. Compared with always taking the opening sentences it can find facts later in a section at negligible local cost. It can still miss a fact, show irrelevant text, or combine separated sentences without their intervening context.

Hosted generation could improve synthesis and fluency but adds credentials, cost, and network-dependent testing. A local generative model avoids paid inference but adds downloads/hardware demands. The optional OpenRouter client uses AsyncOpenAI, the configured endpoint/model, temperature zero, 500 max tokens, a 30-second SDK timeout, and two retries. The timeout applies per attempt. Prompts are centralized and require source-only claims, citations, explicit not-found responses, and treating source text as data.

**Reasoning-token caveat:** The default openai/gpt-oss-20b is a reasoning model. In the [live smoke test](#live-llm-smoke-test), providers counted reasoning tokens inside completion_tokens (up to 178 of 180). Every reply finished with finish_reason=stop, and the largest completion used 180 of the 500-token limit. Whether max_tokens also caps reasoning was not observed, because no reply came near it. (Later observed: it does. A whole-policy request hit the 500-token limit, so the limit was raised to 2,000; see [0003](0003-hosted-models-only.md#answer-length-limit).) No real LLM is needed for the evaluations or tests.

finish_reason=length raises TruncatedCompletion, mapped to 503, rather than displaying partial output. Missing/null usage counts yield usage=None while preserving a complete answer. Missing credentials fail in both Settings and the client. SDK tests use MockTransport with fake credentials. Future quality work includes prompted refusal, claim support, prompt injection, request budgets, cancellation, and managed client lifecycles.

## Answering beyond the documents

In OpenRouter mode the assistant tries to answer every message, in two steps:
1. **Grounded:** the message and its top 5 sources go to generation, as before. A reply with a valid citation is returned as the answer (`route=grounded`).
2. **Fallback:** if the reply declines or has no valid citation, the message goes to `LLMClient.converse` alone, with no sources. The conversation prompt allows three outcomes:
   - a short reply to greetings, thanks and questions about the assistant, with an example question (`route=conversation`);
   - a brief answer to a general-knowledge question that isn't about the company, starting with the fixed label "General knowledge (not from the company policies):" (`route=general`);
   - the not-found reply for any question about the company's own rules, benefits, amounts, deadlines or procedures, which the model must never guess. The refusal reason comes from the grounded step.

Fallback replies carry no citations, and any `[n]` marker in them is removed, because no source was shown. Extractive mode, since removed ([0003](0003-hosted-models-only.md)), declined below-threshold messages instead.

**Alternatives considered:**
- **Refuse anything the documents don't cover:** the most conservative option, but unhelpful for greetings and general questions.
- **Let the model answer freely:** it would also answer company questions from what is "typical", which misleads employees about *this* company's rules.
- **One combined prompt:** an uncited general answer couldn't be told apart from an ungrounded policy claim.
- **Similarity routing** (the earlier design): it blocked answerable paraphrases; see [the threshold fix](#relevance-threshold-in-llm-mode).

**Costs and risks:**
- Every message now costs a grounded call, about $0.00006.
- Small talk, general questions and declines add a second call. That doubles their cost and latency; one greeting took 12 seconds through two calls.
- **Mislabelled general answers:** the model could still answer a company question with a general-knowledge label. Three things limit that: the prompt forbids it, the label makes any such answer visibly non-policy, and the evaluation measures it. In round 2 none of the 11 unanswerable company questions received a general answer.

## Conversation memory

Each question is answered on its own: the assistant keeps neither the current conversation (short-term memory) nor anything about a user across sessions (long-term memory).

**Short-term memory: deferred.** A follow-up such as "And for part-time staff?" lacks its subject, so it is retrieved and answered as written. That usually ends in a decline rather than a wrong answer. The standard fix is history-aware query rewriting:
- the client sends the last few turns;
- the LLM rewrites the follow-up into a standalone question;
- that question goes through the normal retrieval and citation flow.

That costs one short extra call per follow-up, a chat-style UI, multi-turn evaluation cases, and a new failure mode, a wrong rewrite. The brief asks for a complete question-to-grounded-answer flow, which a single turn shows clearly, so this is the first addition if users turn out to ask follow-ups.

**Long-term memory: deliberately excluded.** Remembering users across sessions would need authentication and persistent storage, both outside this PoC. It would also mean storing questions that can contain personal details, which this design avoids even in logs. And it conflicts with grounding: answers should come from the current policies, and a remembered answer can go stale when a policy changes. Per-user preferences, if ever needed, should be explicit settings rather than inferred memory.

## Citation handling and observability

Numeric brackets, comma/space groups, and hyphen/en-dash ranges are recognized. Ranges are expanded only within available source bounds, invalid references are dropped, and duplicate references within a group are collapsed. Citations are renumbered by first appearance across the whole answer. Brackets outside the numeric citation pattern, such as [a label], remain when a valid citation survives elsewhere. Explicit NOT_FOUND becomes model_not_found; no valid reference becomes uncited. Both return no citations.

Returning all retrieved chunks misstates which sources support a reply; preserving original numbering creates UI gaps; structured model output offers stronger syntax control with more schema/provider coupling. Current handling is deterministic and cheap but validates identity/syntax only. Valid citations can still accompany unsupported claims. At larger scale, add semantic claim-support checks and document/version/page links.

Request-local metrics retain the answer(question) protocol and record refusal_reason, query timings, chunk count, and complete token usage when available. LLM failures/timeouts become generic 503s. Unexpected exceptions log their type at ERROR and are re-raised as 500s. For errors, application log lines record only types and statuses, without question text or provider response bodies; SDK debug logging is disabled. Unexpected exceptions are re-raised through Starlette, and the server logs their traceback, including the exception message. Sensitive question or provider text can therefore reach server logs if an unexpected exception contains it; application log fields alone do not guarantee log privacy.

## Known issues

These issues remain unresolved in the frozen evaluation baseline. Any future correction needs separate validation and fresh held-out evidence.

### Sentence selection

> **Superseded by [0003](0003-hosted-models-only.md).** Extractive mode was removed, so this issue no longer applies. Kept as history.

The tokenizer leaves the fragment "s" in "What's", and the fixed stopword list does not remove it. In h11, it matches the possessive in "Director's", promoting the hotel-approval sentence over the sentence containing SGD 220. The correct section ranks first, but the displayed quotation misses the requested cap. The [saved hotel smoke](../../backend/evals/smoke_results.json) and [held-out h11 result](../../backend/evals/retrieval_results_heldout.json) record the failure.

### Title parsing

documents.py searches for the first line matching `^# (.+)$` independently of the chunker's heading/fence logic. This can put the wrong title into indexed text and citations or lose the intended title:

- `# Leave Policy #` produces the title "Leave Policy #"; the loader retains the closing hash.
- `#\tLeave Policy` (a tab after the hash) falls back to the filename stem. The chunker recognizes that first H1 and omits it from the body/section path, so its heading text is lost.
- A `# ...` line inside a code fence before the real H1 becomes the title. The chunker ignores the fenced line as a heading and omits the first real H1, so the intended title disappears from title/section metadata and chunk body.

The bundled corpus uses ordinary space-separated H1 titles without these edge cases and is unaffected. Sharing a title/heading parser would be future work.

### Citation brackets

In OpenRouter-generated output, any nonempty bracket group containing only digits, commas, whitespace, and hyphen/en-dash characters is treated as citation syntax. With the default five sources, `[2024-2025]` is out of range and removed; `[ ]` is also removed. In-range prose such as "clause [2]" becomes a source citation rather than remaining a clause reference.

Renumbering follows first appearance, so an expanded range need not display in ascending order. With three sources, `First [2]. Next [3]. Last [1-2].` becomes `First [1]. Next [2]. Last [3, 1].` If no valid citation survives anywhere, the whole response becomes NOT_FOUND. These rules can erase meaningful numeric prose or misrepresent it as supporting evidence.

These generated-output examples concern OpenRouter. Extractive mode inserts its own [1] after stripping matching numeric groups from the source text, so its displayed quotation also does not preserve such source brackets; the returned citation snippet retains the raw chunk. Context-aware citation syntax or structured references would be future work.

### Stitched quotes

> **Superseded by [0003](0003-hosted-models-only.md).** Extractive mode was removed, so this issue no longer applies. Kept as history.

The selector restores document order after choosing its top sentences, then joins them with a space. Non-adjacent sentences have no ellipsis or omission marker. A reader may mistake them for consecutive text and miss an intervening qualification; the source snippet is available for comparison. Explicit omission markers or contiguous spans would be future work.

### PDF extraction limits

The converter handles text-based PDFs with size-distinguished headings, like the generated corpus:
- **Scans:** a scanned PDF has no text layer and is rejected; it needs OCR.
- **Layout:** tables and multi-column layouts aren't reconstructed, so their reading order can be wrong.
- **Headings:** they are recognized by font size only. A bold heading at body size is missed, and heading levels are relative within each document.
- **Page boundaries:** two paragraphs split exactly at a page boundary merge into one. Only chunk packing is affected, not the text.
- **Citations:** they name sections, not page numbers.

### Fallback latency

In OpenRouter mode, messages the documents don't answer make two calls: grounded, then fallback. Small talk, general-knowledge questions and declines therefore take longer and cost about twice as much; one greeting took 12 seconds. A cheap pre-check for obvious small talk could skip the grounded call, but its cutoff would need fresh validation questions.

## Fixed after the held-out run

The first two issues surfaced in the [live smoke test](#live-llm-smoke-test), the last two in round 1 of the [live-LLM evaluation](#live-llm-evaluation). None changes a saved extractive result.

### Phrase matching on LLM output

**Problem:** run_retrieval_eval.py matched expected phrases with a plain case-insensitive substring test. gpt-oss-20b wrote `31 March`, `SGD 220` (U+202F narrow no-break space) and `multi‑factor` (U+2011 non-breaking hyphen), so all three correct answers failed the check.

**Fix:** `normalize_for_match` applies NFKC, maps dash and curly-quote variants to ASCII, drops Markdown emphasis and code markers, collapses whitespace, and casefolds. It runs on both the phrase and the answer. Tests cover the exact forms the model produced, and also check that different wording still fails.

Recomputing from the saved report rows gives the same results: calibration 15/15 and held-out 3/7, with zero rows changing. Neither evaluator was rerun, and the three live answers now pass. This was the first code change to a frozen file after the held-out run; the protocol records it under `evaluator_normalization_change`. The second, the [conversation path and answering beyond the documents](#answering-beyond-the-documents), with a live-LLM evaluation mode, is recorded under `conversation_path_and_llm_evaluation`.

### Markdown in LLM output

**Problem:** the model formats answers with Markdown, for example `**up to 5 unused annual leave days**`, and the UI showed the asterisks literally.

**Fix:** the UI renders responses with react-markdown, limited to paragraphs, emphasis, lists, code and blockquotes. Links are unwrapped to plain text, and images are dropped. Otherwise a prompt-injected source could make the model emit an image whose URL leaks data when the browser loads it. Raw HTML is dropped.

Asking for plain text in the prompt was rejected as the main fix: models don't reliably comply, and it would change the frozen prompt. Six Vitest tests cover rendering and the security limits, and all 35 saved answers render with no text lost. The cost is about 35 KB more gzipped JavaScript.

### Lenticular-bracket citations

**Problem:** in round 1, gpt-oss-20b cited three correct held-out answers (h09, h10, h11) with CJK lenticular brackets, `【1】` (U+3010/U+3011), instead of `[1]`. `CITATION_GROUP` accepted only ASCII brackets, so those answers were replaced by the not-found reply with refusal_reason=uncited.

**Fix:** `normalize_brackets` rewrites lenticular and fullwidth brackets as ASCII before citations are parsed. Tests cover `【1】`, `［1］`, and a not-found reply carrying `【2】`. In round 2 all three answers are kept and cite the expected section.

### Relevance threshold in LLM mode

**Problem:** MIN_RELEVANCE = 0.735 was calibrated for extractive mode, which has no other way to refuse. In round 1 it kept the sources of 3 held-out paraphrases (h07, h12 and h13, scoring 0.635–0.694) away from the model, so all three were declined.

**Fix:** the threshold now applies only to clients that can't judge answerability. In OpenRouter mode every message is answered from its sources first; see [answering beyond the documents](#answering-beyond-the-documents). This removes a cutoff rather than tuning one, but the change was prompted by round-1 held-out failures. Held-out round 2 therefore confirms the fix and is not a blind estimate.

## Evaluation protocol and results

> **Superseded by [0003](0003-hosted-models-only.md).** These are the extractive-mode baseline results, kept as history. Current results are in [0003](0003-hosted-models-only.md#results).

Calibration has 15 answerable and 5 unsupported questions. Held-out data are the reviewer's unchanged 13 questions: 7 supported paraphrases and 6 unsupported near misses. Every answerable row requires a nonblank expected_answer_contains phrase; unsupported rows cannot carry that label. Calibration phrases are exact quotes from their labelled sections.

Hits require exact document and full section path. MRR is truncated at k. Refusal rates use separate supported/unsupported denominators. For accepted answerable questions, evaluation runs the actual ExtractiveLLM and process_citations, then checks for the phrase after normalizing case and typography. All answerable questions form the denominator, including refusals as misses. Unsupported accepted questions are counted in refusal metrics without running extraction. Phrase presence is not a semantic correctness guarantee.

| Metric | Calibration (20 questions) | Held-out (13 questions) |
| --- | --- | --- |
| Hit@1 | 100% (15/15) | 85.71% (6/7) |
| Hit@5 | 100% (15/15) | 100% (7/7) |
| MRR@5 | 1.0000 | 0.9286 |
| Unanswerable refusal rate | 100% (5/5) | 33.33% (2/6) |
| Answerable false-refusal rate | 0% (0/15) | 42.86% (3/7) |
| Extractive answer-contains rate | 100% (15/15) | 42.86% (3/7) |

The implementation, threshold (0.735), stopword list, corpus, and calibration labels were frozen before the full held-out evaluation. The 43 source/configuration/data/test hashes were recorded immediately after the held-out run: they are a post-run snapshot, not a pre-run attestation. File modification times showed that no frozen file changed between the held-out report and that snapshot. Modification times are filesystem evidence, not an immutable history of edits.

**Comment-only revision (2026-10-10):** Docstrings and code comments were later added to the 19 frozen Python source files. A snapshot of each file's abstract syntax tree, with docstrings removed, was taken before editing and compared afterwards. Comments are not part of the syntax tree, and all 19 files show no code difference. All 94 tests then in the suite still passed, and the frontend production bundle was byte-identical. The protocol keeps the original 43-file snapshot and records the revised hashes and verification under `comment_only_revision`. At that point the other 24 frozen files, including the corpus, labels, tests and lockfile, were byte-identical to the snapshot. The later evaluator fix changed one of them, tests/test_eval.py. Docstrings now appear as OpenAPI descriptions in /docs; API responses are unchanged. Two code changes followed: [phrase matching on LLM output](#phrase-matching-on-llm-output) and the [conversation path](#answering-beyond-the-documents).

The held-out report has no run ID or timestamp; its elapsed_seconds is a duration, not a run timestamp. The statement that the full evaluator ran exactly once rests on the protocol record, not on independently timestamped run records. Evidence-file hashes for the calibration, held-out, and smoke reports were recorded during the final documentation pass; they identify the current artifacts but do not prove their earlier history or run count. No frozen file changed in that pass. Later changes are documentation, evidence, the comment-only revision above, the evaluator normalization and Markdown rendering fixes, the conversation path and live-LLM evaluation mode, frontend comments, and the UI heading from "Answer" to "Response".

**Exposure caveat:** h11 is also the required hotel HTTP smoke question and was exercised before the full held-out pass. Its failure was inspected: tokenization leaves the contraction fragment "s" in "What's", which matches the possessive in "Director's" and promotes the approval sentence over the SGD 220 sentence. The selector was deliberately left frozen. Thus h11 was not a blind question, although no tuning followed its exposure. Separate HTTP smoke requests also repeat h11; the protocol records one complete held-out evaluator run.

Held-out labels are unchanged and there are no label disagreements. The content misses are h07, h12, and h13 (refused despite supported labels), plus h11 (accepted quotation omits the labelled hotel cap). Unsupported h01, h02, h04, and h05 pass the similarity pre-filter. These results do not support using cosine similarity alone as an answerability decision.

Calibration has perfect ranking and refusal metrics on its in-sample questions, while the held-out pre-filter accepts four of six unsupported near misses and refuses three of seven supported paraphrases. Quotation labelling and syntactic citation checks do not solve that semantic problem; the saved reports measure the current frozen baseline.

[Calibration report](../../backend/evals/retrieval_results.json), [held-out report](../../backend/evals/retrieval_results_heldout.json), and [freeze/protocol evidence](../../backend/evals/evaluation_protocol.json) preserve the measurements and source hashes.

Both saved reports identify BAAI/bge-small-en-v1.5, top_k=5, and MIN_RELEVANCE=0.735. The protocol records extractive operation with HF_HUB_OFFLINE enabled. Dependency versions are pinned in backend/uv.lock; the saved report metadata do not independently identify a model-weight snapshot.

## Live LLM smoke test

On 2026-10-10 the opt-in OpenRouter path ran once on 6 questions with a $0.01 budget cap. The run used the unmodified pipeline code, gpt-oss-20b at temperature 0, and a 500-token limit. A worst-case cost guard covered every call, including SDK retries. Total spend was $0.000287; the providers' reported costs and the key's usage record agree. Four questions (h01, h04, h09, h11) are already-exposed held-out items, so this is a smoke test, not a measurement, and it does not change the frozen results.

| Question | Result |
| --- | --- |
| q02 carry-over | Correct: 5 days, use by 31 March, written exception; cites Annual Leave 3.2 |
| h11 hotel limit | Correct: SGD 220 per room per night; cites Travel and Expenses 2.2 (extractive mode misses this) |
| h09 two-factor | Correct: multi-factor authentication required; cites Information Security 2.1 |
| h01 sick leave (near miss) | Passed the pre-filter; the model returned the not-found reply |
| h04 day-trip meals (near miss) | Passed the pre-filter; the model returned the not-found reply |
| q16 dental | Refused by the pre-filter; no LLM call |

Observations:
- The prompted refusal caught both near misses the pre-filter let through. Two questions cannot establish a refusal rate.
- Plain phrase matching failed all three correct answers because of Unicode spacing and hyphens. This is now fixed; see [phrase matching on LLM output](#phrase-matching-on-llm-output).
- Answers used Markdown, which the UI now renders safely; see [Markdown in LLM output](#markdown-in-llm-output).
- OpenRouter routed the 5 calls to 5 different providers (AkashML, CoreWeave, DeepInfra, DekaLLM, Google). Cost per call varied about fourfold (USD 0.000031–0.000118) and latency ran from 0.95 to 4.3 seconds. Reproducible LLM-mode evaluation should pin a provider through OpenRouter's routing options.

Per-question details are in [the smoke evidence](../../backend/evals/live_llm_smoke_results.json); the API key was never printed or stored.

## Live-LLM evaluation

On 2026-10-10 the calibration and held-out sets ran once through the full pipeline with the live model (gpt-oss-20b, temperature 0, 500-token limit), including the conversation path. A budget guard reserved each call's worst-case cost at listed prices: $0.006 for calibration and $0.004 for held-out. The run spent $0.0015 for 33 calls, measured on the key's usage record; token-priced at list it would have been $0.0009. Each row keeps the raw model reply.

| Metric | Calibration (20 questions) | Held-out (13 questions) |
| --- | --- | --- |
| Expected section cited | 100% (15/15) | 14.29% (1/7) |
| Answer-contains (normalized phrase) | 80% (12/15) | 14.29% (1/7) |
| Unanswerable refusal rate | 100% (5/5) | 100% (6/6) |
| Answerable false-refusal rate | 0% (0/15) | 85.71% (6/7) |
| Refusal reasons | low_relevance 5 | low_relevance 5, model_not_found 4, uncited 3 |

Reading the raw replies:
- The calibration phrase misses (q06, q11, q14) are correct paraphrases; for example "within **1 hour** of discovering it" against the label "within 1 hour of discovery". Exact-quote labels suit extractive quotations, not generated prose, so answer-contains understates LLM correctness. Short key-fact labels or graded human review would measure it better.
- The held-out false refusals have two causes: [lenticular-bracket citations](#lenticular-bracket-citations) (h09, h10, h11) and the [relevance threshold](#relevance-threshold-in-llm-mode) (h07, h12, h13).
- Unanswerable questions were all declined: four by the model after grounded generation, and the rest on the conversation path.

The held-out questions were already exposed, and the run is a single sample with provider routing that varies per call. Treat these figures as the second configuration's results, not a blind estimate. [Calibration report](../../backend/evals/llm_results.json) and [held-out report](../../backend/evals/llm_results_heldout.json).

### Round 2, after the fixes

Both round-1 causes were fixed: [`【1】` citations](#lenticular-bracket-citations) and [the LLM-mode threshold](#relevance-threshold-in-llm-mode). The fallback was also extended to [answer beyond the documents](#answering-beyond-the-documents). The same 33 questions then ran again, with the guard reserving both possible calls per question: $0.009 for calibration and $0.006 for held-out.

| Metric | Calibration round 1 | Calibration round 2 | Held-out round 1 | Held-out round 2 |
| --- | --- | --- | --- | --- |
| Expected section cited | 15/15 | 15/15 | 1/7 | 7/7 |
| Answer-contains (normalized phrase) | 12/15 | 13/15 | 1/7 | 6/7 |
| Unanswerable refusal rate | 5/5 | 5/5 | 6/6 | 6/6 |
| Answerable false-refusal rate | 0/15 | 0/15 | 6/7 | 0/7 |
| Refusal reasons | low_relevance 5 | model_not_found 5 | low_relevance 5, model_not_found 4, uncited 3 | model_not_found 6 |

The remaining phrase misses (q06, q14, h12) are correct paraphrases, judged from the raw replies, so every answerable question in both sets was answered correctly with the expected citation. None of the 11 unanswerable questions received a general-knowledge answer.

The first held-out attempt was lost to a provider error that aborted the run; that cost about $0.001. `evaluate_pipeline` now records such failures in the row and continues, and a regression test covers it. Round 2 cost about $0.003 in total.

Because the fixes were prompted by round-1 held-out failures, held-out round 2 is not a blind measurement. Estimating quality needs fresh questions. [Round 2 calibration](../../backend/evals/llm_results_r2.json) and [round 2 held-out](../../backend/evals/llm_results_heldout_r2.json).

## Verification

> **Superseded by [0003](0003-hosted-models-only.md).** This records the extractive-era state; extractive mode and its hotel smoke failure no longer exist. Current checks are in the [overview](../overview.md#tests-and-checks).

137 offline pytest tests and 15 frontend Vitest tests pass, as do ruff check, ruff format --check, frontend lint, and frontend build. Tests use fakes and SDK MockTransport; no test calls a live LLM.

All three HTTP smoke requests return 200. Carry-over quotes the expected Annual Leave 3.2 passage; dental coverage returns NOT_FOUND without citations. **The hotel content check fails:** the quotation does not contain SGD 220. [Smoke evidence](../../backend/evals/smoke_results.json) records content_checks.hotel=false and all_content_checks_passed=false. The known contraction-fragment issue remains unfixed to preserve the frozen evaluation. It is future selection work to test on a new validation set, not a passed acceptance criterion.

Pytest treats warnings as errors except PytestCacheWarning, because Windows accounts may not share write access to its optional cache. This exception affects cache bookkeeping only. The UI result heading is "Response" to cover both local quotations and optional generated answers; the remaining known issues are documented without changing the frozen implementation.
