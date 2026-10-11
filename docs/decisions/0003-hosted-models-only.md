# 0003: Hosted models only

- **Status:** Accepted
- **Date:** 2026-10-11
- **Supersedes in part:** [0002](0002-retrieval-and-generation.md). The local embedding model, the extractive quotation mode and the similarity pre-filter are replaced. Chunking, hybrid retrieval, citations and the answer-beyond-the-documents fallback are unchanged.

## Context

Until now the app had two modes, selected by `LLM_PROVIDER`:
- **Extractive (the default):** offline. It quoted up to two sentences from the top section, and declined anything below a similarity threshold (`MIN_RELEVANCE=0.735`).
- **OpenRouter (opt-in):** generated answers.

Both modes embedded text locally with fastembed (`BAAI/bge-small-en-v1.5`).

This had three problems:
- **Reviewers saw the weaker mode.** Anyone following the README got quotations, not answers. On the held-out set, extractive mode refused 3 of 7 answerable paraphrases, accepted 4 of 6 unanswerable near misses, and contained the expected fact in only 3 of 7 answers. The generated path did far better on the same questions (0 false refusals, 7/7 correct citations), but it was off by default.
- **Two code paths for one product.** Each mode had its own routing, prompts, response kind (`quote`), tests and documentation. The threshold already applied only to extractive mode.
- **The local model added setup.** fastembed downloads its model on first use. Running without network access needed cache variables (`FASTEMBED_CACHE_PATH`, `HF_HUB_OFFLINE`).

The user asked for one path that uses real API calls only.

## Decision

Use OpenRouter for both embeddings and generation, with no offline mode.

- **Embeddings:** `openai/text-embedding-3-small` through OpenRouter's OpenAI-compatible `/embeddings` endpoint.
  - `OpenRouterEmbedder` uses the synchronous OpenAI SDK client, because retrieval already runs in a worker thread.
  - It shares generation's timeout (30 s per attempt) and retry count (2).
  - Rows are put back in input order by their `index`. A wrong row count is an error, not a silent mismatch.
  - Provider errors and timeouts become the same `LLMError` as generation failures. The API returns a generic 503 and logs only the error type and status.
- **Generation:** unchanged. `openai/gpt-oss-20b` at temperature 0, with 500 max tokens (raised to 2,000; see [answer length limit](#answer-length-limit)). Every message is first answered from its top 5 sources; if that yields no cited answer, a source-free fallback handles small talk, labelled general knowledge, or the decline.
- **The API key is required.** `Settings` rejects a missing or blank `OPENROUTER_API_KEY` at startup with a clear message, before any request.
- **Removed:**
  - `ExtractiveLLM` and its sentence selection;
  - the `LLM_PROVIDER` and `MIN_RELEVANCE` settings;
  - the `quote` response kind;
  - fastembed, with its cache settings.
- **Tests stay mocked.** They set a fake key, block outside network access, and use `httpx2`'s `MockTransport` in place of the API. They cost nothing and need no key.
- **The evaluator is live-only.** Its budget guard now also reserves the cost of embedding the corpus and each question, at OpenRouter's listed prices.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Keep extractive mode as a no-key fallback | It runs without a key, but it would remain a second product with its own failure modes. It quotes rather than answers, and its held-out results were much weaker. |
| Hosted LLM with local fastembed embeddings | Retrieval would need no network and no embedding cost. But it keeps the model download and cache settings to save very little: indexing the corpus through the API costs about $0.00013. |
| `openai/text-embedding-3-large` | Larger vectors and 6.5 times the price. Retrieval with the small model is already perfect on both evaluation sets, so there is nothing to gain at this corpus size. |
| Keep a similarity threshold | The 0.735 cutoff was calibrated on another model's scores, so it doesn't transfer. It had already blocked answerable paraphrases in LLM mode; the model's own refusal decides answerability better. |

## Consequences

- **A key and network access are required.** Without a key the backend refuses to start. If OpenRouter is unreachable, questions return 503; there is no offline demo.
- **Cost:**
  - Indexing the corpus (36 chunks, about 6,700 tokens) costs about $0.00013. It runs once per backend start, because the index lives in memory.
  - Each question adds one query embedding, which costs a negligible amount, plus one or two chat calls of about $0.00006 each.
- **Cold start:** the first question also embeds the corpus over the network. It took 7.3 seconds in live testing, including generation.
- **Reliability:** both models depend on one gateway. A failed index build isn't cached, so a later request retries it.
- **Privacy:** questions already reached the generation provider. Now the embedding provider also receives each question, and the policy text when the corpus is indexed.
- **Scores change scale:** citation scores are cosine similarities from the new model. The 0.735 figures in 0002 apply only to the old model.
- **History stays readable:** the extractive and round 1–2 results remain in `backend/evals/` and in 0002 as labelled history. The evaluation protocol records this change under `hosted_models_only`.

## Answer length limit

*Added 2026-10-11, after live use.*

**Problem:** "return me the annual leave policy details" returned a 503 twice. The log showed `TruncatedCompletion`: the reply hit the 500-token limit. Two things combined:
- The question asks for a whole policy, so the answer is long.
- gpt-oss-20b is a reasoning model. Its hidden reasoning counts toward the same limit, so less than 500 tokens remain for the answer.

The pipeline deliberately never shows a cut-off reply, which could lose a qualification or a citation. But it reported the cut-off with the same generic 503 as an outage, and that message ("try again shortly") was wrong, because a retry hits the same limit.

**Decision:**
- **Raise the default `LLM_MAX_TOKENS` to 2,000.** Billing counts tokens actually generated, so replies that already fit cost the same. A full 2,000-token reply costs about $0.0002.
- **A reply cut off by the limit, in either call, becomes `kind=too_long`.** The UI asks for a narrower question and offers examples. A cut-off reply is still never shown.
- **The cut-off call's tokens are logged** (`refusal_reason=truncated`), because they are billed. The evaluator counts `too_long` as a refusal.

**Alternatives considered:**
- **Show the partial reply with a warning:** this is common in chat apps, but a policy answer cut off mid-sentence can drop a condition or its citation.
- **Lower the model's reasoning effort:** this frees tokens for the answer, but it changes every answer, so the saved evaluation would no longer describe the system.
- **A higher limit, or none:** this would raise the worst-case cost and latency per call. 2,000 tokens covers a five-section summary with room for reasoning.

**Consequences:**
- **Cost:** the evaluator's worst-case reservation grows about fourfold, so its documented caps rose to $0.03 and $0.02. Actual spend per run is unchanged at about $0.002.
- **Latency:** a long answer can take 10–20 seconds, within the 30-second timeout per attempt.
- **Evaluation:** none of the 33 evaluation replies hit the 500-token limit, and a higher limit doesn't change a reply that finished within it. The results below should therefore still hold, but they were not re-run.

## Results

On 2026-10-11 both evaluation sets ran once through the full hosted pipeline, with the 500-token limit, under a budget guard ($0.012 cap for calibration, $0.008 for held-out). No call failed and no question was skipped.

| Metric | Calibration (20 questions) | Held-out (13 questions) |
| --- | --- | --- |
| Hit@1 (document and full section path) | 100% (15/15) | 100% (7/7) |
| MRR@5 | 1.0000 | 1.0000 |
| Expected section cited | 100% (15/15) | 100% (7/7) |
| Answer-contains (normalized phrase) | 86.67% (13/15) | 85.71% (6/7) |
| Unanswerable refusal rate | 100% (5/5) | 100% (6/6) |
| Answerable false-refusal rate | 0% (0/15) | 0% (0/7) |

- **Phrase misses are paraphrases.** All three (q06, q14, h12) are correct answers worded differently from the label. For example, q14's label is "to the Compliance Lead before taking part in the affected decision". The answer was "disclose the conflict of interest to the **Compliance Lead** before you participate in the supplier selection decision".
- **Declines held.** Every unanswerable question got the not-found reply, never a general-knowledge answer.
- **Retrieval improved.** Held-out hit@1 rose from 6/7 with the local model to 7/7.

**Caveat:** the held-out questions were seen during earlier rounds, so these figures are not a blind estimate. Estimating quality needs fresh questions.

This round cost $0.0023 on the key's usage record, covering a 6-question UI smoke test and both runs. All live testing so far cost $0.0085. Reports: [calibration](../../backend/evals/llm_results_hosted.json), [held-out](../../backend/evals/llm_results_heldout_hosted.json).
