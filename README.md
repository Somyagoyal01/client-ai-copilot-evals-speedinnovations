# Client AI Copilot with an Evaluation Harness (ERP + banking docs)

> Personal project by Somya Goyal, built while applying to Speed Innovations. Not affiliated with or endorsed by Speed Innovations; uses public/synthetic data.

## Why this matters for Speed Innovations
- **Problem:** a services team ships RAG assistants and AI agents into many different client products (their AI Developer JD lists banking, ERP and SaaS). Each new client needs proof that the assistant answers from the client's documents, calls the client's APIs safely, and says "I don't know" when it should.
- **Who feels it:** delivery engineers and the client's stakeholders who must sign off before go-live. The JD asks the team to "create evaluation sets and track accuracy, hallucination, latency and cost" and to "add guardrails".
- **What this shows:** one small, reusable pattern. Hybrid RAG with citations plus function-calling into ERP tools, behind a FastAPI endpoint, with an eval harness that reports routing, retrieval, answer accuracy, refusals, latency and an estimated cost for each client. It also includes a hand-written held-out test that exposes where the assistant breaks.
- **Expected impact (estimate):** if a client sign-off cycle needs ~1 day of manual question testing, a repeatable eval run with a fixed question set could cut that to minutes per release. This is an assumption, not a measurement.

Source: Speed Innovations AI Developer job description, speedinnovations.in/careers (posted 9 Oct 2026).

## Demo
```bash
pip install -r requirements.txt
python -m data.make_corpus          # synthetic docs + ERP records
python -m evals.run_eval dev        # in-distribution set
python -m evals.run_eval heldout    # hand-written held-out set
uvicorn copilot.api:app --reload    # POST /ask {"question": "status of INV-1005?"}
pytest -q
```
Example answer:
```json
{"answer": "Invoices between INR 50,001 and INR 5,00,000 need approval from the Finance Controller.",
 "route": "rag", "citations": ["ap_invoice_policy#2"], "tool_calls": [], "confidence": 1.0}
```

## How it works
| Part | File | What it does |
|---|---|---|
| Ingestion + chunking | `copilot/retrieval.py` | Splits markdown by section; each chunk keeps its doc title and heading |
| Hybrid retrieval | `copilot/retrieval.py` | BM25 + TF-IDF cosine, fused with Reciprocal Rank Fusion, then a light re-rank |
| Function calling | `copilot/tools.py` | OpenAI-format tool schemas (`get_invoice`, `get_vendor_terms`); arguments are validated before execution |
| Copilot | `copilot/agent.py` | Routes each question to a tool, to RAG or to a refusal; returns a pydantic-validated JSON answer |
| Guardrails | `copilot/agent.py` | Refuses below a retrieval threshold, keeps only citations that were retrieved, redacts account-number-like digits, rejects malformed tool calls |
| API | `copilot/api.py` | `POST /ask`, `GET /metrics` (latency, route counts) |
| Evals | `evals/run_eval.py` | Routing / hit@3 / answer accuracy / citation validity / refusals / latency / cost estimate |

**Offline mode:** `ScriptedLLM` stands in for the model (pattern-based tool planner and extractive answers), so everything runs without API keys. **Real-LLM mode:** set `COPILOT_USE_OPENAI=1` and `OPENAI_API_KEY`. `OpenAIPlanner` sends the same tool schemas through chat-completions function calling. Azure OpenAI works the same way through the OpenAI SDK's Azure client.

## Results (measured on this repo, offline mode)
| Metric | Dev set (18 Qs, written alongside the docs) | Held-out set (24 Qs, hand-written: informal, typos, shorthand) |
|---|---|---|
| Routing accuracy | 100% | 83.3% |
| Retrieval hit@3 | 100% | **100%** |
| RAG answer accuracy | 100% | **60%** |
| Citation valid | 100% | 93.3% |
| Tool-call accuracy | 100% | 100% |
| Refusal accuracy | 100% | **40%** (3 out-of-scope questions answered anyway) |
| Overall | 100% | **62.5%** |
| Latency p50 | 0.6 ms | 0.6 ms (offline, no model call) |
| Est. cost / 1k questions | ~$0.07 | ~$0.07 (assumed small hosted model prices; see `run_eval.py`) |

**What the gap says.** Retrieval is fine: the right document is always in the top 3. The failures are in the last two steps:
1. The extractive answer step picks the wrong sentence within the right document (e.g. "min salary" matched the age line).
2. Similar-sounding out-of-scope questions ("home loan interest rate", "GST rate on steel") pass a lexical threshold because they share words with in-scope docs.

The thresholds were chosen on the dev set only and were not re-tuned after seeing the held-out results.

## Limitations / next steps
- Replace the extractive composer with a real LLM that sees the top 3 chunks, then re-run the held-out set.
- Swap TF-IDF for real embeddings (FAISS/Chroma or Azure AI Search) and add a cross-encoder re-ranker.
- Refusals: add an "answerability" check (does the retrieved chunk mention the asked entity, e.g. *home* loan vs *personal* loan) or an LLM judge.
- Track evals over time in CI (the workflow already runs the held-out eval) and log real token usage instead of the length/4 estimate.
- The ERP is a JSON file; a real deployment would call the client's API with auth and rate limits.
