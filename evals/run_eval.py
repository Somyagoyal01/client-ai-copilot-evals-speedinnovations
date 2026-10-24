"""Evaluate the copilot: routing accuracy, retrieval hit@3, answer accuracy, citation validity, refusals,
latency and an ESTIMATED cost per 1,000 questions.

Usage: python -m evals.run_eval [dev|heldout]
The held-out set was hand-written with different wording (typos, informal, Hinglish-style shorthand)
and was NOT used to choose thresholds.
"""
import json
import statistics
import sys
from pathlib import Path

from copilot.agent import Copilot

HERE = Path(__file__).parent
# Assumption for the cost estimate only: a small hosted model at USD 0.15 / 1M input and 0.60 / 1M output tokens,
# ~600 prompt tokens of retrieved context per RAG call. Replace with your provider's real prices.
PRICE_IN, PRICE_OUT, CONTEXT_TOKENS = 0.15e-6, 0.60e-6, 600


def evaluate(name: str = "dev", bot: Copilot = None):
    items = json.loads((HERE / f"{name}_set.json").read_text(encoding="utf-8"))
    bot = bot or Copilot()
    rows = []
    for it in items:
        ans = bot.ask(it["q"])
        hits = bot.retriever.search(it["q"], k=3) if it["route"] == "rag" else []
        r = {"q": it["q"], "want": it["route"], "got": ans.route, "route_ok": ans.route == it["route"]}
        if it["route"] == "rag":
            r["hit3"] = any(h[0].chunk_id.split("#")[0] == it["doc"] for h in hits)
            r["correct"] = ans.route == "rag" and it["expect"].lower() in ans.answer.lower()
            r["cites_ok"] = ans.route != "rag" or (bool(ans.citations) and all(c.split("#")[0] == it["doc"] for c in ans.citations))
        elif it["route"] == "tool":
            r["correct"] = ans.route == "tool" and ans.tool_calls and ans.tool_calls[0]["name"] == it["tool"]
        else:
            r["correct"] = ans.route == "refuse"
        rows.append(r)

    def rate(key, subset):
        s = [x[key] for x in subset if key in x]
        return round(sum(s) / len(s), 3) if s else None

    rag = [r for r in rows if r["want"] == "rag"]
    ms = [x["ms"] for x in bot.log]
    tin = sum(x["prompt_tokens"] for x in bot.log) + CONTEXT_TOKENS * len(rag)
    tout = sum(x["completion_tokens"] for x in bot.log)
    report = {
        "set": name, "n": len(rows),
        "routing_acc": rate("route_ok", rows),
        "rag_hit@3": rate("hit3", rag),
        "rag_answer_acc": rate("correct", rag),
        "rag_citation_valid": rate("cites_ok", rag),
        "tool_acc": rate("correct", [r for r in rows if r["want"] == "tool"]),
        "refusal_acc": rate("correct", [r for r in rows if r["want"] == "refuse"]),
        "overall_acc": rate("correct", rows),
        "false_refusals": sum(1 for r in rows if r["want"] != "refuse" and r["got"] == "refuse"),
        "hallucinated_answers": sum(1 for r in rows if r["want"] == "refuse" and r["got"] != "refuse"),
        "latency_p50_ms": round(statistics.median(ms), 2), "latency_max_ms": round(max(ms), 2),
        "est_cost_usd_per_1k_questions": round((tin * PRICE_IN + tout * PRICE_OUT) / len(rows) * 1000, 4),
        "failures": [r for r in rows if not r["correct"]],
    }
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1] if len(sys.argv) > 1 else "dev"), indent=2))
