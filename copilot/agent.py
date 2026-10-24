"""Copilot: routes a question to ERP tools (function calling) or to RAG, returns a validated JSON answer.

Offline mode uses a deterministic `ScriptedLLM` so everything runs without API keys.
Real-LLM mode: set OPENAI_API_KEY (or Azure OpenAI env vars) and use `OpenAIPlanner`; same tool schemas,
same output schema, same guardrails.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

from .retrieval import HybridRetriever, tokenize
from .tools import TOOL_SCHEMAS, ERPTools, ToolError

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
ACCT = re.compile(r"\b\d{9,18}\b")  # bank-account-like numbers


class Answer(BaseModel):
    answer: str
    route: str = Field(pattern="^(rag|tool|refuse)$")
    citations: List[str] = []
    tool_calls: List[dict] = []
    confidence: float = Field(ge=0, le=1)


class ScriptedLLM:
    """Stand-in for an LLM: picks a tool call by pattern, or extracts the best supporting sentence."""

    def plan(self, question: str, vendors):
        m = re.search(r"\bINV-?\s?(\d{4})\b", question, re.I)
        if m:
            return {"name": "get_invoice", "arguments": json.dumps({"invoice_id": f"INV-{m.group(1)}"})}
        for v in vendors:
            if v.lower() in question.lower() and re.search(r"term|pay|msme", question, re.I):
                return {"name": "get_vendor_terms", "arguments": json.dumps({"vendor": v})}
        return None

    def compose(self, question, hits):
        qt = set(tokenize(question))
        best = max(hits, key=lambda h: len(qt & set(tokenize(h[0].text))))
        return best[0].text, [best[0].chunk_id]

    def usage(self, prompt: str, completion: str):
        return len(prompt) // 4, len(completion) // 4  # rough token estimate


class OpenAIPlanner(ScriptedLLM):  # pragma: no cover - needs a key
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client, self.model = OpenAI(), model

    def plan(self, question, vendors):
        r = self.client.chat.completions.create(
            model=self.model, tools=TOOL_SCHEMAS, tool_choice="auto",
            messages=[{"role": "system", "content": "Call an ERP tool only for invoice or vendor-term lookups."},
                      {"role": "user", "content": question}])
        calls = r.choices[0].message.tool_calls
        return {"name": calls[0].function.name, "arguments": calls[0].function.arguments} if calls else None


class Copilot:
    def __init__(self, llm=None, min_dense: float = 0.08, min_bm25: float = 2.0):
        self.retriever = HybridRetriever.from_dir(DOCS_DIR)
        self.tools = ERPTools()
        self.llm = llm or (OpenAIPlanner() if os.getenv("COPILOT_USE_OPENAI") else ScriptedLLM())
        self.min_dense, self.min_bm25 = min_dense, min_bm25
        self.log = []  # per-request telemetry: latency, tokens, route

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        out = self._answer(question)
        out.answer = ACCT.sub("[REDACTED]", out.answer)  # guardrail: never echo account numbers
        Answer.model_validate(out.model_dump())  # schema check on every response
        pt, ct = self.llm.usage(question + str(out.citations), out.answer)
        self.log.append({"q": question, "route": out.route, "ms": (time.perf_counter() - t0) * 1000,
                         "prompt_tokens": pt, "completion_tokens": ct})
        return out

    def _answer(self, question: str) -> Answer:
        call = self.llm.plan(question, self.tools.vendors)
        if call:
            try:
                args = json.loads(call["arguments"])
                result = self.tools.call(call["name"], args)
            except (ToolError, json.JSONDecodeError) as e:
                return Answer(answer=f"I couldn't run that lookup ({e}).", route="refuse", confidence=0.0)
            if "error" in result:
                return Answer(answer="No such record in the ERP.", route="tool",
                              tool_calls=[{"name": call["name"], "args": args}], confidence=0.9)
            return Answer(answer=self._format(call["name"], result), route="tool",
                          tool_calls=[{"name": call["name"], "args": args}], confidence=0.95)
        hits = self.retriever.search(question, k=3)
        top_dense, top_bm = max(h[1] for h in hits), max(h[2] for h in hits)
        if top_dense < self.min_dense or top_bm < self.min_bm25:
            return Answer(answer="I don't have a source for that in the knowledge base.", route="refuse",
                          confidence=round(min(top_dense * 3, 1.0), 2))
        text, cites = self.llm.compose(question, hits)
        retrieved = {h[0].chunk_id for h in hits}
        cites = [c for c in cites if c in retrieved]  # guardrail: citations must come from retrieved chunks
        return Answer(answer=text, route="rag", citations=cites, confidence=round(min(top_dense * 2.5, 1.0), 2))

    @staticmethod
    def _format(name, r):
        if name == "get_invoice":
            return (f"{r['invoice_id']} from {r['vendor']} for INR {r['amount_inr']:,} is {r['status'].replace('_', ' ')}"
                    f" (PO {r['po_number']}).")
        msme = "MSME-registered" if r["msme"] else "not MSME-registered"
        return f"{r['vendor']} is on Net {r['payment_terms_days']} terms and is {msme}."


def to_json(ans: Answer) -> str:
    return ans.model_dump_json(indent=2)
