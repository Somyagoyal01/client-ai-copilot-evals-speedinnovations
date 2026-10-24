import json

import pytest
from fastapi.testclient import TestClient

from copilot.agent import Answer, Copilot
from copilot.retrieval import HybridRetriever, tokenize
from copilot.tools import ERPTools, ToolError
from evals.run_eval import evaluate


@pytest.fixture(scope="module")
def bot():
    return Copilot()


def test_tokenize_keeps_numbers():
    assert "5,00,000" in tokenize("above INR 5,00,000")


def test_retrieval_finds_policy(bot):
    hits = bot.retriever.search("who approves invoices above 5,00,000", k=3)
    assert hits[0][0].doc.startswith("Accounts Payable")


def test_rag_answer_is_cited(bot):
    a = bot.ask("What CIBIL score is required for Personal Loan Plus?")
    assert a.route == "rag" and "700" in a.answer and a.citations[0].startswith("loan_product_note")


def test_tool_call_and_schema(bot):
    a = bot.ask("status of INV-1005?")
    assert a.route == "tool" and a.tool_calls[0]["name"] == "get_invoice"
    Answer.model_validate_json(a.model_dump_json())


def test_tool_validation_rejects_bad_args():
    t = ERPTools()
    with pytest.raises(ToolError):
        t.call("get_invoice", {"invoice_id": "1005; DROP TABLE"})
    with pytest.raises(ToolError):
        t.call("delete_invoice", {"invoice_id": "INV-1005"})


def test_unknown_record(bot):
    assert "No such record" in bot.ask("what about INV-9999").answer


def test_refuses_out_of_scope(bot):
    assert bot.ask("Who won the cricket world cup?").route == "refuse"


def test_account_numbers_are_redacted(bot):
    class Leaky:
        def plan(self, q, v): return None
        def compose(self, q, hits): return "Pay to account 123456789012 today", [hits[0][0].chunk_id]
        def usage(self, p, c): return 1, 1
    b = Copilot(llm=Leaky())
    assert "123456789012" not in b.ask("What are the standard vendor payment terms?").answer


def test_dev_eval_gate():
    r = evaluate("dev")
    assert r["routing_acc"] >= 0.9 and r["hallucinated_answers"] == 0


def test_api():
    from copilot.api import app
    c = TestClient(app)
    r = c.post("/ask", json={"question": "How often is bank reconciliation done?"})
    assert r.status_code == 200 and r.json()["route"] == "rag"
    assert c.get("/metrics").json()["requests"] >= 1
