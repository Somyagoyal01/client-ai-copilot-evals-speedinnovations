"""FastAPI service: POST /ask {"question": "..."} -> structured JSON answer. GET /metrics -> latency/route stats."""
import statistics

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import Answer, Copilot

app = FastAPI(title="Client AI Copilot (demo)")
bot = Copilot()


class Q(BaseModel):
    question: str


@app.post("/ask", response_model=Answer)
def ask(q: Q):
    return bot.ask(q.question)


@app.get("/metrics")
def metrics():
    ms = [r["ms"] for r in bot.log] or [0.0]
    routes = {}
    for r in bot.log:
        routes[r["route"]] = routes.get(r["route"], 0) + 1
    return {"requests": len(bot.log), "p50_ms": statistics.median(ms), "max_ms": max(ms), "routes": routes}
