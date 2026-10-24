"""ERP tools exposed to the LLM via function calling. Schemas follow the OpenAI 'tools' JSON format."""
import json
import re
from pathlib import Path

ERP_PATH = Path(__file__).resolve().parent.parent / "data" / "erp.json"

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_invoice",
        "description": "Look up one invoice in the ERP by its id (format INV-1234).",
        "parameters": {"type": "object", "properties": {"invoice_id": {"type": "string", "pattern": "^INV-[0-9]{4}$"}},
                       "required": ["invoice_id"]}}},
    {"type": "function", "function": {
        "name": "get_vendor_terms",
        "description": "Get payment terms and MSME status for a vendor by exact vendor name.",
        "parameters": {"type": "object", "properties": {"vendor": {"type": "string"}}, "required": ["vendor"]}}},
]


class ToolError(ValueError):
    pass


class ERPTools:
    def __init__(self, path: Path = ERP_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.invoices = {r["invoice_id"]: r for r in data["invoices"]}
        self.terms = data["vendor_terms"]

    @property
    def vendors(self):
        return list(self.terms)

    def validate(self, name: str, args: dict):
        """Reject malformed calls before they touch the 'ERP' (guardrail against bad LLM output)."""
        schema = next((t["function"] for t in TOOL_SCHEMAS if t["function"]["name"] == name), None)
        if schema is None:
            raise ToolError(f"unknown tool {name!r}")
        props = schema["parameters"]["properties"]
        for req in schema["parameters"]["required"]:
            if req not in args or not isinstance(args[req], str):
                raise ToolError(f"{name}: missing/invalid {req}")
        extra = set(args) - set(props)
        if extra:
            raise ToolError(f"{name}: unexpected args {sorted(extra)}")
        pat = props.get("invoice_id", {}).get("pattern")
        if pat and "invoice_id" in args and not re.match(pat, args["invoice_id"]):
            raise ToolError("get_invoice: bad invoice id format")

    def call(self, name: str, args: dict):
        self.validate(name, args)
        if name == "get_invoice":
            rec = self.invoices.get(args["invoice_id"])
            return rec or {"error": "not_found"}
        rec = self.terms.get(args["vendor"])
        return {"vendor": args["vendor"], **rec} if rec else {"error": "not_found"}
