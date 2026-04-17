"""
Demo 8 – Dynamic Procurement Agent (LangGraph + Interrupt + Tool Calls + Live API)

Tasks implemented:
1) Dynamic quantity extraction from request text + LLM tool calls for vendor pricing
2) Conditional approval only if best quote total > €10,000
3) Proper rejection flow that skips PO creation
4) Live pricing data from dummyjson.com/products/category/laptops

Run:
  python demo8.1-purchase-agent.py
  python demo8.1-purchase-agent.py --resume
  python demo8.1-purchase-agent.py --resume "Rejected — over budget"
"""

import sys
import os
import re
import json
import time
import sqlite3
import logging
from typing import TypedDict, Optional, Any

import requests
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool


# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── State ────────────────────────────────────────────────────────────────────

class ProcurementState(TypedDict, total=False):
    request: str
    quantity: int
    item_description: str
    team_name: str

    vendors: list[dict]
    quotes: list[dict]
    best_quote: dict

    approval_status: str
    rejection_reason: str

    po_number: str
    notification: str


# ─── LLMs ─────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


# ─── Helpers ──────────────────────────────────────────────────────────────────

DEFAULT_FALLBACK_PRICES = {
    "Dell": 899.0,
    "Lenovo": 849.0,
    "HP": 879.0,
}

APPROVAL_THRESHOLD = 10_000
DUMMYJSON_LAPTOPS_URL = "https://dummyjson.com/products/category/laptops"


def parse_quantity_and_context(request_text: str) -> dict:
    """
    Extract quantity and a bit of context from a free-text request.
    Example:
      "Order 30 laptops for the sales team"
    """
    quantity_match = re.search(r"\b(\d+)\b", request_text)
    quantity = int(quantity_match.group(1)) if quantity_match else 1

    item_description = "laptops"
    item_match = re.search(r"\b\d+\s+([a-zA-Z\- ]+?)\s+for\b", request_text, re.IGNORECASE)
    if item_match:
        item_description = item_match.group(1).strip()

    team_name = "requesting team"
    team_match = re.search(r"\bfor the ([a-zA-Z\- ]+?)(?: team)?\b", request_text, re.IGNORECASE)
    if team_match:
        team_name = team_match.group(1).strip()

    return {
        "quantity": quantity,
        "item_description": item_description,
        "team_name": team_name,
    }


def parse_shipping_days(shipping_info: str) -> Optional[int]:
    """
    Convert textual shipping info into approximate day count.
    Accept only <= 14 days as 'within 2 weeks'.
    """
    if not shipping_info:
        return None

    text = shipping_info.strip().lower()

    if "overnight" in text:
        return 1
    if "same day" in text:
        return 0

    day_match = re.search(r"(\d+)\s+day", text)
    if day_match:
        return int(day_match.group(1))

    week_match = re.search(r"(\d+)\s+week", text)
    if week_match:
        return int(week_match.group(1)) * 7

    month_match = re.search(r"(\d+)\s+month", text)
    if month_match:
        return int(month_match.group(1)) * 30

    return None


def fetch_laptop_catalog() -> list[dict]:
    """Fetch laptop products from DummyJSON."""
    try:
        response = requests.get(DUMMYJSON_LAPTOPS_URL, timeout=15)
        response.raise_for_status()
        payload = response.json()
        return payload.get("products", [])
    except Exception as exc:
        logger.warning("Failed to fetch laptop catalog from DummyJSON: %s", exc)
        return []


def choose_product_for_vendor(vendor: str) -> dict:
    """
    Pick the cheapest suitable laptop for a vendor:
    1) Prefer products matching vendor brand/title and available within 2 weeks
    2) If none, fallback to any cheapest laptop within 2 weeks
    3) If still none, fallback to hardcoded default
    """
    products = fetch_laptop_catalog()
    vendor_lower = vendor.lower()

    def normalize_product(product: dict) -> dict:
        shipping_days = parse_shipping_days(product.get("shippingInformation", ""))
        return {
            "id": product.get("id"),
            "title": product.get("title", "Unknown laptop"),
            "brand": product.get("brand", ""),
            "price": float(product.get("price", DEFAULT_FALLBACK_PRICES.get(vendor, 899.0))),
            "stock": product.get("stock", 0),
            "shipping_days": shipping_days if shipping_days is not None else 999,
            "availability_status": product.get("availabilityStatus", "Unknown"),
            "raw": product,
        }

    normalized = [normalize_product(p) for p in products]

    matching_in_2_weeks = [
        p for p in normalized
        if p["stock"] > 0
        and p["shipping_days"] <= 14
        and (
            vendor_lower in p["brand"].lower()
            or vendor_lower in p["title"].lower()
        )
    ]

    if matching_in_2_weeks:
        return min(matching_in_2_weeks, key=lambda p: p["price"])

    fallback_in_2_weeks = [
        p for p in normalized
        if p["stock"] > 0 and p["shipping_days"] <= 14
    ]
    if fallback_in_2_weeks:
        chosen = min(fallback_in_2_weeks, key=lambda p: p["price"])
        logger.warning(
            "No vendor-specific product found for %s within 2 weeks. Falling back to cheapest available laptop: %s",
            vendor,
            chosen["title"],
        )
        return chosen

    logger.warning(
        "No suitable live laptop found for %s. Falling back to default hardcoded price.",
        vendor,
    )
    return {
        "id": None,
        "title": f"Fallback laptop for {vendor}",
        "brand": vendor,
        "price": DEFAULT_FALLBACK_PRICES.get(vendor, 899.0),
        "stock": 0,
        "shipping_days": 14,
        "availability_status": "Fallback",
        "raw": {},
    }


# ─── Tool ─────────────────────────────────────────────────────────────────────

@tool
def get_unit_price(vendor: str) -> str:
    """
    Get the current laptop unit price for a vendor.
    Returns JSON string with unit_price and product details.
    """
    product = choose_product_for_vendor(vendor)
    payload = {
        "vendor": vendor,
        "unit_price": float(product["price"]),
        "product_title": product["title"],
        "product_brand": product["brand"],
        "delivery_days": int(product["shipping_days"]),
        "stock": int(product["stock"]),
        "availability_status": product["availability_status"],
        "product_id": product["id"],
    }
    return json.dumps(payload)


pricing_llm = llm.bind_tools([get_unit_price])


# ─── Node functions ──────────────────────────────────────────────────────────

def lookup_vendors(state: ProcurementState) -> dict:
    """Step 1: Parse request and look up approved vendors for laptops."""
    print("\n[Step 1] Looking up approved vendors and parsing request...")
    time.sleep(1)

    parsed = parse_quantity_and_context(state["request"])

    vendors = [
        {"name": "Dell", "id": "V-001", "category": "laptops", "rating": 4.5},
        {"name": "Lenovo", "id": "V-002", "category": "laptops", "rating": 4.3},
        {"name": "HP", "id": "V-003", "category": "laptops", "rating": 4.1},
    ]

    print(f"   Parsed quantity: {parsed['quantity']}")
    print(f"   Item: {parsed['item_description']}")
    print(f"   Team: {parsed['team_name']}")

    for v in vendors:
        print(f"   Found vendor: {v['name']} (rating {v['rating']})")

    return {
        "quantity": parsed["quantity"],
        "item_description": parsed["item_description"],
        "team_name": parsed["team_name"],
        "vendors": vendors,
    }


def fetch_pricing(state: ProcurementState) -> dict:
    """
    Step 2: Use LLM tool calls to get a live unit price for each vendor.
    The LLM must call get_unit_price once per vendor.
    """
    print("\n[Step 2] Fetching pricing from suppliers via LLM tool calls...")
    time.sleep(1)

    quantity = state["quantity"]
    vendor_names = [v["name"] for v in state["vendors"]]

    prompt = (
        "You are helping with procurement pricing.\n"
        f"The user request is: {state['request']}\n"
        f"Quantity requested: {quantity}\n"
        f"Vendors: {', '.join(vendor_names)}\n\n"
        "Call the get_unit_price tool exactly once for each vendor listed above.\n"
        "Do not skip any vendor.\n"
        "Return only tool calls."
    )

    ai_msg = pricing_llm.invoke(prompt)

    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    if not tool_calls:
        raise RuntimeError("LLM did not produce any tool calls for pricing.")

    seen_vendors = set()
    quotes = []

    for call in tool_calls:
        call_name = call.get("name")
        call_args = call.get("args", {})

        if call_name != "get_unit_price":
            continue

        vendor = call_args.get("vendor")
        if not vendor or vendor in seen_vendors:
            continue

        seen_vendors.add(vendor)
        tool_result = get_unit_price.invoke({"vendor": vendor})

        try:
            parsed = json.loads(tool_result)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Tool returned invalid JSON for vendor {vendor}: {tool_result}") from exc

        unit_price = float(parsed["unit_price"])
        delivery_days = int(parsed["delivery_days"])
        total = round(unit_price * quantity, 2)

        quote = {
            "vendor": vendor,
            "unit_price": unit_price,
            "total": total,
            "delivery_days": delivery_days,
            "product_title": parsed["product_title"],
            "product_brand": parsed.get("product_brand", vendor),
            "availability_status": parsed.get("availability_status", "Unknown"),
            "stock": parsed.get("stock", 0),
            "product_id": parsed.get("product_id"),
        }
        quotes.append(quote)

    missing_vendors = [v for v in vendor_names if v not in seen_vendors]
    if missing_vendors:
        raise RuntimeError(
            f"LLM failed to call pricing tool for all vendors. Missing: {missing_vendors}"
        )

    for q in quotes:
        print(
            f"   {q['vendor']}: {q['product_title']} | "
            f"€{q['unit_price']}/unit x {quantity} = €{q['total']:,} "
            f"({q['delivery_days']} day delivery)"
        )

    return {"quotes": quotes}


def compare_quotes(state: ProcurementState) -> dict:
    """Step 3: Compare quotes and pick the best one."""
    print("\n[Step 3] Comparing quotes...")
    time.sleep(0.5)

    best = min(state["quotes"], key=lambda q: q["total"])
    max_total = max(q["total"] for q in state["quotes"])
    savings = round(max_total - best["total"], 2)

    print(f"   Best quote: {best['vendor']} at €{best['total']:,}")
    print(f"   Product: {best['product_title']}")
    print(f"   Saves €{savings:,} vs most expensive option")

    return {"best_quote": best}


def route_after_compare(state: ProcurementState) -> str:
    """
    Task 2:
    If best quote total > 10,000 => request approval
    Else => skip approval and go straight to PO creation
    """
    best_total = state["best_quote"]["total"]
    if best_total > APPROVAL_THRESHOLD:
        return "request_approval"
    return "submit_purchase_order"


def request_approval(state: ProcurementState) -> dict:
    """Step 4: Human-in-the-loop — request manager approval only when needed."""
    best = state["best_quote"]
    quantity = state["quantity"]
    team_name = state["team_name"]

    print("\n[Step 4] Order exceeds €10,000 — manager approval required!")
    print("   Sending approval request to manager...")

    amount_str = f"€{best['total']:,}"
    delivery_str = f"{best['delivery_days']} business days"
    items_str = f"{quantity} laptops for {team_name} team"
    product_str = best["product_title"][:33]

    print("   ┌─────────────────────────────────────────────┐")
    print("   │  APPROVAL NEEDED                            │")
    print(f"   │  Vendor:   {best['vendor']:<33}│")
    print(f"   │  Product:  {product_str:<33}│")
    print(f"   │  Amount:   {amount_str:<33}│")
    print(f"   │  Items:    {items_str[:33]:<33}│")
    print(f"   │  Delivery: {delivery_str:<33}│")
    print("   └─────────────────────────────────────────────┘")

    decision = interrupt({
        "message": (
            f"Approve purchase of {quantity} laptops "
            f"({best['product_title']}) from {best['vendor']} "
            f"for €{best['total']:,}?"
        ),
        "vendor": best["vendor"],
        "product": best["product_title"],
        "amount": best["total"],
        "quantity": quantity,
    })

    print(f"\n[Step 4] Manager responded: {decision}")
    return {"approval_status": str(decision)}


def route_after_approval(state: ProcurementState) -> str:
    """
    Task 3:
    If approved => continue
    If rejected => skip PO creation and notify employee directly
    """
    approval_text = (state.get("approval_status") or "").strip().lower()
    if "reject" in approval_text:
        return "notify_employee"
    return "submit_purchase_order"


def submit_purchase_order(state: ProcurementState) -> dict:
    """Step 5: Submit the purchase order to the ERP system."""
    print("\n[Step 5] Submitting purchase order to ERP system...")
    time.sleep(1)

    po_number = "PO-2026-00342"
    print(f"   Purchase order created: {po_number}")
    print(f"   Vendor: {state['best_quote']['vendor']}")
    print(f"   Product: {state['best_quote']['product_title']}")
    print(f"   Quantity: {state['quantity']}")
    print(f"   Amount: €{state['best_quote']['total']:,}")

    return {"po_number": po_number}


def notify_employee(state: ProcurementState) -> dict:
    """Step 6: Use LLM to draft and send a notification to the employee."""
    print("\n[Step 6] Notifying employee...")

    quantity = state.get("quantity", 0)
    best = state.get("best_quote", {})
    team_name = state.get("team_name", "requesting team")

    approval_text = state.get("approval_status", "")
    rejected = "reject" in approval_text.lower()

    if rejected:
        rejection_reason = state.get("rejection_reason") or approval_text or "Rejected by manager"
        prompt = (
            f"Write a brief, professional notification (2-3 sentences) to an employee "
            f"that their purchase request for {quantity} laptops for the {team_name} team "
            f"was rejected by the manager. "
            f"Include this reason clearly but politely: {rejection_reason}. "
            f"Be empathetic, concise, and professional."
        )
    else:
        prompt = (
            f"Write a brief, professional notification (2-3 sentences) to an employee "
            f"that their purchase request has been approved and processed. "
            f"Details: {quantity} laptops, vendor {best.get('vendor', 'N/A')}, "
            f"product {best.get('product_title', 'N/A')}, "
            f"€{best.get('total', 0):,}, PO number {state.get('po_number', 'N/A')}, "
            f"delivery in {best.get('delivery_days', 'N/A')} business days."
        )

    response = llm.invoke(prompt)
    notification = response.content

    print("   Employee notification sent:")
    print(f'   "{notification}"')

    return {"notification": notification}


# ─── Build the graph ─────────────────────────────────────────────────────────

builder = StateGraph(ProcurementState)

builder.add_node("lookup_vendors", lookup_vendors)
builder.add_node("fetch_pricing", fetch_pricing)
builder.add_node("compare_quotes", compare_quotes)
builder.add_node("request_approval", request_approval)
builder.add_node("submit_purchase_order", submit_purchase_order)
builder.add_node("notify_employee", notify_employee)

builder.add_edge(START, "lookup_vendors")
builder.add_edge("lookup_vendors", "fetch_pricing")
builder.add_edge("fetch_pricing", "compare_quotes")

# Task 2: conditional approval
builder.add_conditional_edges(
    "compare_quotes",
    route_after_compare,
    {
        "request_approval": "request_approval",
        "submit_purchase_order": "submit_purchase_order",
    },
)

# Task 3: conditional route after approval
builder.add_conditional_edges(
    "request_approval",
    route_after_approval,
    {
        "submit_purchase_order": "submit_purchase_order",
        "notify_employee": "notify_employee",
    },
)

builder.add_edge("submit_purchase_order", "notify_employee")
builder.add_edge("notify_employee", END)


# ─── Checkpointer ────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "procurement_checkpoints.db")
THREAD_ID = "procurement-thread-1"
config = {"configurable": {"thread_id": THREAD_ID}}


# ─── Main ────────────────────────────────────────────────────────────────────

def run_first_invocation(graph):
    """First run: employee submits request, agent does steps until interrupt or completion."""
    print("=" * 60)
    print("  FIRST INVOCATION — Employee submits purchase request")
    print("=" * 60)

    request_text = "Order 30 laptops for the sales team"
    print(f'\nEmployee request: "{request_text}"')

    result = graph.invoke(
        {"request": request_text},
        config,
    )

    if isinstance(result, dict) and "__interrupt__" in result:
        print("\n" + "=" * 60)
        print("AGENT SUSPENDED — waiting for manager approval")
        print("=" * 60)
        print("\n  The agent process can now exit completely.")
        print("  All state is frozen in SQLite.")
        print(f"  Checkpoint DB: {DB_PATH}")
        print(f"  Thread ID: {THREAD_ID}")
        print("\n  To resume with approval:")
        print(f"    python {os.path.basename(__file__)} --resume")
        print("\n  To resume with rejection:")
        print(f'    python {os.path.basename(__file__)} --resume "Rejected — over budget"')
    else:
        print("\n" + "=" * 60)
        print("PROCUREMENT COMPLETE (approval not required)")
        print("=" * 60)
        print(f"\n  PO Number:    {result.get('po_number', 'N/A')}")
        print(f"  Vendor:       {result.get('best_quote', {}).get('vendor', 'N/A')}")
        print(f"  Product:      {result.get('best_quote', {}).get('product_title', 'N/A')}")
        print(f"  Total:        €{result.get('best_quote', {}).get('total', 0):,}")
        print()


def run_second_invocation(graph, resume_value: str):
    """Resume after manager decision."""
    print("=" * 60)
    print("  SECOND INVOCATION — Manager decision")
    print("=" * 60)

    saved_state = graph.get_state(config)
    if not saved_state or not saved_state.values:
        print("\nNo saved state found! Run without --resume first.")
        return

    print("\nLoading state from checkpoint...")
    print(f"  ✓ Request: {saved_state.values.get('request', 'N/A')}")
    print(f"  ✓ Quantity: {saved_state.values.get('quantity', 'N/A')}")
    print(f"  ✓ Vendors found: {len(saved_state.values.get('vendors', []))}")
    print(f"  ✓ Quotes received: {len(saved_state.values.get('quotes', []))}")

    best = saved_state.values.get("best_quote", {})
    print(f"  ✓ Best quote: {best.get('vendor', 'N/A')} at €{best.get('total', 0):,}")
    print(f"  ✓ Product: {best.get('product_title', 'N/A')}")
    print("\n  Steps before the interrupt are NOT re-executed.\n")

    print(f'Manager decision: "{resume_value}"')
    time.sleep(1)

    update_payload = {}
    if "reject" in resume_value.lower():
        update_payload["rejection_reason"] = resume_value

    result = graph.invoke(
        Command(resume=resume_value, update=update_payload),
        config,
    )

    print("\n" + "=" * 60)
    print("PROCUREMENT FINISHED")
    print("=" * 60)
    print(f"\n  PO Number:    {result.get('po_number', 'N/A')}")
    print(f"  Vendor:       {result.get('best_quote', {}).get('vendor', 'N/A')}")
    print(f"  Product:      {result.get('best_quote', {}).get('product_title', 'N/A')}")
    print(f"  Total:        €{result.get('best_quote', {}).get('total', 0):,}")
    print(f"  Approval:     {result.get('approval_status', 'N/A')}")
    print(f"  Rejection:    {result.get('rejection_reason', 'N/A')}")
    print()


if __name__ == "__main__":
    resume_mode = "--resume" in sys.argv

    # Optional manager decision from CLI:
    # python demo8.1-purchase-agent.py --resume "Rejected — over budget"
    resume_value = "Approved — go ahead with the purchase."
    if resume_mode:
        extra_args = [arg for arg in sys.argv[1:] if arg != "--resume"]
        if extra_args:
            resume_value = " ".join(extra_args)

    # Clean start if not resuming
    if not resume_mode and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("(Cleaned up old checkpoint DB)")

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpointer)

    try:
        if resume_mode:
            run_second_invocation(graph, resume_value)
        else:
            run_first_invocation(graph)
    finally:
        conn.close()