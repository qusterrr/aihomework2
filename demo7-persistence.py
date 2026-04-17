"""
Demo 7 – LangGraph Persistence (Memory across invocations)

A minimal "hello world" example showing how MemorySaver lets an LLM
remember earlier messages within the same thread.

Flow:
  1. User says: "Hello, my name is Bumblebee Jack"
  2. LLM responds with a greeting
  3. User says: "Tell a joke based on my name"
  4. LLM remembers the name and tells a relevant joke
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# ─── State ────────────────────────────────────────────────────────────────────
# We use the built-in `messages` key so LangGraph automatically accumulates
# the conversation history across invocations.

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


# ─── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ─── Single node: call the LLM ───────────────────────────────────────────────

def chat(state: State) -> dict:
    """Send the full conversation history to the LLM and append its reply."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ─── Graph ────────────────────────────────────────────────────────────────────
#   START → chat → END

builder = StateGraph(State)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# ─── Checkpointer (in-memory persistence) ────────────────────────────────────

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ─── Thread config ────────────────────────────────────────────────────────────
# All invocations sharing the same thread_id share conversation history.

config = {"configurable": {"thread_id": "demo-session-1"}}

# ─── First message ────────────────────────────────────────────────────────────

print("=" * 60)
print("USER: Hello, my name is Bumblebee Jack")
print("=" * 60)

result = graph.invoke(
    {"messages": [HumanMessage(content="Hello, my name is Bumblebee Jack")]},
    config,
)

print(f"\nASSISTANT: {result['messages'][-1].content}\n")

# ─── Second message (same thread – the LLM remembers the name) ───────────────

print("=" * 60)
print("USER: Tell a joke based on my name")
print("=" * 60)

result = graph.invoke(
    {"messages": [HumanMessage(content="Tell a joke based on my name")]},
    config,
)

print(f"\nASSISTANT: {result['messages'][-1].content}\n")

# ─── Show conversation history ────────────────────────────────────────────────

print("=" * 60)
print("FULL CONVERSATION HISTORY")
print("=" * 60)

state = graph.get_state(config)
for msg in state.values["messages"]:
    role = "USER" if isinstance(msg, HumanMessage) else "ASSISTANT"
    print(f"\n{role}: {msg.content}")