"""
Demo 8 – LangGraph Interrupt (Human-in-the-loop)

Minimal example showing how interrupt() pauses graph execution between nodes,
allowing external code (or a human) to inspect state before resuming.

Flow:
  1. add_hello node  → appends "Hello" to message, then calls interrupt()
     ↳ Graph pauses here; caller inspects state
  2. Caller resumes the graph with Command(resume=...)
  3. add_world node  → appends "World" to message
  4. END
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

# ─── State ────────────────────────────────────────────────────────────────────


class State(TypedDict):
    message: str


# ─── Nodes ────────────────────────────────────────────────────────────────────


def add_hello(state: State) -> dict:
    """Append 'Hello' then pause so the caller can inspect / approve."""
    updated = state["message"] + "Hello"
    print(f"[add_hello] message so far: '{updated}'")

    # interrupt() suspends execution here.  The value passed is available to
    # the caller via the snapshot's interrupt values.  Graph resumes from this
    # point when invoke() is called again with Command(resume=...) and the same
    # thread config.
    command_resume_result = interrupt("Paused after 'Hello' - call invoke(Command(resume=...), config) to continue.")

    # This is here to show that execution truly resumes from the interrupt point, and that
    # the value passed to interrupt() is available after resuming.  You could imagine using this
    # value to pass info from the human approver back into the graph.
    print(f"[add_hello] Resuming with command: {command_resume_result}")

    # This line runs only after the graph is resumed.
    return {"message": updated}


def add_world(state: State) -> dict:
    """Append ' World' to complete the greeting."""
    updated = state["message"] + " World"
    print(f"[add_world] message so far: '{updated}'")
    return {"message": updated}


# ─── Graph ────────────────────────────────────────────────────────────────────
#   START → add_hello → add_world → END

builder = StateGraph(State)
builder.add_node("add_hello", add_hello)
builder.add_node("add_world", add_world)
builder.add_edge(START, "add_hello")
builder.add_edge("add_hello", "add_world")
builder.add_edge("add_world", END)

# MemorySaver is required for interrupt to persist state between invocations.
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ─── Thread config ────────────────────────────────────────────────────────────

config = {"configurable": {"thread_id": "hello-world-thread"}}

# ─── First invocation: runs until interrupt ───────────────────────────────────

print("=" * 50)
print("STEP 1: Starting graph – will pause after add_hello")
print("=" * 50)

# invoke() returns the state at the point of interruption (not the node's
# pending return value), so message will be '' here – that is expected.
result = graph.invoke({"message": ""}, config)
print(f"Graph paused. State returned by invoke: message='{result['message']}'")

# Inspect persisted state at the interruption point.
snapshot = graph.get_state(config)
print(f"\nState at pause: message='{snapshot.values['message']}'")
print(f"Next node(s):   {snapshot.next}")

# ─── Second invocation: resume from the interruption point ───────────────────

print("\n" + "=" * 50)
print("STEP 2: Resuming graph – will run add_world and finish")
print("=" * 50)

# Command(resume=...) signals LangGraph to resume from the interrupt() call
# inside add_hello, rather than restarting the node from scratch.
# Passing None as input (graph.invoke(None, config)) is WRONG – it would
# restart add_hello from the checkpoint and hit interrupt() again.
result = graph.invoke(Command(resume="foobar"), config)
print(f"\nFinal message: '{result['message']}'")
