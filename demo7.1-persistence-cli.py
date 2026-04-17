"""
Demo 7.1 – LangGraph Persistence with SQLite Backend

An interactive CLI chatbot that persists conversation state to a local SQLite
database. When the program is restarted, the conversation picks up exactly
where it left off — the LLM remembers everything you previously discussed.

Usage:
  python demo7.1-persistence-agent.py

  Type your messages at the "Message to LLM:" prompt.
  Type "quit" to exit. Restart the program to continue the conversation.

This demonstrates:
- SQLite-backed persistence across program restarts
- Interactive CLI chat loop
- State management with message history
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage


# ─── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ─── LLM ──────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ─── Nodes ────────────────────────────────────────────────────────────────────

def chat(state: State) -> dict:
    """Process the user message and generate a response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ─── Graph ────────────────────────────────────────────────────────────────────

builder = StateGraph(State)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)


# ─── SQLite persistence ──────────────────────────────────────────────────────

DB_PATH = "demo7.1-chat_history.db"
THREAD_ID = "main-session"


# ─── Interactive CLI ──────────────────────────────────────────────────────────

def main():
    config = {"configurable": {"thread_id": THREAD_ID}}

    # with is a convenient way to ensure the database connection is properly closed when done
    # see for example https://www.geeksforgeeks.org/python/with-statement-in-python/
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        # Show previous conversation if resuming
        saved_state = graph.get_state(config)
        if saved_state.values and saved_state.values.get("messages"):
            print("Resuming previous conversation:\n")
            for msg in saved_state.values["messages"]:
                role = "You" if isinstance(msg, HumanMessage) else "LLM"
                print(f"  {role}: {msg.content}")
            print()

        print('Type "quit" to exit.\n')

        # Start interactive chat loop in the command line
        while True:
            try:
                user_input = input("Message to LLM: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            
            # execute the graph with the new user message, which will be appended to
            # the conversation history in the state
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config,
            )

            print(f"LLM: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
