import json
import os
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser

# ─── Fictionary Creature Catalog ─────────────────────────────────────────────

CREATURES = [
    {
        "name": "Gloomfang",
        "type": "Shadow Beast",
        "habitat": "Dark Forests",
        "size": "Large",
        "abilities": ["Shadow Step", "Fear Aura", "Night Vision"],
        "diet": "Carnivore",
        "danger_level": 8,
        "description": "A wolf-like creature made of living shadow. Its fur absorbs light, making it nearly invisible at night.",
    },
    {
        "name": "Crystalwing",
        "type": "Aerial Elemental",
        "habitat": "Mountain Peaks",
        "size": "Medium",
        "abilities": ["Ice Breath", "Crystal Shield", "Blizzard Call"],
        "diet": "Omnivore",
        "danger_level": 6,
        "description": "A bird with wings made of translucent ice crystals. It soars at high altitudes and can summon blizzards.",
    },
    {
        "name": "Murkwraith",
        "type": "Swamp Spirit",
        "habitat": "Marshes and Bogs",
        "size": "Small",
        "abilities": ["Poison Mist", "Bog Sink", "Mimic Voice"],
        "diet": "Soul Eater",
        "danger_level": 7,
        "description": "A translucent spirit that floats above swamp water, luring travelers with mimicked voices before dragging them under.",
    },
    {
        "name": "Emberclaw",
        "type": "Fire Drake",
        "habitat": "Volcanic Regions",
        "size": "Huge",
        "abilities": ["Magma Breath", "Heat Aura", "Armor Melt"],
        "diet": "Carnivore",
        "danger_level": 9,
        "description": "A small dragon variant with claws that glow like molten rock. It can melt metal armor on contact.",
    },
    {
        "name": "Thornback",
        "type": "Forest Armored",
        "habitat": "Ancient Woodlands",
        "size": "Large",
        "abilities": ["Thorn Volley", "Bark Armor", "Root Grasp"],
        "diet": "Herbivore",
        "danger_level": 4,
        "description": "A tortoise-like creature covered in living thorns. Despite being herbivorous, it aggressively defends its territory.",
    },
    {
        "name": "Voidwhisper",
        "type": "Psychic Specter",
        "habitat": "Abandoned Ruins",
        "size": "Incorporeal",
        "abilities": ["Mind Read", "Memory Steal", "Illusion Cast"],
        "diet": "Memory Eater",
        "danger_level": 8,
        "description": "An invisible entity that feeds on memories. Victims often wake with no recollection of their past.",
    },
    {
        "name": "Saltmaw",
        "type": "Sea Lurker",
        "habitat": "Coastal Waters",
        "size": "Gigantic",
        "abilities": ["Tidal Pull", "Brine Spit", "Echo Roar"],
        "diet": "Piscivore",
        "danger_level": 7,
        "description": "A massive eel-like creature with rows of bioluminescent teeth, known for capsizing fishing boats.",
    },
    {
        "name": "Duskmorel",
        "type": "Fungal Wanderer",
        "habitat": "Underground Caves",
        "size": "Medium",
        "abilities": ["Spore Cloud", "Mycelium Network", "Regenerate"],
        "diet": "Decomposer",
        "danger_level": 3,
        "description": "A walking mushroom colony that releases hallucinogenic spores when threatened. Mostly harmless unless cornered.",
    },
]


# ─── ChromaDB vector store (file-based, built once) ──────────────────────────

CHROMA_DIR = "./chroma_db_demo6"

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

if os.path.exists(CHROMA_DIR):
    print("Loading existing vector store from disk...")
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
else:
    print("Building vector store and persisting to disk...")
    vector_store = Chroma.from_texts(
        texts=[json.dumps(c) for c in CREATURES],
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print("Vector store ready. Waiting for quota cooldown...")
    time.sleep(5)  # The 'Magic Fix' for Free Tier 500 errors

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ─── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    query: str           # user question
    context: list[str]   # retrieved creature entries
    answer: str          # final LLM response
    grade: str           # relevance grade ('relevant' or 'irrelevant')
    retry_count: int     # number of retrieval attempts so far


# ─── LLM ─────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")

str_parser = StrOutputParser()
rewrite_chain = llm | str_parser
grade_chain = llm | str_parser
generate_chain = llm | str_parser

# ─── Nodes ────────────────────────────────────────────────────────────────────

def retrieve(state: State) -> dict:
    """Retrieve documents. Includes a sleep to prevent Free Tier 500 errors."""
    print(f"--- RETRIEVING for: {state['query']} ---")
    print(f"  [DEBUG] query type={type(state['query'])}, len={len(state['query'])}, repr={repr(state['query'])}")

    # 15-second pause prevents the Embedding API from crashing on back-to-back calls
    time.sleep(15)

    docs = retriever.invoke(state["query"])
    return {
        "context": [doc.page_content for doc in docs],
        "retry_count": state["retry_count"] + 1,
    }


def rewrite_query(state: State) -> dict:
    """Rephrase the query to improve retrieval results."""
    messages = [
        SystemMessage(content=(
            "You are a LLM query rewriter. Rephrase the query "
            "to improve RAG retrieval quality. Last retrieval result was irrelevant."            
            "{\n{'name': 'Duskmorel',\n 'type': 'Fungal Wanderer',\n 'habitat': 'Underground Caves',\n 'size': 'Medium',\n 'abilities': ['Spore Cloud', 'Mycelium Network', 'Regenerate'],\n 'diet': 'Decomposer',"
            " 'danger_level': 3,\n 'description': 'A walking mushroom colony that releases hallucinogenic spores when threatened. Mostly harmless unless cornered.'}\n}\n\n"
            
            " Return only the rewritten query."
        )),
        HumanMessage(
            content=f"Original query: {state['query']}"
        ),
    ]
    #response = llm.invoke(messages)
    time.sleep(2) 
    response = rewrite_chain.invoke(messages)

    print(f"Rewritten query: {response}")
    return {"query": str(response)}


def generate(state: State) -> dict:
    """Generate an answer grounded in the retrieved creature entries."""
    context_block = "\n\n---\n\n".join(state["context"])
    

    messages = [
        SystemMessage(content=(
            "You are a knowledgeable guide to a world of fictionary creatures. "
            "Answer the user's question using only the provided creature catalog entries. "
            "Be concise and informative."
        )),
        HumanMessage(content=(
            f"Createure catalog entries:\n\n{context_block}\n\n"
            f"Question: {state['query']}"
        )),
    ]

    print("--- STARTING GENERATION ---")
    time.sleep(2) 

    #response = llm.invoke(messages)
    response = generate_chain.invoke(messages)
    print(f"Generated answer: {response}")
    print(f"--- END OF GENERATION ---\n")
    return {"answer": response}

def grade_relevance(state: State) -> dict:
    """Ask Gemini whether the retrieved documents answer the query."""
    context_block = "\n\n".join(state["context"])

    print("--- STARTING GRADING RELEVANCE ---")

    messages = [
        SystemMessage(content=(
            "You are a relevance grader. Given a user query and a set of retrieved "
            "documents, respond with exactly one word: 'relevant' or 'irrelevant'. "
            "Do not explain. Do not add punctuation."
        )),
        HumanMessage(content=(
            f"Query: {state['query']}\n\n"
            f"Documents:\n{context_block}"
        )),
    ]

    response = grade_chain.invoke(messages)
   
    # Strict cleaning to ensure the router works
    grade = response.strip().lower()
    if "irrelevant" in grade:
        grade = "irrelevant"
    elif "relevant" in grade:
        grade = "relevant"
    
    print(f"Relevance grade: {grade}")
    print(f"--- END OF GRADING ---\n")
    return {"grade": grade}


def route_after_grade(state: State) -> str:
    """Route to generate (if relevant or out of retries) or rewrite_query."""
    if (state["grade"] == "relevant") or (state["retry_count"] >= 2):
        print(f"Routing to generate (grade: {state['grade']}, retries: {state['retry_count']})")
        return "generate"
    
    print(f"Routing to rewrite_query (grade: {state['grade']}, retries: {state['retry_count']})")
    return "rewrite_query"


# ─── Graph ────────────────────────────────────────────────────────────────────
#
#   START → retrieve → grade_relevance → generate → END
#                           ↑                 (if relevant or retries exhausted)
#                     rewrite_query ←── (if irrelevant and retries < 2)
#

builder = StateGraph(State)

builder.add_node("retrieve", retrieve)
builder.add_node("grade_relevance", grade_relevance)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_relevance")
builder.add_conditional_edges("grade_relevance", route_after_grade)
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("generate", END)

graph = builder.compile()


# ─── Demo queries ─────────────────────────────────────────────────────────────

queries = [
    #"How many creatures are in this database?",
    #"How does nuclear fusion work?",
    #"Which creature has the highest intelligence?"
    #"Is the Iron Basilisk dangerous to humans?"
    #"Are Mongrels dangerous to Saltmaws?"
    "Is there any creature which can turn opponent in liquid?"

]

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"Query:  {query}")
    print("-" * 60)

    result = graph.invoke({"query": query, "retry_count": 0})

    print(f"Retrieved context:\n{result['context']}\n")

    print(f"Relevance grade: {result['grade']}  (retries: {result['retry_count']})")

    print(f"Answer:\n{result['answer']}")
