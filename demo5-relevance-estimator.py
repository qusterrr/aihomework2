import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma

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


# ─── ChromaDB vector store (in-memory, built at startup) ─────────────────────

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = Chroma.from_texts(
    texts=[json.dumps(c) for c in CREATURES],
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ─── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    query: str           # user question
    context: list[str]   # retrieved creature entries
    answer: str          # final LLM response
    grade: str           # relevance grade ('relevant' or 'irrelevant')


# ─── LLM ─────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


# ─── Nodes ────────────────────────────────────────────────────────────────────

def retrieve(state: State) -> dict:
    """Retrieve the most relevant creatures via ChromaDB similarity search."""
    docs = retriever.invoke(state["query"])

    #print([doc.page_content for doc in docs])  # debug print to see retrieved documents

    return {"context": [doc.page_content for doc in docs]}


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

    response = llm.invoke(messages)
    return {"answer": response.content}

def grade_relevance(state: State) -> dict:
    """Ask Gemini whether the retrieved documents answer the query."""
    context_block = "\n\n".join(state["context"])

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

    response = llm.invoke(messages)
    grade = response.content.strip().lower()
    return {"grade": grade}


# ─── Graph ────────────────────────────────────────────────────────────────────
#
#   START → retrieve → generate → END
#

builder = StateGraph(State)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("grade_relevance", grade_relevance)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "grade_relevance")

builder.add_edge("grade_relevance", END)

graph = builder.compile()


# ─── Demo queries ─────────────────────────────────────────────────────────────

queries = [
    #"How many creatures are in this database?",
    #"How does nuclear fusion work?",
    #"Which creature has the highest intelligence?"
    #"Is the Iron Basilisk dangerous to humans?"
    "Are Mongrels dangerous to Saltmaws?"

]

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"Query:  {query}")
    print("-" * 60)

    result = graph.invoke({"query": query})

    print(f"Retrieved context:\n{result['context']}\n")

    print(f"Relevance grade: {result['grade']}")
    

    print(f"Answer:\n{result['answer']}")
