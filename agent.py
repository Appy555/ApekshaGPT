
import os
import time
import sqlite3
from pathlib import Path

import certifi
from dotenv import load_dotenv

# Load .env from the project directory
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Configure SSL certificates
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import tools


# Ensure data directory exists
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default model
DEFAULT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
).strip()

# Models this application permits
ALLOWED_MODELS = {
    "gemini-3.8-flash",
}


SYSTEM_PROMPT = """
You are ApekshaGPT, a helpful AI assistant.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- Always understand and answer the user's actual question.
- Do not give generic greetings instead of answering a question.
- For latest news, current events, recent updates, or current prices, use web search.
- For questions about uploaded documents, use search_uploaded_documents.
- When asked to remember something, use remember_this.
- For previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- Summarize web search results clearly.
- Be clear, helpful, and concise.
"""


def normalize_model_name(model_name: str | None = None) -> str:
    """Choose and validate the model name."""

    selected_model = (model_name or DEFAULT_MODEL).strip()

    if selected_model not in ALLOWED_MODELS:
        raise ValueError(
            f"Unsupported Gemini model: {selected_model}. "
            f"Allowed models: {', '.join(sorted(ALLOWED_MODELS))}"
        )

    return selected_model


def build_agent(model_name: str | None = None):
    """Build a LangGraph agent for the selected Gemini model."""

    selected_model = normalize_model_name(model_name)

    print(f"Building agent with Gemini model: {selected_model}")

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.3,
        streaming=True,
    )

    # Bind tools to Gemini
    llm_with_tools = llm.bind_tools(tools)

    # -----------------------------------------
    # Chatbot Node
    # -----------------------------------------

    def chatbot_node(state: MessagesState):

        # Add system prompt to conversation messages
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]

        # Debug: Check messages reaching Gemini
        print("\n========== INPUT TO GEMINI ==========")

        for msg in messages:
            print("Role:", msg.type)
            print("Content:", msg.content)

        # Get response from Gemini
        start_time = time.perf_counter()
        response = llm_with_tools.invoke(messages)
        elapsed = time.perf_counter() - start_time
        print(f"Gemini response time: {elapsed:.2f} seconds")
                

        # Debug: Check Gemini response
        print("\n========== GEMINI RESPONSE ==========")
        print(response.content)
        print("====================================\n")

        return {"messages": [response]}

    # -----------------------------------------
    # Build LangGraph Workflow
    # -----------------------------------------

    workflow = StateGraph(MessagesState)

    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "chatbot")

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition
    )

    workflow.add_edge("tools", "chatbot")

    # -----------------------------------------
    # SQLite Checkpointer
    # -----------------------------------------

    db_path = DATA_DIR / "checkpoints.sqlite"

    connection = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(connection)

    graph = workflow.compile(
        checkpointer=checkpointer
    )

    return graph


# -----------------------------------------
# Agent Cache
# -----------------------------------------

_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """Return a cached agent for the selected model."""

    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:

        _AGENT_CACHE[selected_model] = build_agent(
            selected_model
        )

    return _AGENT_CACHE[selected_model]
