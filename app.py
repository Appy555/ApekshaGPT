import time
import os
import uuid
import certifi

from pathlib import Path

import uvicorn

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
)

from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from agent import get_agent

from voice_service import transcribe_audio

from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations,
)

from tools import set_current_thread_id


# Load environment variables
load_dotenv()

# Configure SSL certificates
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# Create FastAPI application
app = FastAPI(title="ApekshaGPT")


# Get project directory
BASE_DIR = Path(__file__).resolve().parent


# Configure templates folder
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# Create required directories
(BASE_DIR / "uploads").mkdir(exist_ok=True)
(BASE_DIR / "data").mkdir(exist_ok=True)


# Initialize database
init_db()


# -----------------------------------------
# Homepage Route
# -----------------------------------------

@app.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# -----------------------------------------
# Get All Threads
# -----------------------------------------

@app.get("/threads")
async def get_threads():

    try:
        conversations = list_conversations()

        return {
            "threads": [
                {
                    "thread_id": conversation.thread_id,
                    "title": conversation.title,
                    "created_at": (
                        conversation.created_at.isoformat()
                        if conversation.created_at else None
                    ),
                    "updated_at": (
                        conversation.updated_at.isoformat()
                        if conversation.updated_at else None
                    )
                }
                for conversation in conversations
            ]
        }

    except Exception as e:

        print("Threads route error:", repr(e))

        return JSONResponse(
            {"error": "Could not load conversations"},
            status_code=500
        )


# -----------------------------------------
# Get Conversations
# -----------------------------------------

@app.get("/conversations")
async def conversations():

    items = list_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": (
                    item.created_at.isoformat()
                    if item.created_at else None
                ),
                "updated_at": (
                    item.updated_at.isoformat()
                    if item.updated_at else None
                )
            }
            for item in items
        ]
    }


# -----------------------------------------
# Voice Transcription Route
# -----------------------------------------

@app.post("/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):

    audio_path = None

    try:

        # Get upload directory
        upload_dir = BASE_DIR / "uploads"

        upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Generate unique audio file path
        audio_path = upload_dir / f"{uuid.uuid4()}.webm"

        # Read uploaded audio
        audio_bytes = await file.read()

        # Save audio to disk
        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_bytes)

        # File is now closed

        print("Audio path:", audio_path)
        print("Audio size:", audio_path.stat().st_size)

        # Convert audio into text
        text = transcribe_audio(str(audio_path))

        # Check whether speech was detected
        if not text.strip():

            return JSONResponse(
                {
                    "error": "No speech detected. Please try again."
                },
                status_code=400
            )

        print("Transcribed text:", text)

        return {
            "text": text
        }

    except Exception as e:

        print("Transcription error:", repr(e))

        return JSONResponse(
            {
                "error": "Audio transcription failed"
            },
            status_code=500
        )

    finally:

        # Delete temporary audio file
        if audio_path is not None and audio_path.exists():

            audio_path.unlink(missing_ok=True)

        await file.close()


# -----------------------------------------
# Chat Route
# -----------------------------------------

@app.post("/chat")
async def chat(request: Request):

    try:

        # Read incoming JSON data
        data = await request.json()

        message = str(
            data.get("message", "")
        ).strip()

        thread_id = str(
            data.get("thread_id") or uuid.uuid4()
        )

        # Validate message
        if not message:

            return JSONResponse(
                {"error": "Message cannot be empty"},
                status_code=400
            )

        # Create or update conversation
        create_or_update_conversation(
            thread_id,
            message
        )

        # Save user message
        save_chat_message(
            thread_id,
            "user",
            message
        )

        # Set current thread
        set_current_thread_id(thread_id)

        # Get model name
        model_name = os.getenv(
            "MODEL_NAME",
            "gemini-3.8-flash"
        )

        # Initialize agent
        agent = get_agent(model_name)

        # Start response timer
        start_time = time.perf_counter()

        # Execute agent
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=message)
                ]
            },
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        # Calculate actual response time
        elapsed = time.perf_counter() - start_time

        print(
            f"Agent response time: {elapsed:.2f} seconds"
        )

        # Extract AI response
        answer = ""

        for msg in reversed(result.get("messages", [])):

            if isinstance(msg, AIMessage) and msg.content:

                answer = msg.content
                break

        # Handle list-based response content
        if isinstance(answer, list):

            answer = "\n".join(
                block.get("text", "")
                for block in answer
                if isinstance(block, dict)
                and block.get("text")
            )

        # Handle empty response
        if not answer:

            answer = "The agent did not return a text response."

        # Save assistant response
        save_chat_message(
            thread_id,
            "assistant",
            answer
        )

        return {
            "response": answer,
            "thread_id": thread_id
        }

    except Exception as e:

        print("Chat endpoint error:", repr(e))

        return JSONResponse(
            {
                "error": "Chat request failed. Check the terminal logs."
            },
            status_code=500
        )


# -----------------------------------------
# Chat History Route
# -----------------------------------------

@app.get("/history/{thread_id}")
async def get_thread_history(thread_id: str):

    try:

        messages = get_chat_history(thread_id)

        return {
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ]
        }

    except Exception as e:

        print("History route error:", repr(e))

        return JSONResponse(
            {
                "error": "Could not load conversation history"
            },
            status_code=500
        )


# -----------------------------------------
# Run FastAPI Application
# -----------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
    )