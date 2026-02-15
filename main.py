from fastapi import FastAPI,Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_community.chat_message_histories import SQLChatMessageHistory
import message,os
from dotenv import load_dotenv



class UserPrompt(BaseModel):
    Username: str
    prompt: str
store = {}
def get_memory(session_id: str):
    load_dotenv()
    connection = os.getenv("DATABASE_URL")

    return SQLChatMessageHistory(
        session_id=session_id,
        connection=connection
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.AI = message.AI(file_path="cookbook.pdf", db_directory="./chroma_db", role="chef")
    yield
    app.state.AI = None
app = FastAPI(lifespan=lifespan)

@app.post("/prompt")
async def Sendprompt(request: Request,prompt: UserPrompt):
    chat_history = get_memory(session_id=prompt.Username)
    request.app.state.AI.memory = chat_history
    response = request.app.state.AI.generate_response(
        user_input = prompt.prompt,
        )
    
    return {"response": response}