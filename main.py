from fastapi import FastAPI,Request
from httpcore import request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
import message,os
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory


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


# def get_memory(request: Request,session_id: str):
#     load_dotenv()
#     connection = os.getenv("DATABASE_URL")

#     def get_session_history(session_id: str):
#         return SQLChatMessageHistory(
#             session_id=session_id,
#             connection=connection
#         )
#     memory = RunnableWithMessageHistory(
#         request.app.state.AI,
#         get_session_history,
#         input_messages_key=request.app.state.AI.prompt.input_variables[0],
#         history_messages_key="chat_history",
#     )
#     return memory


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