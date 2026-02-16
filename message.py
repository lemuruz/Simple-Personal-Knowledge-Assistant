from urllib import response
import PDFChunking
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field


class RecipeOutput(BaseModel):
    message: str = Field(..., description="The generated response from the AI Chef.")
    # source: str = Field(..., description="The source of the information used to generate the response.")
    ingredients: list[str] = Field(..., description="List of ingredients needed for the recipe.")
    instructions: list[str] = Field(..., description="Step-by-step instructions for preparing the dish.") 
    tips_message: str = Field(..., description="Fun tips related to the recipe or cooking in general.")
    
class AI:
    def __init__(self):

        self.llm = ChatOllama(model="llama3", temperature=0).with_structured_output(RecipeOutput)
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        self.file_path = ["cookbook.pdf","THE LOW-COST COOKBOOK .pdf"]
        self.db_directory = "./chroma_db"
        self.role = "chef"
        self.collection_name = [f.split('.')[0] for f in self.file_path]
        self.history = FileChatMessageHistory('.chat_history.json')
        
        self.pdf_reader = [PDFChunking.readPDF(f) for f in self.file_path]
        
        for reader in self.pdf_reader:
            reader.PDFChunking(chunkSize=700, overlapSize=100) 

        documents = []
        for reader, collection in zip(self.pdf_reader, self.collection_name):
            documents.extend([Document(
                page_content=chunk,
                metadata={"source": f"{collection}"}
            ) for chunk in reader.PDFchunked])
            

        self.vector_store = Chroma(
            persist_directory=self.db_directory, 
            embedding_function=self.embeddings,
            collection_name="recipes"
        )
        if self.vector_store._collection.count() == 0:
            print("Embedding new documents...")
            for doc in documents:
                self.vector_store.add_documents([doc])
        else:
            print("Using existing embeddings...")

        self.memory = None
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content=f"""You are a professional {self.role}.
                Respond in a friendly, human conversational tone.

                Also return structured recipe data.

    
                The "message" field should contain a warm, natural explanation or recommendation.
                The "tips_message" field should explain the tips in a fun,friendly, natural paragraph style.
                
                The other fields must contain clean structured data.
                """
                ),

            MessagesPlaceholder(variable_name='chat_history'),
            HumanMessagePromptTemplate.from_template(
                "Context:\n{context}\n\nQuestion:\n{user_input}"
            )
        ])
        print("LangChain Chef is ready!")
    
    def generate_response(self, user_input):
        chat_history = self.memory.messages if self.memory else []

        if chat_history:
            history_text = " ".join([m.content for m in chat_history])
            search_query = f"{history_text} {user_input}"
        else:
            search_query = user_input
        relevant_docs = self.vector_store.similarity_search(search_query, k=6)   
        print(f"Found {relevant_docs} relevant documents.")
        top_source = relevant_docs[0].metadata["source"]

        context = "\n\n".join([
            f"[SOURCE: {doc.metadata['source']}]\n{doc.page_content}"
            for doc in relevant_docs
        ])
        
        chain = self.prompt | self.llm  
        response = chain.invoke({
            "user_input": user_input,
            "context": context,
            "chat_history": chat_history
        })
        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(str(response))

        
        response_dict = response.model_dump()
        response_dict["source"] = top_source
        return response_dict



