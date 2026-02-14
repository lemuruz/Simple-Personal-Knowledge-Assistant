import PDFChunking
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage


class AI:
    def __init__(self, file_path, db_directory,role = ""):
        self.llm = ChatOllama(model="llama3")
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        self.file_path = file_path
        self.db_directory = db_directory
        self.collection_name = file_path.split('.')[0]
        self.history = FileChatMessageHistory('.chat_history.json')

        self.pdf_reader = PDFChunking.readPDF(file_path)
        self.pdf_reader.PDFChunking(chunkSize=700, overlapSize=100) 

        documents = [Document(page_content=chunk) for chunk in self.pdf_reader.PDFchunked]

        self.vector_store = Chroma(
            persist_directory=db_directory, 
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        if self.vector_store._collection.count() == 0:
            print("Embedding new documents...")
            self.vector_store.add_documents(documents)
        else:
            print("Using existing embeddings...")

        self.memory = None
        self.prompt = ChatPromptTemplate(
            input_variables=['content'],
            messages=[
                SystemMessage(content=f'You are a professional {role}.'),
                MessagesPlaceholder(variable_name='chat_history'),
                HumanMessagePromptTemplate.from_template(
                    "Context:\n{context}\n\nQuestion:\n{user_input}"
                )
            ]
        )
        print("LangChain Chef is ready!")
    
    def generate_response(self, user_input):
        chat_history = self.memory.messages if self.memory else None
        if chat_history:
            history_text = " ".join([m.content for m in chat_history])
            search_query = f"{history_text} {user_input}"
        else:
            search_query = user_input
        relevant_docs = self.vector_store.similarity_search(search_query, k=6)    
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        
        chain = self.prompt | self.llm
        response = chain.invoke({
            "user_input": user_input,
            "context": context,
            "chat_history": chat_history
        })
        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(response.content)
        return response.content




# llm = ChatOllama(model="llama3")
# embeddings = OllamaEmbeddings(model="mxbai-embed-large")


# FILE_PATH = "cookbook.pdf"
# Database_Directory = "./chroma_db"
# collectionName = FILE_PATH.split('.')[0]
# history = FileChatMessageHistory('.chat_history.json')

# pdf_reader = PDFChunking.readPDF(FILE_PATH)
# pdf_reader.PDFChunking(chunkSize=700, overlapSize=100) 

# documents = [Document(page_content=chunk) for chunk in pdf_reader.PDFchunked]

# vector_store = Chroma(
#     persist_directory=Database_Directory, 
#     embedding_function=embeddings,
#     collection_name=collectionName
# )
# if vector_store._collection.count() == 0:
#     print("Embedding new documents...")
#     vector_store.add_documents(documents)
# else:
#     print("Using existing embeddings...")

# memory: ConversationBufferMemory = ConversationBufferMemory(
#     memory_key='chat_history',
#     chat_memory=history,
#     return_messages=True
# )

# print("LangChain Chef is ready!")

# prompt = ChatPromptTemplate(
#     input_variables=['content'],
#     messages=[
#         SystemMessage(content='You are a professional chef.'),
#         MessagesPlaceholder(variable_name='chat_history'),
#         HumanMessagePromptTemplate.from_template(
#             "Context:\n{context}\n\nQuestion:\n{user_input}"
#         )
#     ]
# )



# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit": break

    
#     chat_history = memory.load_memory_variables({})['chat_history']
#     if chat_history:
#         # last_question = chat_history[-1][0]
#         search_query = f"{chat_history} {user_input}"
#     else:
#         search_query = user_input
#     relevant_docs = vector_store.similarity_search(search_query, k=6)    
#     context = "\n\n".join([doc.page_content for doc in relevant_docs])
    

#     chain = prompt | llm
#     response = chain.invoke({
#         "user_input": user_input,
#         "context": context,
#         "chat_history": chat_history
#     })
#     memory.save_context({"content": user_input}, {"content": response.content})
#     print("Chef:", response.content)
#     # print("Chef:", response['answer'])



