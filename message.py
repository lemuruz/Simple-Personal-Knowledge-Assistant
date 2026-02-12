import ollama,PDFChunking
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
llm = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


FILE_PATH = "cookbook.pdf"
Database_Directory = "./chroma_db"
collectionName = FILE_PATH.split('.')[0]

pdf_reader = PDFChunking.readPDF(FILE_PATH)
pdf_reader.PDFChunking(chunkSize=700, overlapSize=100) 

documents = [Document(page_content=chunk) for chunk in pdf_reader.PDFchunked]

vector_store = Chroma(
    persist_directory=Database_Directory, 
    embedding_function=embeddings,
    collection_name=collectionName
)
if vector_store._collection.count() == 0:
    print("Embedding new documents...")
    vector_store.add_documents(documents)
else:
    print("Using existing embeddings...")

# retriever = vector_store.as_retriever()
# prompt = ChatPromptTemplate.from_template("""
# You are a professional chef. Answer the user's question based ONLY on the context below.
                                          
# <context>
# {context}
# </context>

# Question: {input}
# """)
# document_chain = create_stuff_documents_chain(llm, prompt)
# retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("LangChain Chef is ready!")

# test
# test_docs = retriever.invoke("Stuffed Potatoes")
# print(f"Found {len(test_docs)} chunks!")
# print(f"First chunk content: {test_docs[0].page_content[:100]}...")
full_prompt = """You are a professional chef. Answer the user's question based ONLY on the context below.                                        
    <context>
    {context}
    </context>
    Chat History:{chat_history}
    Question: {user_input}
    """
chat_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit": break

    

    if chat_history:
        # last_question = chat_history[-1][0]
        search_query = f"{chat_history} {user_input}"
    else:
        search_query = user_input
    relevant_docs = vector_store.similarity_search(search_query, k=6)    
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    

    # This one line does: Embedding -> Searching -> Context Stuffing -> Generating
    # response = retrieval_chain.invoke({"input": user_input})
    # Generate answer (what llm does)
    response = llm.invoke(full_prompt.format(context=context, user_input=user_input,chat_history=chat_history))
    chat_history.append((user_input, response.content))
    print("Chef:", response.content)
    # print("Chef:", response['answer'])



