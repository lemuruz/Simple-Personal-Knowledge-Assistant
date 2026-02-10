import ollama,PDFChunking
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


FILE_PATH = "cookbook.pdf"
Database_Directory = "./chroma_db"
collectionName = FILE_PATH.split('.')[0]


vector_store = Chroma(
    persist_directory=Database_Directory, 
    embedding_function=embeddings,
    collection_name=collectionName
)

retriever = vector_store.as_retriever()
prompt = ChatPromptTemplate.from_template("""
You are a professional chef. Answer the user's question based ONLY on the context below.
                                          
<context>
{context}
</context>

Question: {input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
print("LangChain Chef is ready!")

# test
# test_docs = retriever.invoke("Stuffed Potatoes")
# print(f"Found {len(test_docs)} chunks!")
# print(f"First chunk content: {test_docs[0].page_content[:100]}...")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit": break
    
    # This one line does: Embedding -> Searching -> Context Stuffing -> Generating
    response = retrieval_chain.invoke({"input": user_input})
    
    print("Chef:", response['answer'])

# pdf = PDFChunking.readPDF(FILE_PATH)
# pdf.PDFChunking(chunkSize=700, overlapSize=20)
# pdf.PDFEmbeddedAndStore(db_path=Database_Directory,collection_name=collectionName)



# print("Chef Bot is ready! Type 'exit' to stop.")
# # user_input = "What should I cook today?"
# conversation_history = [
#     {'role': 'system', 'content': 'You are a professional chef.'}
# ]
# while True:
#     user_input = input("input : ")
#     if user_input == "exit" or "quit":
#         break
#     response = ollama.embed(
#         model="mxbai-embed-large",
#         input=user_input
#     )
#     results = pdf.collection.query(
#         query_embeddings=[response["embeddings"][0]],
#         n_results=2
#     )
#     data = results['documents'][0][0]
#     conversation_history.append(
#         {
#             'role': 'user',
#             'content': f"Question: {user_input} \n\n Context from my book: {data}",
#         },
#     )

#     client = ollama.Client()
#     fullResponse = ""
#     for responsePart in client.chat('llama3',messages=conversation_history,
#         stream=True):
#         text = responsePart.message.content
#         print(text,end='',flush=True)
#         fullResponse += responsePart.message.content
#     print("\n")
#     conversation_history.append(
#         {
#             'role': 'assistant',
#             'content': fullResponse
#         }
#     )

