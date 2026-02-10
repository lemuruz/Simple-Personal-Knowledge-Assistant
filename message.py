import ollama,PDFChunking,os,chromadb

FILE_PATH = "cookbook.pdf"
Database_Directory = "./chroma_db"
collectionName = FILE_PATH.split('.')[0]




pdf = PDFChunking.readPDF(FILE_PATH)
pdf.PDFChunking(chunkSize=700, overlapSize=20)
pdf.PDFEmbeddedAndStore(db_path=Database_Directory,collection_name=collectionName)



print("Chef Bot is ready! Type 'exit' to stop.")
# user_input = "What should I cook today?"
conversation_history = [
    {'role': 'system', 'content': 'You are a professional chef.'}
]
while True:
    user_input = input("input : ")
    if user_input == "exit":
        break
    response = ollama.embed(
        model="mxbai-embed-large",
        input=user_input
    )
    results = pdf.collection.query(
        query_embeddings=[response["embeddings"][0]],
        n_results=2
    )
    data = results['documents'][0][0]
    conversation_history.append(
        {
            'role': 'user',
            'content': f"Question: {user_input} \n\n Context from my book: {data}",
        },
    )

    client = ollama.Client()
    fullResponse = ""
    for responsePart in client.chat('llama3',messages=conversation_history,
        stream=True):
        text = responsePart.message.content
        print(text,end='',flush=True)
        fullResponse += responsePart.message.content
    print("\n")
    conversation_history.append(
        {
            'role': 'assistant',
            'content': fullResponse
        }
    )

