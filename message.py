import ollama,PDFChunking,os,chromadb

FILE_PATH = "cookbook.pdf"
Database_Directory = "./chroma_db"
collectionName = FILE_PATH.split('.')[0]




pdf = PDFChunking.readPDF(FILE_PATH)
pdf.PDFChunking(chunkSize=700, overlapSize=20)
pdf.PDFEmbeddedAndStore(db_path=Database_Directory,collection_name=collectionName)


user_input = "What should I cook today?"

response = ollama.embed(
  model="mxbai-embed-large",
  input=user_input
)
results = pdf.collection.query(
  query_embeddings=[response["embeddings"][0]],
  n_results=2
)
data = results['documents'][0][0]
message = [
    {
        'role':'system',
        'content':'You are a professional chef.',
    },
    {
        'role': 'user',
        'content': f"Question: {user_input} \n\n Context from my book: {data}",
    },
]

client = ollama.Client()
for responsePart in client.chat('llama3',messages=message,
    stream=True):
    print(responsePart.message.content,end='',flush=True)