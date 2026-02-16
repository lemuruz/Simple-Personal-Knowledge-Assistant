import pymupdf 

class readPDF:
    def __init__(self,PDFfile):
        self.PDFfile = PDFfile
        self.PDFtext = ""
        # reader = pypdf.PdfReader(PDFfile)
        reader = pymupdf.open(PDFfile)
        # for i in range(0,len(reader.pages)):
        for page in reader:
            try:
                text = page.get_text("text")
                if text:
                    self.PDFtext += text
            except Exception as e:
                print(f"Skipping problematic page: {e}")
                
        # self.PDFtext = "".join([p.extract_text() for p in reader.pages])
        self.textlen = len(self.PDFtext)
        self.PDFchunked = []
        self.collection = None

    
    def PDFChunking(self,chunkSize=700,overlapSize=20):
        step = chunkSize-overlapSize
        if step <= 0:raise ValueError("Chunk size must be greater than overlap size.")
        for i in range(0,self.textlen,step):
            self.PDFchunked.append(self.PDFtext[i:i+chunkSize])
        
        

    # def PDFEmbeddedAndStore(self,db_path, collection_name):
    #     client = chromadb.PersistentClient(path=db_path)
        
    #     collection = client.get_or_create_collection(name=collection_name)
    #     if collection.count() == 0:
    #         print(f"Embedding new file: {collection_name}")
    #         for i, d in enumerate(self.PDFchunked):
    #             response = ollama.embed(model="mxbai-embed-large", input=d)
    #             embeddings = response["embeddings"]
    #             collection.add(
    #                 ids=[str(i)],
    #                 embeddings=embeddings,
    #                 documents=[d]
    #             )
    #     else:
    #         print(f"Using existing embeddings for: {collection_name}")
    #     self.collection = collection

