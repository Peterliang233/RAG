import os
import shutil
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrieval.loader import load_document
import uuid

class ChromaDB:
    def __init__(self, chroma_db_path):
        self.chroma_db_path = chroma_db_path
        if os.path.exists(self.chroma_db_path):
            shutil.rmtree(self.chroma_db_path)
        self.client = self.init__chromadb_client()
        
        
    def init__chromadb_client(self):
        client = chromadb.PersistentClient(path = self.chroma_db_path)
        return client
    
    def get_or_create_doc(self, document_name):
        return self.client.get_or_create_collection(name = document_name)
    
    def doc_indexing_process(self, folder_path, embedding_model, collection):
        all_chunks = []
        all_ids = []
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path):
                document_text = load_document(file_path)
                # print(f"docuemnt total length: {len(document_text)}")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 512,
                    chunk_overlap = 128,
                    separators=["\n\n", "。", "\n", "！", "？", " ", ""],
                )
                chunks = text_splitter.split_text(document_text)
                all_chunks.extend(chunks)
                all_ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])
        
        embeddings = [embedding_model.encode(chunk, normalize_before=True).tolist() for chunk in all_chunks]
        collection.add(ids = all_ids, embeddings = embeddings, documents = all_chunks)

        return all_ids, all_chunks, embeddings
    
    def query_retrieval_process(self, query, collection, embedding_model=None, top_k = 3):
        query_embedding = embedding_model.encode(query, normalize_before=True).tolist()
        
        result = collection.query(
            query_embeddings = [query_embedding],
            n_results = top_k
        )
        
        retrieved_chunks = []
        for doc_id, doc, score in zip(result['ids'][0], result['documents'][0], result['distances'][0]):
            print(f"doc_id: {doc_id}, doc: {doc}, score: {score}")
            retrieved_chunks.append(doc)
        
        return retrieved_chunks