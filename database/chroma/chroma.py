import chromadb
import shutil
import os
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
        
    def add(self, chunks, embedding_data, collection_name):
        all_ids = [[str(uuid.uuid4()) for _ in range(len(chunks))]]
        collection = self.get_or_create_doc(collection_name)
        collection.add(ids = all_ids, embeddings = embedding_data, documents = chunks)
    
    def search(self, query_embedding_data, collection_name, top_k):
        collection = self.client.get_or_create_collection(name=collection_name)
        results = collection.query(query_embeddings=query_embedding_data, n_results=top_k)
        return results