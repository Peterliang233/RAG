import faiss
import numpy as np

class Faiss:
    def client(self, embeddings):
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        
        return self
    
    def add(self, vectors):
        self.index.add(vectors)
        
    def search(self, query, k = 5):
        return self.index.search(query, k)