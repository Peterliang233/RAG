from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import os
from retrieval.loader import load_document

def embedding_data(chunks, embedding_model):
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk, normalize_before=True)
        embeddings.append(embedding)
    
    embeddings_np = np.array(embeddings)
    
    dimension = embeddings_np.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    return index, chunks

def indexing_process(folder_path, embedding_model):
    all_chunks = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            document_text = load_document(file_path)
            # print(f"docuemnt total length: {len(document_text)}")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 512,
                chunk_overlap = 128
            )
            chunks = text_splitter.split_text(document_text)
            all_chunks.extend(chunks)

    index, chunks = embedding_data(all_chunks, embedding_model)
    
    return index, chunks

def retrieval_process(query, index, chunks, embedding_model, top_k = 3):
    query_embedding = embedding_model.encode(query, normalize_before=True)
    query_embedding = np.array([query_embedding])
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        result_chunk = chunks[indices[0][i]]
        result_distance = distances[0][i]
        results.append(result_chunk)
        
    return result_chunk
    
        