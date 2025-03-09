from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import os
from retrieval.loader import load_document


def embedding_chunks(chunks, embedding_model):
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk, normalize_before=True)
        embeddings.append(embedding)
    
    embeddings_np = np.array(embeddings)

    return embeddings_np

    """doc chunks
    Args:
        folder_path (str): resource folder path
    Returns:
        chunks (list): all chunks after textsplitter
    """
def doc_chunks(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path):
            document_text = load_document(file_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 512,
                chunk_overlap = 128,
                separators=["\n\n", "。", "\n", "！", "？", " ", ""],
            )
            chunks = text_splitter.split_text(document_text)
            all_chunks.extend(chunks)

    return all_chunks


def query_embedding(query, embedding_model):
    query_embedding = embedding_model.encode(query, normalize_before=True)
    query_embedding = np.array([query_embedding])
    
    return query_embedding