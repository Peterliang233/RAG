from database.chroma.chroma import ChromaDB
from database.faiss.faiss import Faiss
from config import app_config
from retrieval import StoreEnum
from retrieval.embedding import embedding_chunks

"""doc -> vectors -> store
    Args:
        resource_folder_path (str): resource folder path
        embedding_model (object): embedding model
        store_type (str): store type
 """
def indexing_process(chunks, embedding_model, store_type, collection_name):    
    if store_type == StoreEnum.FAISS:
        embedding_data = embedding_chunks(chunks, embedding_model)
        faiss_client = Faiss(embedding_data)
        faiss_client.add(embedding_data)
        return faiss_client
    if store_type == StoreEnum.CHROMA:
        embedding_data = [embedding_model.encode(chunk, normalize_before=True).tolist() for chunk in chunks]
        chrome_client = ChromaDB(app_config.database.chroma.chroma_db_path)
        chrome_client.add(chunks, embedding_data, collection_name=collection_name)
        return chrome_client
    else:
        raise ValueError(f"Invalid store type: {store_type}")
    