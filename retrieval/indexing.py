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
def indexing_process(chunks, embedding_model, store_type):
    embedding_data = embedding_chunks(chunks, embedding_model)
    
    if store_type == StoreEnum.FAISS:
        faiss_client = Faiss.client(embedding_data)
        faiss_client.add(embedding_data)
        return faiss_client
    if store_type == StoreEnum.CHROMA:
        chrome_client = ChromaDB(app_config.database.chroma.chroma_db_path)
        chrome_client.add(chunks, embedding_data, collection_name="documents")
        return chrome_client
    else:
        raise ValueError(f"Invalid store type: {store_type}")
    