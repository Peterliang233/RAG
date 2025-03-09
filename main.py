from llm_model.embedding_model import load_embedding_model
from retrieval.embedding import doc_chunks, query_embedding
from retrieval.indexing import indexing_process
from generator.generate import generate_answer
from retrieval.retrieval import retrieval_process
from retrieval.retrieval import StoreEnum
from config import app_config

def faiss_main():
    query = "介绍一下MapReduce"
    collection_name="test"
        
    embedding_model = load_embedding_model(app_config.llm.embedding_model)
    
    # 1、doc chunks
    chunks = doc_chunks(app_config.resource_folder_path)
    
    # 2、doc indexing and embedding, then strore the vector 
    store_client = indexing_process(chunks, embedding_model, StoreEnum.FAISS, collection_name)

    # 3、query embedding
    query = "介绍一下MapReduce"
    query_embedding_data = query_embedding(query, embedding_model)
    
    # 4、query retrieval
    retrieval_data = retrieval_process(store_client, chunks, query_embedding_data, StoreEnum.FAISS, collection_name, top_k=5)
    
    # 5、generate answer
    generate_answer(query, retrieval_data)
    
def chroma_main():
    query = "介绍一下MapReduce"
    collection_name="test"
        
    embedding_model = load_embedding_model(app_config.llm.embedding_model)
    
    # 1、doc chunks
    chunks = doc_chunks(app_config.resource_folder_path)
    
    # 2、doc indexing and embedding, then strore the vector 
    store_client = indexing_process(chunks, embedding_model, StoreEnum.CHROMA, collection_name)

    # 3、query embedding
    query = "介绍一下MapReduce"
    query_embedding_data = query_embedding(query, embedding_model)
    
    # 4、query retrieval
    retrieval_data = retrieval_process(store_client, chunks, query_embedding_data, StoreEnum.CHROMA, collection_name, top_k=5)
    
    # 5、generate answer
    generate_answer(query, retrieval_data)
    
    
    
if __name__ == "__main__":
    faiss_main()