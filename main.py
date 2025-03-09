from llm_model.embedding_model import load_embedding_model
from retrieval.embedding import doc_chunks, query_embedding
from retrieval.indexing import indexing_process
from generator.generate import generate_answer
from retrieval.retrieval import retrieval_process
from retrieval.retrieval import StoreEnum
from config import app_config

def faiss_main():
    query = "介绍一下MapReduce"
    
    embedding_model = load_embedding_model(app_config.llm.embedding_model)

    folder_path = "source/"
    index, chunks = indexing_process(folder_path, embedding_model)
    
    result_chunks = retrieval_process(query, index, chunks, embedding_model)
    
    generate_answer(query, result_chunks)
    
def chroma_main():
    query = "介绍一下MapReduce"
        
    embedding_model = load_embedding_model(app_config.llm.embedding_model)
    
    # 1、doc chunks
    chunks = doc_chunks(app_config.resource_folder_path)
    
    # 2、doc indexing and embedding, then strore the vector 
    store_client = indexing_process(chunks, embedding_model, StoreEnum.CHROMA)

    # 3、query embedding
    query = "介绍一下MapReduce"
    query_embedding_data = query_embedding(query, embedding_model)
    
    # 4、query retrieval
    retrieval_data = retrieval_process(store_client, chunks, query_embedding_data, StoreEnum.CHROMA)
    
    # 5、generate answer
    generate_answer(query, retrieval_data)
    
    
    
if __name__ == "__main__":
    chroma_main()