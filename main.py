from config.llm_model import ali_api_key, qwen_model
from llm_model.embedding_model import load_embedding_model
from llm_model import min_lm_model
from retrieval.embedding import indexing_process, retrieval_process
from generator.generate import generate_answer

def main():
    query = "介绍一下MapReduce"
    
    embedding_model = load_embedding_model(min_lm_model)

    folder_path = "source/"
    index, chunks = indexing_process(folder_path, embedding_model)
    
    result_chunks = retrieval_process(query, index, chunks, embedding_model)
    
    generate_answer(query, result_chunks)
    
    
if __name__ == "__main__":
    main()
    