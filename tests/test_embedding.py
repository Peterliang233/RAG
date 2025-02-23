from retrieval.embedding import embedding_data
from llm_model.embedding_model import load_embedding_model
from llm_model import min_lm_model
from retrieval import PDF

def test_embedding():
    embedding_model = load_embedding_model(min_lm_model)
    pdf_file = "/Users/yanpingliang/code/python/RAG/source/MIT6.824.pdf" # 使用绝对路径
    index, chunks = embedding_data(pdf_file, PDF, embedding_model)
    print(chunks)