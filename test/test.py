import pytest

from main import load_embedding_model
from main import indexing_process
from main import retrieval_process
from main import generate_answer

def test_embedding_model():
    # 定义一些句子
    sentences = [
        "This is an example sentence",
        "Each sentence is converted"
    ]
    
    embeddings = load_embedding_model()
    
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        
def test_index_processing():
    model = load_embedding_model()
    pdf_file = "/Users/yanpingliang/code/python/RAG/source/MIT6.824.pdf" # 使用绝对路径
    indexing_process(pdf_file, model)
    
def test_retrieval_process():
    model = load_embedding_model()
    pdf_file = "/Users/yanpingliang/code/python/RAG/source/MIT6.824.pdf" # 使用绝对路径
    index, chunks = indexing_process(pdf_file, model)
    query = "介绍一下MIT6.824的实验背景"
    retrieval_process(query, index, chunks, model, 5)
    
def test_generate_answer():
    model = load_embedding_model()
    pdf_file = "/Users/yanpingliang/code/python/RAG/source/MIT6.824.pdf" # 使用绝对路径
    index, chunks = indexing_process(pdf_file, model)
    query = "介绍一下MapReduce的Map部分"
    retrieval_chunks = retrieval_process(query, index, chunks, model, 5)
    generate_answer(query, retrieval_chunks)