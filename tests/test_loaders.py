from retrieval.loader import load_web_page, load_pdf_file, load_data
from retrieval import WebPage, PDF
from utils.obj import format_json

def test_web_page():
    docs = load_web_page("https://blog.peterliang.top")
    
def test_pdf():
    docs = load_pdf_file("/Users/yanpingliang/code/python/RAG/source/MIT6.824.pdf")
    
def test_loader():
    docs = load_data("https://blog.peterliang.top", WebPage)
    print(format_json(docs))
    