from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)

import os
from utils.obj import format_json

def load_document(file_path):
    DOCUMENT_LOADER_MAPPING = {
        ".pdf": (PDFPlumberLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
        "doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        "ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".xls": (UnstructuredExcelLoader, {}),
        ".xlsx": (UnstructuredExcelLoader, {}),
        ".csv": (CSVLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".xml": (UnstructuredXMLLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
    }    
    
    ext = os.path.splitext(file_path)[1]
    loader_tuple = DOCUMENT_LOADER_MAPPING.get(ext)
    
    
    if loader_tuple:
        loader_class, loader_args = loader_tuple
        loader = loader_class(file_path, **loader_args)
        documents = loader.load()
        
        content = "\n".join([doc.page_content for doc in documents])
        return content
    print("document is invalid")
    return ""
