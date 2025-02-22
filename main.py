from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import ali_api_key, qwen_model
import numpy as np
import dashscope
import faiss
from http import HTTPStatus


def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def indexing_process(pdf_file, embedding_model):
    pdf_loader = PyPDFLoader(pdf_file, extract_images=False)
    
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 128
    )
    
    pdf_content_list = pdf_loader.load()
    
    print(f"PDF页数：{len(pdf_content_list)}")
    
    pdf_text = "\n".join([page.page_content for page in pdf_content_list])
    
    print(f"pdf内容:{pdf_text}")
    
    print(f"PDF文本长度：{len(pdf_text)}")
    
    chunks = text_spliter.split_text(pdf_text)
    print(f"PDF文本分割后长度：{len(chunks)}")
    
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk, normalize_before=True)
        embeddings.append(embedding)
            
    embeddings_np = np.array(embeddings)
    
    print(f"向量化结果:{embeddings_np}")
    
    dimension = embeddings_np.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    print(f"完成索引")

    return index, chunks


def retrieval_process(query, index, chunks, embedding_model, top_k = 3):
    query_embedding = embedding_model.encode(query, normalize_before=True)
    query_embedding = np.array([query_embedding])
    
    distances, indices = index.search(query_embedding, top_k)
    
    print(f"查询语句: {query}")
    print(f"最相似的前{top_k}个文本块")
    
    results = []
    for i in range(top_k):
        result_chunk = chunks[indices[0][i]]
        print(f"文本块 {i}:\n{result_chunk}")
        
        result_distance = distances[0][i]
        print(f"相似度: {result_distance}\n")
        
        results.append(result_chunk)
        
    print(f"完成检索")
    
    return results


def generate_answer(query, chunks):
    llm_model = qwen_model
    dashscope.api_key = ali_api_key
    
    context = ""
    for i, chunks in enumerate(chunks):
        context += f"参考文档{i+1}:\n{chunks}\n\n"
    
    prompt = f"根据参考文档回答问题:{query}\n\n{context}"
    print(f"prompt:{prompt}")
    
    
    messages = [{'role': 'user', 'content': prompt}]
    
    try:
        responses = dashscope.Generation.call(
            model = llm_model,
            messages = messages,
            result_format = "message",
            stream = True,
            incremental_output = True
        )
        
        generated_reponse = ""
        print("生成过程开始")
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0]['message']['content']
                generated_reponse += content
                print(content, end ='')
            else:
                print(f"请求失败: {response.status_code} - {response.message}")
                return None
        print("\n生成过程结束")
        return generated_reponse
    except Exception as e:
        print(f"请求失败: {e}") 
        return None
