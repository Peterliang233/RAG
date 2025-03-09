
import dashscope
from config import app_config
from http import HTTPStatus

def generate_answer(query, doc_chunks):
    llm_model = app_config.llm.model
    dashscope.api_key = app_config.llm.api_key

    context = ""
    for i, chunk in enumerate(doc_chunks):
        context += f"参考文档{i+1}:\n{chunk}\n\n"
    
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

