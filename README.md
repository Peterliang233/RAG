# Basic
+ LLM_model use qwen-turbo, which is a large language model developed by Alibaba.And you need to get your ali_api_key: <a href="https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen?spm=a2c4g.11186623.0.0.731a7468iv7kWt">click me</a>. Then set the secret key in your local .env file, such as `export ALI_API_KEY=xxx` in your ~/.zshrc or ~/.bashrc file.
+ Embedding_model use all-MiniLM-L6-v2, which is a sentence embedding model developed by Sentence Transformers.
# Chunk
+ chunk_size: 512
+ chunk_overlap: 20
# VectorStore