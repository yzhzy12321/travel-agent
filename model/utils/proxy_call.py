import os
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from model.config import api_key,base_url
class OpenaiCall:
    def __init__(self, api_key: str = None, base_url: str = None):
        # Chat/LLM 调用使用 DeepSeek（兼容 OpenAI SDK）
        deepseek_key = api_key or os.getenv("DEEPSEEK_API_KEY") or globals().get("api_key", "")
        deepseek_base = base_url or os.getenv("DEEPSEEK_BASE_URL") or globals().get("base_url", "")
        self.client = OpenAI(
            api_key=deepseek_key,
            base_url=deepseek_base
        )
        # Embedding 使用本地 sentence-transformers 模型（无需 API）
        self.embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    def chat(self, messages, model="deepseek-v4-flash", temperature=0):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    def stream_chat(self, messages, model="deepseek-v4-flash", temperature=0):
        for chunk in self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        ):
            yield chunk.choices[0].delta.content
    
    def embedding(self, input_data):
        """使用本地 BAAI/bge-small-zh-v1.5 生成 embedding，返回 numpy 数组 (n, 512)"""
        if isinstance(input_data, str):
            input_data = [input_data]
        return self.embedding_model.encode(input_data, normalize_embeddings=True)