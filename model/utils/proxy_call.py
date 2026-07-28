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

    def chat(self, messages, model="deepseek-v4-flash", temperature=0,
             tools=None, tool_choice=None):
        """对话，支持 tools/function calling。
        返回:
          - 如果有 tool_calls → (Message对象, reasoning_content)
          - 否则 → (content_str, reasoning_content)
        """
        kwargs = dict(model=model, messages=messages, temperature=temperature)
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        reasoning = getattr(msg, 'reasoning_content', '') or ''

        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            return msg, reasoning
        return msg.content, reasoning

    def stream_chat(self, messages, model="deepseek-v4-flash", temperature=0):
        """流式调用，返回 (reasoning_content, content) 元组"""
        for chunk in self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        ):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None) or ""
            content = getattr(delta, "content", None) or ""
            if reasoning or content:
                yield (reasoning, content)

    def stream_chat_with_tools(self, messages, model="deepseek-v4-flash", temperature=0, tools=None):
        """流式调用 + function calling 支持。

        边流式输出 reasoning/content，边累积 tool_calls（如有）。
        流结束后若累积了 tool_calls 则返回之，否则返回最终文本。

        Yields:
            dict: {"type": "reasoning", "content": "..."}
            dict: {"type": "content", "content": "..."}
            dict: {"type": "done", "final_text": "..." | None, "tool_calls": [...] | None}
        """
        import json as _json

        accumulated_reasoning = []
        accumulated_content = []
        accumulated_tool_calls = {}  # {index: {"id":..., "name":..., "arguments":""}}

        for chunk in self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            stream=True,
            stream_options={"include_usage": True},
        ):
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # 推理过程（DeepSeek 特有）
            reasoning = getattr(delta, "reasoning_content", None) or ""
            if reasoning:
                accumulated_reasoning.append(reasoning)
                yield {"type": "reasoning", "content": reasoning}

            # 普通文本
            content = getattr(delta, "content", None) or ""
            if content:
                accumulated_content.append(content)
                yield {"type": "content", "content": content}

            # 工具调用增量
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "id": tc.id or "",
                            "function": {"name": "", "arguments": ""}
                        }
                    if tc.id:
                        accumulated_tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            accumulated_tool_calls[idx]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            accumulated_tool_calls[idx]["function"]["arguments"] += tc.function.arguments

            # 流结束
            if finish_reason == "tool_calls":
                tool_list = [
                    {
                        "id": v["id"],
                        "function": {
                            "name": v["function"]["name"],
                            "arguments": v["function"]["arguments"]
                        }
                    }
                    for v in sorted(accumulated_tool_calls.values(), key=lambda x: int(x.get("_idx", 0)))
                ]
                yield {
                    "type": "done",
                    "final_text": "".join(accumulated_content) if accumulated_content else None,
                    "tool_calls": tool_list,
                    "reasoning": "".join(accumulated_reasoning),
                }
            elif finish_reason == "stop":
                yield {
                    "type": "done",
                    "final_text": "".join(accumulated_content),
                    "tool_calls": None,
                    "reasoning": "".join(accumulated_reasoning),
                }
            # 其他情况（如 length）也视为结束
            elif finish_reason and finish_reason != "tool_calls":
                yield {
                    "type": "done",
                    "final_text": "".join(accumulated_content) if accumulated_content else None,
                    "tool_calls": None,
                    "reasoning": "".join(accumulated_reasoning),
                }
    
    def embedding(self, input_data):
        """使用本地 BAAI/bge-small-zh-v1.5 生成 embedding，返回 numpy 数组 (n, 512)"""
        if isinstance(input_data, str):
            input_data = [input_data]
        return self.embedding_model.encode(input_data, normalize_embeddings=True)