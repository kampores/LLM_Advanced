
# openai_server.py - OpenAI 호환 API 서버
# 실행: uvicorn openai_server:app --host 0.0.0.0 --port 9200

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
import time
import json
import uuid
import uvicorn

app = FastAPI(title="OpenAI Compatible API")

# ============================================
# OpenAI 호환 스키마
# ============================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

# 모델 로딩
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

@app.on_event("startup")
async def startup():
    global model, tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    print("모델 로딩 완료!")

# ============================================
# OpenAI 호환 엔드포인트
# ============================================
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    if request.stream:
        # 스트리밍 응답
        return StreamingResponse(
            stream_generate(inputs, request),
            media_type="text/event-stream"
        )
    else:
        # 일반 응답
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # OpenAI 형식으로 응답
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(new_tokens),
                "total_tokens": inputs["input_ids"].shape[1] + len(new_tokens)
            }
        }

async def stream_generate(inputs, request):
    """스트리밍 생성"""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        do_sample=request.temperature > 0,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text in streamer:
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"

# 모델 목록 (OpenAI 호환)
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "local-model",
            "object": "model",
            "owned_by": "local"
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9200)
