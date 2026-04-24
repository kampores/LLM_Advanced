
# server.py - FastAPI LLM 서버
# 실행: uvicorn server:app --host 0.0.0.0 --port 9200

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import uvicorn

# ============================================
# 1. FastAPI 앱 생성
# ============================================
app = FastAPI(
    title="LLM API Server",
    description="파인튜닝된 LLM 모델 API 서버",
    version="1.0.0"
)

# ============================================
# 2. 요청/응답 스키마 정의
# ============================================
class Message(BaseModel):
    role: str          # "system", "user", "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    time_seconds: float

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7

# ============================================
# 3. 모델 로딩 (서버 시작 시 1회)
# ============================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print(f"모델 로딩 중: {MODEL_NAME}")

    # 4bit 양자화로 로딩 (RTX 4060 호환)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print("모델 로딩 완료!")

# ============================================
# 4. API 엔드포인트
# ============================================

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 API"""
    try:
        # 메시지를 chat template으로 변환
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
            )
        elapsed = time.time() - start_time

        # 생성된 텍스트 추출
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return ChatResponse(
            response=response_text,
            tokens_generated=len(new_tokens),
            time_seconds=round(elapsed, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate(request: GenerateRequest):
    """텍스트 생성 API"""
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# 5. 서버 실행
# ============================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9200)
