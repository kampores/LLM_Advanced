
# finetuned_server.py - 파인튜닝 모델 FastAPI 서버
# 실행: uvicorn finetuned_server:app --port 8002

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time

app = FastAPI(title="Fine-tuned LLM API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

# 설정
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_ADAPTER = "./my-lora-adapter"  # LoRA 어댑터 경로
# 또는 병합된 모델 경로
# MERGED_MODEL = "./merged-model"

@app.on_event("startup")
async def load_model():
    global model, tokenizer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 방법 1: LoRA 어댑터 로딩
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)

    # 방법 2: 병합된 모델 직접 로딩 (더 빠름)
    # model = AutoModelForCausalLM.from_pretrained(
    #     MERGED_MODEL, quantization_config=bnb_config, device_map="auto"
    # )

    print("파인튜닝 모델 로딩 완료!")

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "choices": [{"message": {"role": "assistant", "content": response_text}}]
    }
