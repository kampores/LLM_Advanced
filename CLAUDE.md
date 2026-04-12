# LLM 파인튜닝 실습 환경 및 커리큘럼 정리

## 서버 사양 (강사용 / 이 컴퓨터)

| 항목 | 스펙 |
|------|------|
| CPU | Intel Xeon Silver 4208 x2 (총 32스레드) |
| RAM | 128GB |
| GPU | NVIDIA TITAN RTX 24GB x2 (총 48GB VRAM) |
| 드라이버 | 590.48.01 |
| CUDA | 13.1 |
| 디스크 | 98GB (잔여 6.5GB - 정리 필요!) |

### 서버 용도
- 강사 시연 (13B 등 큰 모델 데모)
- 모델 파일 배포 서버 (NAS/내부 네트워크)
- 학생 결과 취합/평가
- 7B~13B 모델 LoRA 파인튜닝 시연

---

## 실습장 PC 사양

| 항목 | 스펙 |
|------|------|
| GPU | NVIDIA RTX 4060 (8GB VRAM) |
| 수강생 | 25명 (각자 개인 PC에서 실습) |

### RTX 4060 제한사항
- VRAM 8GB → 7B 모델은 QLoRA 필수, 3B 이하는 LoRA 가능
- FFT(Full Fine-Tuning)는 1.5B까지만 가능
- batch_size=1~2, gradient_accumulation_steps=8 권장
- max_seq_length=1024~2048 제한

---

## TITAN RTX vs RTX 4060 비교

| 항목 | TITAN RTX | RTX 4060 |
|------|----------|----------|
| 아키텍처 | Turing (2018) | Ada Lovelace (2023) |
| 공정 | 12nm | 5nm |
| VRAM | 24GB GDDR6 | 8GB GDDR6 |
| TDP | 280W | 115W |
| 게이밍 | 비슷 (~4% 차이) | 비슷 |
| AI/ML 학습 | VRAM 24GB로 우세 | VRAM 부족 |

---

## 실습 환경 점검 명령어 (수강생용)

```bash
# 한 줄로 전체 점검
echo "=== GPU ===" && nvidia-smi -L && echo "=== Python ===" && python3 --version && echo "=== CUDA ===" && nvcc --version 2>/dev/null || echo "nvcc not found" && echo "=== Packages ===" && pip list 2>/dev/null | grep -E "torch|transformers|peft|trl|datasets|accelerate|bitsandbytes" && echo "=== PyTorch GPU ===" && python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')"
```

### 필수 패키지
torch, transformers, peft, trl, datasets, accelerate, bitsandbytes

---

## 커리큘럼 구성

### Part 1: 기초 & 모델 서빙 (01~06)
- 생성 AI와 LLM / sLLM의 부상과 실용성
- 트랜스포머 구조, 생성형 AI 작동 방식
- LLM 발전 과정, 프롬프트 엔지니어링 / RAG / Agent / 파인튜닝 개요
- OpenAI API / Responses API 실습
- LangChain 주요 모듈 소개 및 기본 실습
- 주요 LLM 모델 비교 / Ollama 로컬 모델 서빙
- Transformers 라이브러리 모델 로딩/추론 / HuggingFace 연결

### Part 2: RAG & SFT 이론 (07~15)
- RAG 파이프라인 / 벡터 DB / 시맨틱 서치
- LangChain RAG 어플리케이션 구현
- GPU 클라우드 접속 / Ollama·vLLM 서빙 환경 구축
- RAG 성능 평가 / Advanced RAG 기술 실습
- 딥러닝 기본 원리 / 하이퍼파라미터
- SFT와 RL / Continuous Pretraining / Instruction Tuning
- 데이터 수집/증강/정제 파이프라인
- Synthetic Data Generation / Distillation

### Part 3: 파인튜닝 & Tool Calling (16~24)
- Next Token Prediction 기반 SFT
- Continuous Pretraining / Instruction Tuning 실습
- LoRA / FFT 비교 실습
- Unsloth 기반 파인튜닝
- Tool Calling 파인튜닝 (데이터 준비 → 학습 → 테스트)

### Part 4: 강화학습 & 양자화 & 배포 (25~31)
- PPO/DPO/GRPO 강화학습 기법
- Preference 데이터 수집/생성 실습
- Rejection Sampling + SFT / DPO 성능 향상 실습
- DeepSeek R1 Case Study (추론 모델 학습)
- 양자화 / 서빙 효율 향상
- API 서빙 + Streamlit 웹앱 구현
- 평가 메트릭 / LLM-as-a-judge

### Part 5: 프로젝트 (32~36)
- 도메인 선택, 문제 정의, 데이터 전략 수립
- 데이터 증강/정제 파이프라인 구축
- LoRA/FFT 학습 수행
- 성능 평가 및 반복 개선
- 양자화/배포/어플리케이션 구현
- 도메인 특화 sLLM 성능 향상

---

## 실습별 RTX 4060 적합도

### GPU 불필요 (API 기반)
- OpenAI API, LangChain, RAG, 프롬프트 엔지니어링, Tool Call, 데이터 수집/생성, 평가, Streamlit

### 가능 (모델 크기 제한)
- Ollama 서빙: 1.5B~3B (Q4), 7B 추론만 가능
- Transformers 추론: 3B FP16, 7B 4bit
- LoRA SFT/Instruction Tuning: 1.5B~3B
- QLoRA: 7B까지 빠듯하게 가능
- FFT: 1.5B까지만
- LoRA vs FFT 비교: 1.5B로 통일
- Unsloth: 3B~7B QLoRA (메모리 최적화)
- DPO/GRPO 강화학습: 1.5B~3B
- 양자화: 7B GPTQ/AWQ 가능

### 클라우드 필요
- vLLM 서빙 (7B+)
- FFT 비교 (3B+)
- 13B 모델 실험

---

## 실습 운영 권장사항

1. **실습 모델 통일**: Qwen2.5-1.5B-Instruct 또는 Qwen2.5-3B-Instruct
2. **모델 사전 배포**: USB/NAS로 미리 복사 (25명 동시 HF 다운로드 방지)
3. **HF_HOME 환경변수**: 공유 캐시 경로 지정
4. **GPU 클라우드 병행**: Colab Pro / Kaggle / Runpod (큰 모델 구간)
5. **서버 디스크 정리 필수**: 현재 6.5GB 잔여 → 최소 500GB 확보 필요

---

## 실습 자료 파일 구조

```
LLM_Advanced/
├── CLAUDE.md                      # 이 문서
├── requirements.txt               # 필수 패키지 목록
├── setup_check.ipynb              # 환경 점검 노트북
├── data/samples/                  # 실습용 샘플 데이터
│   ├── alpaca_ko_sample.json      # 한국어 Alpaca 형식 50건
│   ├── chatml_ko_sample.json      # ChatML 형식 샘플
│   ├── preference_sample.json     # chosen/rejected 선호도 30건
│   ├── preference_dpo_format.json # DPO 학습용 선호도 데이터
│   ├── tool_calling_sample.json   # Function calling 20건
│   ├── tool_calling_training_data.json # Tool calling 학습 데이터
│   ├── synthetic_data.json        # 합성 데이터
│   └── domain_text_sample.txt     # AI/ML 도메인 텍스트
├── utils/
│   ├── __init__.py
│   └── gpu_monitor.py             # print_gpu_memory(), clear_gpu_memory()
├── part1_basics/                  # Part 1: 기초 & 모델 서빙 (6개)
│   ├── 01_generative_ai_overview.ipynb
│   ├── 02_llm_landscape.ipynb
│   ├── 03_openai_api.ipynb
│   ├── 04_langchain_basics.ipynb
│   ├── 05_model_comparison_ollama.ipynb
│   └── 06_transformers_huggingface.ipynb
├── part2_serving_rag_sft/         # Part 2: RAG & SFT 이론 (9개)
│   ├── 07_rag_pipeline_vectordb.ipynb
│   ├── 08_langchain_rag_app.ipynb
│   ├── 09_cloud_gpu_vllm.ipynb
│   ├── 10_rag_evaluation.ipynb
│   ├── 11_advanced_rag.ipynb
│   ├── 12_deep_learning_fundamentals.ipynb
│   ├── 13_finetuning_concepts.ipynb
│   ├── 14_data_pipeline.ipynb
│   └── 15_synthetic_data_distillation.ipynb
├── part3_finetuning_tool_calling/ # Part 3: 파인튜닝 & Tool Calling (9개)
│   ├── 16_next_token_prediction_sft.ipynb
│   ├── 17_continuous_pretraining.ipynb
│   ├── 18_instruction_tuning.ipynb
│   ├── 19_lora_vs_fft_theory.ipynb
│   ├── 20_lora_fft_comparison.ipynb
│   ├── 21_unsloth_finetuning.ipynb
│   ├── 22_tool_calling_concepts.ipynb
│   ├── 23_tool_calling_data.ipynb
│   └── 24_tool_calling_finetuning.ipynb
├── part4_reinforcement_quant_deployment/ # Part 4: 강화학습 & 양자화 & 배포 (10개+)
│   ├── 25_rl_concepts.ipynb
│   ├── 26_preference_data.ipynb
│   ├── 26b_rejection_sampling_sft.ipynb
│   ├── 27_dpo_training.ipynb
│   ├── 28_deepseek_r1_case_study.ipynb
│   ├── 29_quantization.ipynb
│   ├── 30_api_serving_streamlit.ipynb
│   ├── 31_evaluation_llm_judge.ipynb
│   ├── quantization_comparison.ipynb
│   ├── quant_perform.ipynb
│   └── quant_simple.ipynb
└── part5_project/                 # Part 5: 프로젝트 (5개)
    ├── 32_project_planning.ipynb
    ├── 33_project_data_pipeline.ipynb
    ├── 34_project_training.ipynb
    ├── 35_project_evaluation.ipynb
    └── 36_project_deployment.ipynb
```

---

## 노트북 상세 목록 (총 36개 + setup + 보충 노트북)

### Part 1: 기초 & 모델 서빙 (part1_basics/)

| # | 파일명 | 세션 | 핵심 내용 | GPU | 셀 수 |
|---|--------|------|-----------|-----|-------|
| - | `setup_check.ipynb` | - | 환경 점검 (GPU, 패키지, API키, 디스크) | No | 12 |
| 1 | `01_generative_ai_overview.ipynb` | 1 | 생성AI 개요, 트랜스포머 구조, 토큰화(tiktoken/BPE/WordPiece), 생성 전략(Greedy/Beam/Top-k/Top-p), 한영 토큰 효율 비교 | No | 19 |
| 2 | `02_llm_landscape.ipynb` | 2 | LLM 발전(GPT→LLaMA→Qwen), sLLM, 프롬프트/RAG/Agent/파인튜닝 비교, 모델 크기 vs 성능 시각화 | No | 17 |
| 3 | `03_openai_api.ipynb` | 3 | Chat Completions API, 시스템 프롬프트, temperature/top_p, 스트리밍, 멀티턴 대화, JSON 모드 | No | 23 |
| 4 | `04_langchain_basics.ipynb` | 4 | LangChain 모듈, ChatPromptTemplate, LCEL 파이프라인, Output Parser, Memory, 챗봇 구현 | No | 24 |
| 5 | `05_model_comparison_ollama.ipynb` | 5 | 오픈소스 LLM 비교, Ollama 설치/설정, REST API, 모델 크기별 벤치마크, 스트리밍 | Yes | 26 |
| 6 | `06_transformers_huggingface.ipynb` | 6 | HF Hub, AutoModel/AutoTokenizer, Pipeline API, 4bit BitsAndBytes 양자화, FP16 vs 4bit 비교 | Yes | 28 |

### Part 2: RAG & SFT 이론 (part2_serving_rag_sft/)

| # | 파일명 | 세션 | 핵심 내용 | GPU | 셀 수 |
|---|--------|------|-----------|-----|-------|
| 7 | `07_rag_pipeline_vectordb.ipynb` | 7 | RAG 개념, 문서 청킹, SentenceTransformer 임베딩, ChromaDB, 시맨틱 서치, 키워드 vs 시맨틱 비교 | No | 23 |
| 8 | `08_langchain_rag_app.ipynb` | 8 | Document Loaders, Text Splitters, VectorStore, RetrievalQA, ConversationalRetrievalChain, LCEL RAG | No | 26 |
| 9 | `09_cloud_gpu_vllm.ipynb` | 9 | GPU 클라우드(Colab/Kaggle/RunPod), vLLM PagedAttention, OpenAI 호환 API 서버, 벤치마크 | Cloud | 25 |
| 10 | `10_rag_evaluation.ipynb` | 10 | RAGAS(Faithfulness/Relevancy/Precision/Recall), LLM-as-a-Judge 구현, 평가 결과 분석 | No | 25 |
| 11 | `11_advanced_rag.ipynb` | 11 | 기본 RAG 한계, HyDE, Reranking(CrossEncoder), Ensemble Retriever(BM25+Semantic), Parent Document Retriever | Optional | 29 |
| 12 | `12_deep_learning_fundamentals.ipynb` | 12 | 활성화함수, Cross-Entropy Loss, 경사하강법/역전파, 옵티마이저(SGD/Adam/AdamW), 과적합/정규화, PyTorch 학습 루프 | Optional | 27 |
| 13 | `13_finetuning_concepts.ipynb` | 13 | 학습 3단계(Pretrain→SFT→RLHF), CPT/IT, FFT vs PEFT, LoRA 원리(행렬 분해/r/alpha), QLoRA(NF4), RTX 4060 가이드 | No | 21 |
| 14 | `14_data_pipeline.ipynb` | 14 | Alpaca/ShareGPT/ChatML 형식, 데이터 수집/정제(DataCleaner), 증강, HF datasets, 품질 검증 체크리스트 | No | 22 |
| 15 | `15_synthetic_data_distillation.ipynb` | 15 | Self-Instruct, GPT-4 데이터 생성, Seed 기반 확장(10→100), Distillation 3가지 전략, 비용 계산 | No | 22 |

### Part 3: 파인튜닝 & Tool Calling (part3_finetuning_tool_calling/)

| # | 파일명 | 세션 | 모델 | GPU | 셀 수 |
|---|--------|------|------|-----|-------|
| 16 | `16_next_token_prediction_sft.ipynb` | 16 | Qwen2.5-1.5B (LoRA SFT) | Yes | 26 |
| 17 | `17_continuous_pretraining.ipynb` | 17 | Qwen2.5-1.5B (LoRA) | Yes | ~25 |
| 18 | `18_instruction_tuning.ipynb` | 18 | Qwen2.5-1.5B-Instruct (LoRA) | Yes | ~25 |
| 19 | `19_lora_vs_fft_theory.ipynb` | 19 | Qwen2.5-1.5B | Yes | ~25 |
| 20 | `20_lora_fft_comparison.ipynb` | 20 | Qwen2.5-1.5B | Yes | ~25 |
| 21 | `21_unsloth_finetuning.ipynb` | 21 | Qwen2.5-3B/7B (Unsloth) | Yes | ~25 |
| 22 | `22_tool_calling_concepts.ipynb` | 22 | gpt-4o-mini (API) | No | ~25 |
| 23 | `23_tool_calling_data.ipynb` | 23 | gpt-4o-mini (합성 데이터) | No | ~25 |
| 24 | `24_tool_calling_finetuning.ipynb` | 24 | Qwen2.5-3B (QLoRA) | Yes | ~25 |

### Part 4: 강화학습 & 양자화 & 배포 (part4_reinforcement_quant_deployment/)

| # | 파일명 | 세션 | 모델 | GPU | 셀 수 |
|---|--------|------|------|-----|-------|
| 25 | `25_rl_concepts.ipynb` | 25 | 없음 (개념/시뮬레이션) | No | 25 |
| 26 | `26_preference_data.ipynb` | 26 | gpt-4o-mini + Qwen2.5-1.5B | Optional | 31 |
| 26b | `26b_rejection_sampling_sft.ipynb` | 26 | Rejection Sampling + SFT | Yes | - |
| 27 | `27_dpo_training.ipynb` | 27 | Qwen2.5-1.5B (SFT→DPO) | Yes | 29 |
| 28 | `28_deepseek_r1_case_study.ipynb` | 28 | Qwen2.5-1.5B (GRPO) | Yes | 31 |
| 29 | `29_quantization.ipynb` | 29 | 양자화 기법 | Yes | 35 |
| 30 | `30_api_serving_streamlit.ipynb` | 30 | API 서빙 + Streamlit | Optional | 26 |
| 31 | `31_evaluation_llm_judge.ipynb` | 31 | LLM-as-a-Judge 평가 | Optional | 31 |
| - | `quantization_comparison.ipynb` | - | 양자화 비교 (보충) | Yes | - |
| - | `quant_perform.ipynb` | - | 양자화 성능 측정 (보충) | Yes | - |
| - | `quant_simple.ipynb` | - | 양자화 간단 실습 (보충) | Yes | - |

### Part 5: 프로젝트 (part5_project/)

| # | 파일명 | 세션 | 셀 수 |
|---|--------|------|-------|
| 32 | `32_project_planning.ipynb` | 32 | 17 |
| 33 | `33_project_data_pipeline.ipynb` | 33 | 22 |
| 34 | `34_project_training.ipynb` | 34 | 23 |
| 35 | `35_project_evaluation.ipynb` | 35 | 22 |
| 36 | `36_project_deployment.ipynb` | 36 | 19 |

---

## 노트북 공통 패턴

### 스타일
- 한국어 설명, 이모지 불릿 (🎯, 1️⃣, 2️⃣, 3️⃣ ...)
- 섹션별 마크다운 + 코드 셀 교대
- print문으로 진행 상황 출력
- 통일 모델: Qwen2.5-1.5B-Instruct (기본), 3B/7B (심화)

### GPU 메모리 모니터링 (모든 GPU 노트북)
```python
import torch, gc
def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[{tag}] GPU: {allocated:.1f}GB / {total:.1f}GB")
```

### RTX 4060 안전 설정 (학습 노트북 기본값)
```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
fp16 = True
max_seq_length = 1024
gradient_checkpointing = True
```

### Before/After 비교 패턴
모든 파인튜닝 노트북에 동일 프롬프트로 학습 전후 비교 포함:
```python
test_prompts = [...]  # 학습 전 추론 → 학습 → 학습 후 추론 → 비교
```

### LoRA 기본 설정
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

### 4bit 양자화 설정
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

---

## 샘플 데이터 설명

| 파일 | 형식 | 용도 |
|------|------|------|
| `alpaca_ko_sample.json` | Alpaca (instruction/input/output) 50건 | Part 2-3 SFT 학습 |
| `chatml_ko_sample.json` | ChatML 형식 | Part 2-3 SFT 학습 |
| `preference_sample.json` | prompt/chosen/rejected 30건 | Part 4 DPO 학습 |
| `preference_dpo_format.json` | DPO 학습용 선호도 | Part 4 DPO 학습 |
| `tool_calling_sample.json` | OpenAI messages 20건 | Part 3 Tool Calling |
| `tool_calling_training_data.json` | Tool calling 학습 데이터 | Part 3 Tool Calling |
| `synthetic_data.json` | 합성 데이터 | Part 2 합성 데이터 실습 |
| `domain_text_sample.txt` | 일반 텍스트 ~3000자 | Part 3 CPT 학습 |

---

## 세션 → 노트북 매핑

| 세션 | 노트북 | Part |
|------|--------|------|
| 1 | 01 | Part 1 |
| 2 | 02 | Part 1 |
| 3 | 03 | Part 1 |
| 4 | 04 | Part 1 |
| 5 | 05 | Part 1 |
| 6 | 06 | Part 1 |
| 7 | 07 | Part 2 |
| 8 | 08 | Part 2 |
| 9 | 09 | Part 2 |
| 10 | 10 | Part 2 |
| 11 | 11 | Part 2 |
| 12 | 12 | Part 2 |
| 13 | 13 | Part 2 |
| 14 | 14 | Part 2 |
| 15 | 15 | Part 2 |
| 16 | 16 | Part 3 |
| 17 | 17 | Part 3 |
| 18 | 18 | Part 3 |
| 19 | 19 | Part 3 |
| 20 | 20 | Part 3 |
| 21 | 21 | Part 3 |
| 22 | 22 | Part 3 |
| 23 | 23 | Part 3 |
| 24 | 24 | Part 3 |
| 25 | 25 | Part 4 |
| 26 | 26, 26b | Part 4 |
| 27 | 27 | Part 4 |
| 28 | 28 | Part 4 |
| 29 | 29 | Part 4 |
| 30 | 30 | Part 4 |
| 31 | 31 | Part 4 |
| 32 | 32 | Part 5 |
| 33 | 33 | Part 5 |
| 34 | 34 | Part 5 |
| 35 | 35 | Part 5 |
| 36 | 36 | Part 5 |

---

## 작업 이력

### 2026-03-21: 09_cloud_gpu_vllm.ipynb 수정

#### 포트 변경: 8000 → 9000
- **이유**: 포트 8000이 다른 앱(`/home/ejkim/workuru/auth/` uvicorn, PID 423321)에 의해 사용 중
- **변경 셀**: cell-5(방식B 예시), cell-9(서버 실행 명령어), cell-10(터미널 출력), cell-12(API 클라이언트 URL/health 체크), cell-14(curl 명령어), cell-15(벤치마크 설명), cell-17(벤치마크 코드)
- **미변경**: cell-11(파라미터 표) — vLLM 기본값 설명이므로 8000 유지

#### 인터랙티브 비교 셀 추가 (cell-13 뒤에 삽입)
- **목적**: Transformers 직접 추론 vs vLLM API 서버 스트리밍 속도를 눈으로 체감
- **구조**:
  - `CUDA_VISIBLE_DEVICES=0`으로 GPU 0에 Transformers 모델 로드
  - vLLM 서버는 GPU 1 (포트 9000)에서 실행 중
  - `input()` 루프로 질문 입력 → 양쪽 스트리밍 출력 → TTFT/총시간/tok/s 비교표
  - `TextIteratorStreamer`로 Transformers 토큰 스트리밍 구현
  - 종료 시 GPU 메모리 자동 해제
- **제한**: 듀얼 GPU 환경 전용 (TITAN RTX ×2)
- **실행 조건**: 커널 재시작 필수 (CUDA_VISIBLE_DEVICES 설정이 torch import 전에 필요)

#### 서버 포트 사용 현황
- 포트 8000: `/home/ejkim/workuru/auth/` (uvicorn, PID 423321) — 건드리지 않음
- 포트 9000: vLLM API 서버 (Qwen2.5-7B-Instruct, GPU 1)
- 포트 11434: Ollama 서버

#### vLLM 서버 실행 명령어 (TITAN RTX)
```bash
CUDA_VISIBLE_DEVICES=1 /home/ejkim/multicampus/venv/bin/python \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype float16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 9000
```

#### HuggingFace 모델 캐시 현황
- `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct` (존재)
- `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct` (존재)
