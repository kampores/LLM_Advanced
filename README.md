# LLM 파인튜닝 실습 과정

멀티캠퍼스 LLM 파인튜닝 과정 실습 자료입니다.

> **Copyright AIDENTIFY. All rights reserved.**
> 본 자료는 멀티캠퍼스 LLM 파인튜닝 과정 수강생을 위해 제작되었으며, 강의 목적으로만 사용 가능합니다.

---

## 빠른 시작 가이드

### 1. 저장소 복제

```bash
git clone https://github.com/choki0715/LLM_Advanced.git
cd LLM_Advanced
```

### 2. 환경 설정 (자동)

```bash
bash setup.sh
```

이 스크립트가 자동으로 처리하는 항목:
- Python 가상환경(venv) 생성 및 활성화
- PyTorch (CUDA 12.1) 설치
- 필수 패키지 설치 (transformers, peft, trl, bitsandbytes 등)
- Jupyter 커널 등록 (`Python (LLM)`)
- Ollama 설치
- `.env` 파일 생성
- GPU 점검

### 3. API 키 설정

```bash
# .env 파일을 열어서 본인의 API 키를 입력
vi .env
```

```
OPENAI_API_KEY=sk-your-api-key-here
HF_TOKEN=hf_your-token-here
```
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

- **OPENAI_API_KEY**: Part 1, 3, 4, 5 실습에 필요
- **HF_TOKEN**: HuggingFace 모델 다운로드에 필요 ([발급 링크](https://huggingface.co/settings/tokens))

### 4. 환경 점검

VS Code에서 커널을 `Python (LLM)`으로 선택한 뒤:

```
setup_check.ipynb 실행
```

GPU, 패키지, API 키가 정상인지 확인합니다.

### 5. 가상환경 활성화 (매번 터미널 열 때)

```bash
source venv/bin/activate
```

---

## 실습 환경 요구사항

| 항목 | 최소 사양 |
|------|----------|
| GPU | NVIDIA RTX 4060 (8GB VRAM) 이상 |
| Python | 3.10 이상 |
| CUDA | 12.1 이상 |
| 디스크 | 50GB 이상 여유 공간 |

---

## 커리큘럼 (총 36개 노트북, 5 Parts)

| Part | 폴더 | 주제 | 노트북 | GPU |
|------|------|------|--------|-----|
| **Part 1** | `part1_basics/` | 기초, API, 모델 서빙 | 01~06 | 일부 필요 |
| **Part 2** | `part2_serving_rag_sft/` | RAG, 파인튜닝 이론 | 07~15 | 일부 필요 |
| **Part 3** | `part3_finetuning_tool_calling/` | 파인튜닝 실습, Tool Calling | 16~24 | 필요 |
| **Part 4** | `part4_reinforcement_quant_deployment/` | 강화학습, 양자화, 배포 | 25~31 | 필요 |
| **Part 5** | `part5_project/` | 프로젝트 | 32~36 | 필요 |

---

## 폴더 구조

```
LLM_Advanced/
├── setup.sh                              # 환경 자동 설정 스크립트
├── setup_check.ipynb                     # 환경 점검 노트북
├── .env.example                          # API 키 템플릿
├── data/samples/                         # 실습용 샘플 데이터
├── utils/                                # 유틸리티 (GPU 모니터링 등)
├── part1_basics/                         # 01~06: 기초, API, 모델 서빙
├── part2_serving_rag_sft/                # 07~15: RAG, 파인튜닝 이론
├── part3_finetuning_tool_calling/        # 16~24: 파인튜닝 실습, Tool Calling
├── part4_reinforcement_quant_deployment/ # 25~31: 강화학습, 양자화, 배포
└── part5_project/                        # 32~36: 프로젝트
```

---

## 문제 해결


### GPU를 인식하지 못하는 경우

```bash
nvidia-smi                    # 드라이버 확인
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 패키지 설치 오류

```bash
source venv/bin/activate      # 가상환경 활성화 확인
pip install --upgrade pip
pip install -r requirements.txt
```

### Ollama 모델 다운로드

```bash
ollama pull qwen2.5:1.5b      # 1.5B 모델 (필수)
ollama pull qwen2.5:3b         # 3B 모델 (선택)
```

### HuggingFace 모델 캐시 공유 (다수 수강생 환경)

```bash
# 공유 캐시 경로 설정 (.env에 추가)
HF_HOME=/shared/huggingface_cache
```
