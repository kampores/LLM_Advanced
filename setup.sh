#!/bin/bash
# ===========================================
# LLM 파인튜닝 실습 환경 자동 설정 스크립트
# 대상: 깡통 Ubuntu (아무것도 없는 상태)
# 사용법: bash setup.sh
# ===========================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
NC='\033[0m'

TOTAL_STEPS=13
CURRENT_STEP=0

print_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    local pct=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    local filled=$((pct / 5))
    local empty=$((20 - filled))
    local bar=""
    if [ "$filled" -gt 0 ]; then
        bar=$(printf '%0.s#' $(seq 1 "$filled"))
    fi
    if [ "$empty" -gt 0 ]; then
        bar="${bar}$(printf '%0.s-' $(seq 1 "$empty"))"
    fi
    echo ""
    echo -e "${CYAN}[${CURRENT_STEP}/${TOTAL_STEPS}]${NC} ${GREEN}$1${NC}"
    echo -e "${BLUE}  [${bar}] ${pct}%${NC}"
}
print_warn() { echo -e "  ${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "  ${RED}[ERROR]${NC} $1"; }
print_ok() { echo -e "  ${GREEN}[OK]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}   LLM 파인튜닝 실습 환경 설정${NC}"
echo -e "${CYAN}   총 ${TOTAL_STEPS}단계를 진행합니다${NC}"
echo -e "${CYAN}==========================================${NC}"

# ----- 1. 시스템 패키지 설치 -----
print_step "시스템 패키지 설치 (sudo 필요)"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-venv python3-pip python3-dev \
    build-essential cmake \
    libaio-dev \
    curl wget git \
    fonts-nanum fonts-nanum-coding \
    > /dev/null 2>&1
# matplotlib 폰트 캐시 갱신 (한글 그래프용)
fc-cache -fv > /dev/null 2>&1 || true
print_ok "시스템 패키지 설치 완료 (한글 폰트 포함)"

# ----- 2. NVIDIA 드라이버 확인 -----
print_step "NVIDIA 드라이버 확인"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    print_ok "드라이버 ${DRIVER_VER} / GPU: ${GPU_NAME}"
else
    print_warn "NVIDIA 드라이버가 없습니다. GPU 학습이 불가능합니다."
    print_warn "설치: sudo apt install nvidia-driver-550  (이후 재부팅 필요)"
fi

# ----- 3. Python 확인 -----
print_step "Python 버전 확인"
PY_VERSION=$(python3 --version 2>&1)
print_ok "$PY_VERSION"

# ----- 4. 가상환경 생성 -----
print_step "가상환경 생성 (venv)"
if [ -d "venv" ]; then
    print_warn "venv가 이미 존재합니다. 기존 환경을 사용합니다."
else
    python3 -m venv venv
    print_ok "venv 생성 완료"
fi

# ----- 5. 가상환경 활성화 + pip 업그레이드 -----
print_step "가상환경 활성화 + pip 업그레이드"
source venv/bin/activate
pip install --upgrade pip -q
print_ok "활성화 완료 ($(which python3))"

# ----- 6. PyTorch 설치 (CUDA) -----
print_step "PyTorch 설치 (CUDA 12.8) - 시간이 걸릴 수 있습니다"
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q
print_ok "PyTorch 설치 완료"

# ----- 7. 필수 패키지 설치 (버전 고정) -----
print_step "필수 패키지 설치 - 시간이 걸릴 수 있습니다"

# 7-1. 핵심 ML 패키지 (pin)
pip install \
    "transformers==4.57.2" \
    "accelerate==1.13.0" \
    "datasets==4.8.4" \
    "peft==0.18.1" \
    "trl==0.23.0" \
    "bitsandbytes==0.49.2" \
    "huggingface-hub==0.36.2" \
    -q

# 7-2. LangChain / RAG
pip install \
    "langchain==0.3.28" \
    "langchain-openai==0.3.35" \
    "langchain-community==0.3.31" \
    "langchain-ollama==0.3.10" \
    "langchain-chroma==0.2.6" \
    "chromadb==1.5.7" \
    "sentence-transformers==5.4.0" \
    "rank_bm25==0.2.2" \
    -q

# 7-3. OpenAI / 토큰화 / 데이터
pip install \
    "openai==2.31.0" \
    "tiktoken==0.12.0" \
    "pandas==3.0.2" \
    "numpy==2.2.6" \
    "python-dotenv==1.2.2" \
    "tqdm==4.67.3" \
    -q

# 7-4. 평가 라이브러리
pip install \
    "ragas" \
    "nltk==3.9.4" \
    "rouge-score==0.1.2" \
    "bert-score==0.3.13" \
    "evaluate>=0.4.0" \
    "scikit-learn>=1.3.0" \
    -q

# 7-5. 시각화 / UI
pip install \
    "matplotlib==3.10.8" \
    "streamlit==1.56.0" \
    -q

# 7-6. API 서버
pip install \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.27.0" \
    "pydantic>=2.0.0" \
    "requests>=2.31.0" \
    -q

# 7-7. 서빙 / 고급
pip install "vllm" -q
pip install "deepspeed==0.18.9" -q
pip install "llama-cpp-python==0.3.20" -q

# 7-8. Unsloth (의존성 충돌 많음 — 마지막에 느슨하게 설치)
pip install "unsloth" --no-deps -q || print_warn "Unsloth 설치 실패 - 21 노트북만 영향 (수동 설치: pip install unsloth)"

print_ok "패키지 설치 완료"

# ----- 8. Jupyter 커널 등록 -----
print_step "Jupyter 커널 등록"
pip install ipykernel -q
python3 -m ipykernel install --user --name venv --display-name "Python (LLM)"
print_ok "커널 'Python (LLM)' 등록 완료"

# ----- 9. Ollama 설치 -----
print_step "Ollama 설치"
if command -v ollama &> /dev/null; then
    OLLAMA_VER=$(ollama --version 2>&1)
    print_warn "Ollama가 이미 설치되어 있습니다: $OLLAMA_VER"
else
    curl -fsSL https://ollama.ai/install.sh | sh
    print_ok "Ollama 설치 완료"
fi

# ----- 10. .env 파일 생성 -----
print_step ".env 파일 확인"
if [ -f ".env" ]; then
    print_warn ".env 파일이 이미 존재합니다. 건너뜁니다."
else
    cp .env.example .env
    print_ok ".env 파일 생성 완료 — API 키를 입력해주세요"
fi

# ----- 11. GPU 점검 -----
print_step "GPU 점검"
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU: {name}')
    print(f'  VRAM: {vram:.1f} GB')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  GPU를 찾을 수 없습니다!')
"
print_ok "GPU 점검 완료"

# ----- 12. 설치 확인 -----
print_step "설치된 주요 패키지 확인"
python3 -c "
import importlib
pkgs = ['torch','transformers','peft','trl','datasets','accelerate','bitsandbytes',
        'langchain','openai','chromadb','tiktoken','streamlit','vllm']
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, '__version__', 'OK')
        print(f'  {p:20s} {v}')
    except ImportError:
        print(f'  {p:20s} NOT INSTALLED')
"
print_ok "패키지 확인 완료"

# ----- 13. VS Code 설치 -----
print_step "VS Code 설치"
if command -v code &> /dev/null; then
    print_ok "VS Code가 이미 설치되어 있습니다."
else
    print_ok "VS Code 설치 중..."
    wget -qO /tmp/vscode.deb "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64"
    sudo dpkg -i /tmp/vscode.deb || sudo apt-get install -f -y -qq > /dev/null 2>&1
    rm -f /tmp/vscode.deb
    print_ok "VS Code 설치 완료"
fi

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  [####################] 100% 설정 완료!${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo "  다음 단계:"
echo "  1. .env 파일에 API 키 입력"
echo "     - OPENAI_API_KEY=sk-..."
echo "     - HF_TOKEN=hf_..."
echo "  2. VS Code에서 커널 'Python (LLM)' 선택"
echo "  3. setup_check.ipynb 실행하여 최종 점검"
echo ""
