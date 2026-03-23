#!/bin/bash
# ===========================================
# LLM 파인튜닝 실습 환경 자동 설정 스크립트
# 사용법: bash setup.sh
# ===========================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
NC='\033[0m'

TOTAL_STEPS=11
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

# ----- 1. Python 확인 -----
print_step "Python 버전 확인"
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    print_ok "$PY_VERSION"
else
    print_error "python3이 설치되어 있지 않습니다."
    exit 1
fi

# ----- 2. 가상환경 생성 -----
print_step "가상환경 생성 (venv)"
if [ -d "venv" ]; then
    print_warn "venv가 이미 존재합니다. 기존 환경을 사용합니다."
else
    python3 -m venv venv
    print_ok "venv 생성 완료"
fi

# ----- 3. 가상환경 활성화 -----
print_step "가상환경 활성화"
source venv/bin/activate
print_ok "활성화 완료 ($(which python3))"

# ----- 4. pip 업그레이드 -----
print_step "pip 업그레이드"
pip install --upgrade pip -q
print_ok "pip 최신 버전"

# ----- 5. PyTorch 설치 (CUDA) -----
print_step "PyTorch 설치 (CUDA 12.1) - 시간이 걸릴 수 있습니다"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
print_ok "PyTorch 설치 완료"

# ----- 6. 필수 패키지 설치 -----
print_step "필수 패키지 설치 - 시간이 걸릴 수 있습니다"
pip install \
    "transformers>=4.40.0" \
    "accelerate>=0.28.0" \
    "datasets>=2.18.0" \
    "peft>=0.10.0" \
    "trl>=0.8.0" \
    "bitsandbytes>=0.43.0" \
    "langchain>=0.3.0,<1.0.0" \
    "langchain-openai>=0.1.0" \
    "langchain-community>=0.0.20,<1.0.0" \
    "langchain-ollama>=0.1.0" \
    "langchain-chroma>=0.1.0" \
    "chromadb>=0.4.0" \
    "sentence-transformers>=2.6.0" \
    "openai>=1.12.0" \
    "tiktoken>=0.6.0" \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "ragas>=0.1.0" \
    "matplotlib>=3.7.0" \
    "streamlit>=1.32.0" \
    "python-dotenv>=1.0.0" \
    "tqdm>=4.66.0" \
    "huggingface-hub>=0.22.0" \
    "vllm" \
    -q
print_ok "패키지 설치 완료"

# ----- 7. Jupyter 커널 등록 -----
print_step "Jupyter 커널 등록"
pip install ipykernel
python3 -m ipykernel install --user --name venv --display-name "Python (LLM)"
print_ok "커널 'Python (LLM)' 등록 완료"

# ----- 8. Ollama 설치 -----
print_step "Ollama 설치"
if command -v ollama &> /dev/null; then
    OLLAMA_VER=$(ollama --version 2>&1)
    print_warn "Ollama가 이미 설치되어 있습니다: $OLLAMA_VER"
else
    print_ok "Ollama 설치 중..."
    curl -fsSL https://ollama.ai/install.sh | sh
    print_ok "Ollama 설치 완료"
fi

# ----- 9. .env 파일 생성 -----
print_step ".env 파일 확인"
if [ -f ".env" ]; then
    print_warn ".env 파일이 이미 존재합니다. 건너뜁니다."
else
    cp .env.example .env
    print_ok ".env 파일 생성 완료 — API 키를 입력해주세요"
fi

# ----- 10. GPU 점검 -----
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

# ----- 11. 설치 확인 -----
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
