# LLM 파인튜닝 실습 환경 설정 가이드

## 1단계: 레포지토리 클론

```bash
git clone https://github.com/choki0715/LLM_Lecture.git multicampus
cd multicampus
```

---

## 2단계: 자동 환경 설정

```bash
bash setup.sh
```

이 스크립트가 자동으로 처리하는 것:
- Python 가상환경(venv) 생성 및 활성화
- PyTorch (CUDA 12.1) 설치
- 필수 패키지 전체 설치
- Jupyter 커널 등록
- .env 파일 생성
- GPU 점검 및 패키지 확인

---

## 3단계: API 키 발급 및 설정

### 3-1. OpenAI API Key 발급

1. https://platform.openai.com 접속
2. 회원가입 또는 로그인
3. 좌측 메뉴에서 **API keys** 클릭
4. **Create new secret key** 클릭
5. 이름 입력 후 생성
6. `sk-...` 로 시작하는 키를 복사 (이 화면을 벗어나면 다시 볼 수 없음!)

### 3-2. HuggingFace Token 발급

1. https://huggingface.co 접속
2. 회원가입 또는 로그인
3. 우측 상단 **프로필 아이콘** 클릭 → **Settings**
4. 왼쪽 메뉴에서 **Access Tokens** 클릭
5. **Create new token** 클릭
6. 이름: 아무거나 입력 (예: `lecture`)
7. Type: **Read** 선택
8. **Create token** 클릭
9. `hf_...` 로 시작하는 토큰 복사

### 3-3. .env 파일에 키 입력

프로젝트 폴더의 `.env` 파일을 열어 발급받은 키를 입력합니다:

```
OPENAI_API_KEY=sk-여기에-발급받은-키-입력
HF_TOKEN=hf_여기에-발급받은-토큰-입력
CUDA_VISIBLE_DEVICES=0
```

---

## 4단계: VS Code 커널 선택

1. VS Code에서 `.ipynb` 파일 열기
2. 우측 상단 **커널 선택** 버튼 클릭
3. **Python Environments** → `venv (Python 3.x.x)` 선택

---

## 5단계: 환경 점검

`setup_check.ipynb` 노트북을 열고 모든 셀을 실행하여 환경이 정상인지 확인합니다.

정상이면 다음 항목이 모두 통과됩니다:
- GPU 인식 및 VRAM 확인
- 필수 패키지 설치 확인
- API 키 로딩 확인
- 디스크 용량 확인

---

## 문제 해결

### GPU가 인식되지 않는 경우
```bash
nvidia-smi
```
출력이 없으면 NVIDIA 드라이버가 설치되지 않은 것입니다. 강사에게 문의하세요.

### 패키지 설치 에러
```bash
# 가상환경 활성화 후 수동 설치
source venv/bin/activate
pip install 패키지이름
```

### 디스크 용량 부족
```bash
df -h /
```
최소 20GB 여유 공간이 필요합니다.

### 커널이 보이지 않는 경우
```bash
source venv/bin/activate
pip install ipykernel
python3 -m ipykernel install --user --name venv --display-name "Python (LLM)"
```
VS Code에서 `Ctrl+Shift+P` → `Developer: Reload Window` 실행
