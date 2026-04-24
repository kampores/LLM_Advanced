
# chatbot.py - Streamlit LLM 챗봇 (Ollama 연동)
# 실행: streamlit run chatbot.py
# 사전 요구: ollama serve && ollama pull qwen2.5:1.5b

import streamlit as st
from openai import OpenAI

# ============================================
# 1. 페이지 설정
# ============================================
st.set_page_config(
    page_title="AI 챗봇",
    page_icon="🤖",
    layout="centered"
)

# ============================================
# 2. 사이드바 - 설정
# ============================================
with st.sidebar:
    st.header("⚙️ 설정")

    # API 백엔드 선택
    backend = st.selectbox(
        "API 백엔드",
        ["Ollama (로컬)", "FastAPI (커스텀)", "OpenAI"]
    )

    if backend == "Ollama (로컬)":
        base_url = "http://localhost:11434/v1"
        api_key = "ollama"
        model = st.selectbox("모델", ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"])
    elif backend == "FastAPI (커스텀)":
        base_url = st.text_input("API URL", "http://localhost:9200/v1")
        api_key = "not-needed"
        model = "local-model"
    else:
        base_url = "https://api.openai.com/v1"
        api_key = st.text_input("API Key", type="password")
        model = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])

    st.divider()

    # 모델 파라미터
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 50, 2048, 512, 50)

    # 시스템 프롬프트
    system_prompt = st.text_area(
        "시스템 프롬프트",
        value="당신은 유용하고 친절한 AI 어시스턴트입니다. 한국어로 답변합니다.",
        height=100
    )

    st.divider()

    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================
# 3. 메인 화면
# ============================================
st.title("🤖 AI 챗봇")
st.caption(f"백엔드: {backend} | 모델: {model}")

# ============================================
# 4. OpenAI 클라이언트 생성
# ============================================
client = OpenAI(base_url=base_url, api_key=api_key)

# ============================================
# 5. 세션 상태 초기화
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
# 6. 대화 히스토리 표시
# ============================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================
# 7. 사용자 입력 처리
# ============================================
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # API 요청용 메시지 구성 (시스템 프롬프트 포함)
    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(st.session_state.messages)

    # AI 응답 생성 (스트리밍)
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            response = st.write_stream(
                chunk.choices[0].delta.content
                for chunk in stream
                if chunk.choices[0].delta.content is not None
            )
        except Exception as e:
            response = f"❌ 오류: {str(e)}"
            st.error(response)

    # 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
