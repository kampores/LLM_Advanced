
# chat_basic.py - Streamlit 기본 채팅 앱
# 실행: streamlit run chat_basic.py

import streamlit as st

# ============================================
# 1. 페이지 설정
# ============================================
st.set_page_config(
    page_title="LLM 챗봇",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 LLM 챗봇")
st.caption("파인튜닝된 모델과 대화해보세요!")

# ============================================
# 2. 세션 상태 초기화 (대화 히스토리)
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
# 3. 기존 대화 히스토리 표시
# ============================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================
# 4. 사용자 입력 처리
# ============================================
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 히스토리에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI 응답 생성 (여기서는 에코 - 나중에 실제 모델 연동)
    response = f"에코: {prompt}"  # 임시 응답

    # AI 응답 표시
    with st.chat_message("assistant"):
        st.markdown(response)

    # 히스토리에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
