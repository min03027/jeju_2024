import os
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import streamlit as st
import google.generativeai as genai

# **1. 보안 설정: API 키 관리**
# **중요:** API 키를 코드에 직접 포함시키지 말고, 환경 변수나 Streamlit Secrets를 통해 관리하세요.
# 여기서는 Streamlit Secrets를 사용하는 예시를 보여드립니다.
# .streamlit/secrets.toml 파일에 다음과 같이 API 키를 추가하세요:
# GOOGLE_API_KEY = "YOUR_ACTUAL_API_KEY"

try:
    # GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key="AIzaSyAsX-SMGt5XlHc6i8TATucxPX3qCDbVyJI")
    model = genai.GenerativeModel("gemini-1.5-flash")
except KeyError:
    st.error("GOOGLE_API_KEY가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인하세요.")
    st.stop()
except Exception as e:
    st.error(f"Gemini 모델 초기화 중 오류가 발생했습니다: {e}")
    st.stop()

# **2. Streamlit 페이지 설정**
st.set_page_config(page_title="🍊참신한 제주 레스토랑!", layout="wide")
st.title("혼저 옵서예!👋")
st.subheader("군맛난 제주 밥집🧑‍🍳 추천해드릴게예")

# **3. 이미지 표시**
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTHBMuNn2EZw3PzOHnLjDg_psyp-egZXcclWbiASta57PBiKwzpW5itBNms9VFU8UwEMQ&usqp=CAU"
image_html = f"""
<div style="display: flex; justify-content: center;">
    <img src="{image_url}" alt="centered image" width="50%">
</div>
"""
st.markdown(image_html, unsafe_allow_html=True)

# **4. 데이터 로드 및 전처리**
data_path = './data'
csv_file_path = "JEJU_DATA.csv"

def load_csv(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, encoding='cp949')
            df = df[df['기준연월'] == df['기준연월'].max()].reset_index(drop=True)
            return df
        except Exception as e:
            st.error(f"CSV 파일 로드 중 오류가 발생했습니다: {e}")
            return pd.DataFrame()
    else:
        st.error(f"{file_path} 파일이 존재하지 않습니다.")
        return pd.DataFrame()

df = load_csv(os.path.join(data_path, csv_file_path))

# **5. FAISS 및 임베딩 설정**
module_path = './modules'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")
    embedding_model = AutoModel.from_pretrained("jhgan/ko-sroberta-multitask").to(device)
except Exception as e:
    st.error(f"토크나이저 또는 임베딩 모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()

def load_embeddings(file_path):
    if os.path.exists(file_path):
        try:
            return np.load(file_path)
        except Exception as e:
            st.error(f"임베딩 파일 로드 중 오류가 발생했습니다: {e}")
            return None
    else:
        st.error(f"{file_path} 파일이 존재하지 않습니다.")
        return None

embeddings = load_embeddings(os.path.join(module_path, 'embeddings_array_file.npy'))

def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    if os.path.exists(index_path):
        try:
            return faiss.read_index(index_path)
        except Exception as e:
            st.error(f"FAISS 인덱스 로드 중 오류가 발생했습니다: {e}")
            return None
    else:
        st.error(f"{index_path} 파일이 존재하지 않습니다.")
        return None

faiss_index = load_faiss_index()

# **6. 텍스트 임베딩 함수**
def embed_text(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.squeeze().cpu().numpy()
    except Exception as e:
        st.error(f"텍스트 임베딩 중 오류가 발생했습니다: {e}")
        return None

# **7. 응답 생성 함수**
def generate_response_with_faiss(question, df, embeddings, faiss_index, model, embed_text, k=3):
    if embeddings is None or faiss_index is None:
        return "임베딩 파일 또는 FAISS 인덱스가 로드되지 않았습니다."

    try:
        query_embedding = embed_text(question).reshape(1, -1)
        distances, indices = faiss_index.search(query_embedding, k * 3)
        filtered_df = df.iloc[indices[0, :]].copy().reset_index(drop=True).head(k)

        if filtered_df.empty:
            return "질문과 일치하는 가게가 없습니다."

        reference_info = "\n".join(filtered_df['text'])
        prompt = (
            f"질문: {question}\n"
            f"대답해줄 때 업종별로 가능하면 하나씩 추천해줘. "
            f"그리고 추가적으로 오래된 맛집과 새로운 맛집을 각각 추천해줘.\n"
            f"참고할 정보: {reference_info}\n응답:"
        )
        response = model.generate_content(prompt)
        
        # 응답 객체에서 텍스트 추출
        extracted_text = response.candidates[0].content.parts[0].text
        return extracted_text
    except Exception as e:
        return f"응답 생성 중 오류가 발생했습니다: {e}"

# **8. 대화 기록 저장 및 로드 기능**
history_path = os.path.join(module_path, 'conversation_history.json')

def save_conversation_history(conversations):
    try:
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=4)
        st.success("대화 기록이 저장되었습니다.")
    except Exception as e:
        st.error(f"대화 기록 저장 중 오류가 발생했습니다: {e}")

def load_conversation_history():
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"대화 기록 로드 중 오류가 발생했습니다: {e}")
            return []
    return []

def initialize_conversation(conv_id=None):
    # conv_id가 제공되면 "대화 세션 {conv_id}"로 제목 설정
    title = f"대화 세션 {conv_id}" if conv_id is not None else ""
    return {
        "id": conv_id,  # UUID 대신 숫자 ID 사용
        "title": title,
        "messages": []
    }

# **10. 대화 세션 상태 초기화**
if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversation_history()
    if not st.session_state.conversations:
        # 대화 세션을 1로 설정
        initial_conversation = initialize_conversation(1)
        st.session_state.conversations.append(initial_conversation)
    else:
        # 기존 대화 세션 수에 따라 새 세션 제목 설정
        next_id = len(st.session_state.conversations) + 1
        initial_conversation = initialize_conversation(next_id)
        st.session_state.conversations.append(initial_conversation)
    st.session_state.current_conversation = st.session_state.conversations[-1]

# **11. 사이드바 유지 및 대화 저장 기능 추가**
with st.sidebar:
    st.header("💬 대화 기록 관리")
    
    # 새로운 대화 시작 버튼
    if st.sidebar.button("새로운 대화 시작"):
        # 다음 세션 ID를 설정
        next_id = len(st.session_state.conversations) + 1
        new_conversation = initialize_conversation(next_id)
        st.session_state.conversations.append(new_conversation)
        st.session_state.current_conversation = new_conversation
    
    # 대화 세션 선택
    if st.session_state.conversations:
        conversation_titles = [f"{conv['title']}" for conv in st.session_state.conversations]
        selected_conversation_title = st.sidebar.selectbox("대화 세션 선택", conversation_titles)
    
        # 선택된 대화 세션 로드
        selected_conversation = next(
            (conv for conv in st.session_state.conversations if conv['title'] == selected_conversation_title), 
            None
        )
    
        if selected_conversation:
            st.session_state.current_conversation = selected_conversation
        else:
            st.error("선택한 대화 세션이 없습니다.")
    
    # 채팅 내역 초기화 버튼
    def clear_chat_history():
        if st.session_state.current_conversation:
            st.session_state.current_conversation["messages"] = []
            # 초기 제목으로 "대화 세션 {id}"로 재설정
            st.session_state.current_conversation["title"] = f"대화 세션 {st.session_state.current_conversation['id']}"
        st.success("채팅 내역이 초기화되었습니다.")
    
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    
    # 대화 저장 버튼
    if st.sidebar.button("대화 저장"):
        save_conversation_history(st.session_state.conversations)

# **12. 채팅 메시지 표시**
for message in st.session_state.current_conversation["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# **13. 채팅 입력 및 응답 처리**
chat_input = st.chat_input("질문을 입력하세요:")
if chat_input:
    current_conv = st.session_state.current_conversation

    # 첫 번째 사용자 메시지일 경우, 제목 설정 (질문 내용으로 변경)
    if current_conv["title"].startswith("대화 세션") and len(current_conv["messages"]) == 0:
        current_conv["title"] = (chat_input[:15] + "...") if len(chat_input) > 15 else chat_input

    # 사용자 메시지 추가
    user_message = {"role": "user", "content": chat_input, "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    current_conv["messages"].append(user_message)
    st.markdown(f"**사용자:** {chat_input}")

    # 어시스턴트 응답 생성
    response = generate_response_with_faiss(chat_input, df, embeddings, faiss_index, model, embed_text)
    assistant_message = {"role": "assistant", "content": response, "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    current_conv["messages"].append(assistant_message)
    st.markdown(f"**어시스턴트:** {response}")

    # 대화 기록 저장
    save_conversation_history(st.session_state.conversations)