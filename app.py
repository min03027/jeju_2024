import os
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from datetime import datetime
import streamlit as st
import google.generativeai as genai
import random

# 경로 설정
data_path = './data'
module_path = './modules'

genai.configure(api_key="AIzaSyAsX-SMGt5XlHc6i8TATucxPX3qCDbVyJI")
model = genai.GenerativeModel("gemini-1.5-flash")

# CSV 파일 로드
csv_file_path = "JEJU_MCT_DATA_modified.csv"
df = pd.read_csv(os.path.join(data_path, csv_file_path))

# 최신연월 데이터만 가져옴
df = df[df['기준연월'] == df['기준연월'].max()].reset_index(drop=True)

# Streamlit App UI
st.set_page_config(page_title="🍊참신한 제주 맛집!")

# 사이드바 설정
with st.sidebar:
    st.title("🍊참신한! 제주 맛집")
    st.subheader("언드레 가신디가?")
    time = st.sidebar.selectbox("", ["아침", "점심", "오후", "저녁", "밤"], key="time")
    opening_date_condition = st.sidebar.selectbox("", ["오래된 맛집", "요즘 뜨는 곳"], key="Opening_date", label_visibility="hidden")
    local_choice = st.radio('', ('제주도민 맛집', '관광객 맛집'))

# 타이틀 및 설명
st.title("혼저 옵서예!👋")
st.subheader("군맛난 제주 밥집🧑‍🍳 추천해드릴게예")
st.write("#흑돼지 #갈치조림 #옥돔구이 #고사리해장국 #전복뚝배기 #한치물회 #빙떡 #오메기떡..🤤")

# 이미지 출력
image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTHBMuNn2EZw3PzOHnLjDg_psyp-egZXcclWbiASta57PBiKwzpW5itBNms9VFU8UwEMQ&usqp=CAU"
st.markdown(f"<div style='display: flex; justify-content: center;'><img src='{image_path}' alt='centered image' width='50%'></div>", unsafe_allow_html=True)

# 채팅 메시지 저장
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}]

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index
    else:
        raise FileNotFoundError(f"{index_path} 파일이 존재하지 않습니다.")

# 텍스트 임베딩 함수
def embed_text(text):
    if not text:
        raise ValueError("Input text cannot be empty.")
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# 임베딩 로드
embeddings = np.load(os.path.join(module_path, 'embeddings_array_file.npy'))

# 추천할 맛집 목록
suggested_restaurants = [
    "제주도 흑돼지 맛집", 
    "갈치조림 전문점", 
    "옥돔구이 맛집", 
    "고사리해장국 집", 
    "전복뚝배기 전문점", 
    "한치물회 집", 
    "빙떡 카페", 
    "오메기떡 가게"
]

# 조건에 따른 필터링 함수
def filter_restaurants(df, time, opening_date_condition):
    filtered_df = df

    # 영업시간 조건
    if time == '아침':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(5, 12)))].reset_index(drop=True)
    elif time == '점심':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(12, 14)))].reset_index(drop=True)
    elif time == '오후':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(14, 18)))].reset_index(drop=True)
    elif time == '저녁':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(18, 23)))].reset_index(drop=True)
    elif time == '밤':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in [23, 24, 1, 2, 3, 4]))].reset_index(drop=True)

    # 개설일자 필터링
    if opening_date_condition == "오래된 맛집":
        filtered_df = filtered_df[filtered_df['가맹점개설일자'].apply(lambda x: 2024 - int(str(x)[:4]) >= 20)]
    elif opening_date_condition == "요즘 뜨는 곳":
        filtered_df = filtered_df[filtered_df['가맹점개설일자'].apply(lambda x: 2024 - int(str(x)[:4]) <= 5)]

    return filtered_df

# 응답 생성 함수
def generate_response_with_faiss(question, df, embeddings, model, embed_text, time, local_choice):
    index = load_faiss_index()

    query_embedding = embed_text(question).reshape(1, -1)
    distances, indices = index.search(query_embedding, 9)  # 최대 9개 검색
    filtered_df = df.iloc[indices[0, :]].copy().reset_index(drop=True)

    # 필터링
    filtered_df = filter_restaurants(filtered_df, time, opening_date_condition)

    # 결과가 없을 경우 조건을 하나씩 제외하며 재추천
    if filtered_df.empty:
        if opening_date_condition == "오래된 맛집":
            # 10년으로 변경하여 재필터링
            filtered_df = df[df['가맹점개설일자'].apply(lambda x: 2024 - int(str(x)[:4]) >= 10)]
        else:
            conditions = ["오래된 맛집", "요즘 뜨는 곳"]
            for condition in conditions:
                # 현재 조건을 제외한 필터링
                temp_df = filter_restaurants(df, time, condition if condition != opening_date_condition else conditions[0])
                if not temp_df.empty:
                    filtered_df = temp_df
                    break

    # 결과가 없으면 추천
    if filtered_df.empty:
        suggested = random.choice(suggested_restaurants)
        return f"선택하신 조건에 맞는 가게가 없습니다. 대신, '{suggested}'를 추천드립니다!"

    # 응답 생성
    reference_info = "\n".join(filtered_df['text'].tolist())
    prompt = f"질문: {question} 특히 {local_choice}을 선호해\n참고할 정보:\n{reference_info}\n응답:"

    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else str(response)

# 사용자 입력 처리
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# 응답 생성
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response_with_faiss(prompt, df, embeddings, model, embed_text, time, local_choice)
            st.markdown(response)
    st

