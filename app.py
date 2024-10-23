from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss
from datetime import datetime
import streamlit as st
import google.generativeai as genai

# 경로 설정
data_path = './data'
@@ -75,7 +75,7 @@

# 텍스트 임베딩 함수
def embed_text(text):
    if not text:  # 텍스트가 비어있으면 오류 발생
        raise ValueError("Input text cannot be empty.")
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
@@ -85,19 +85,26 @@
# 임베딩 로드
embeddings = np.load(os.path.join(module_path, 'embeddings_array_file.npy'))

# 응답 생성 함수
def generate_response_with_faiss(question, df, embeddings, model, embed_text, time, local_choice, index_path=os.path.join(module_path, 'faiss_index.index'), max_count=10, k=3):
    # FAISS 인덱스 로드
    index = load_faiss_index(index_path)

    # 쿼리 임베딩 생성
    query_embedding = embed_text(question).reshape(1, -1)
    # 가장 유사한 텍스트 검색
    distances, indices = index.search(query_embedding, k*3)
    filtered_df = df.iloc[indices[0, :]].copy().reset_index(drop=True)

    # 영업시간 필터링
    if time == '아침':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(5, 12)))].reset_index(drop=True)
    elif time == '점심':
@@ -109,43 +116,41 @@
    elif time == '밤':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in [23, 24, 1, 2, 3, 4]))].reset_index(drop=True)

    # 개설일자 필터링
    current_year = datetime.now().year
    if opening_date_condition == "오래된 맛집":
        filtered_df = filtered_df[filtered_df['가맹점개설일자'].apply(lambda x: 2024 - int(str(x)[:4]) >= 20)]
    elif opening_date_condition == "요즘 뜨는 곳":
        filtered_df = filtered_df[filtered_df['가맹점개설일자'].apply(lambda x: 2024 - int(str(x)[:4]) <= 5)]

    # 필터링된 데이터가 없을 때 처리
    if filtered_df.empty:
        return f"선택하신 조건({opening_date_condition})에 맞는 가게가 없습니다."

    filtered_df = filtered_df.reset_index(drop=True).head(k)
    # 현지인 맛집 옵션
    if local_choice == '제주도민 맛집':
        local_choice = '제주도민(현지인) 맛집'
    elif local_choice == '관광객 맛집':
        local_choice = '현지인 비중이 낮은 관광객 맛집'

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
    st.session_state.messages.append({"role": "assistant", "content": response})
