# 🍊 제주도 맛집 추천 대화형 AI 서비스

> 2024 빅콘테스트 생성형AI 분야 (KAIT 주최 / 신한카드 주관)

사용자의 자연어 질문을 분석해 FAISS 벡터 유사도 검색으로 최적의 제주 맛집을 찾고, Gemini LLM이 자연스러운 추천 문장을 생성하는 RAG 기반 대화형 서비스입니다.

🔗 **서비스 데모**: [shcardbigcontest2024llm-min-12.streamlit.app](https://shcardbigcontest2024llm-min-12.streamlit.app/)

---

## 개발 목적

- 신한카드 제주 가맹점 데이터를 활용한 **개인화 맛집 추천**
- 의미 기반 검색(FAISS)과 LLM을 결합한 RAG 파이프라인 구현
- Streamlit 기반의 직관적인 채팅 인터페이스 제공

---

## 작동 흐름

```
사용자 질문 입력
       ↓
질문 임베딩 생성 (Ko-SRoBERTa)
       ↓
FAISS(HNSW)로 유사 맛집 Top-10 + 관광지 Top-1 검색
       ↓
사이드바 가격대 필터 적용
       ↓
프롬프트 구성 (질문 + 검색 결과 + 추천 지침)
       ↓
Gemini 1.5 Flash 응답 생성
       ↓
Streamlit 채팅 UI 출력
```

---

## 주요 기능

**대화형 맛집 추천** — 위치, 업종, 분위기 등 자유로운 질문으로 맛집 추천을 받을 수 있으며, LLM이 자연스러운 문장으로 응답합니다.

**벡터 유사도 검색** — `jhgan/ko-sroberta-multitask` 모델로 질문을 임베딩하고, FAISS HNSW 인덱스에서 가장 유사한 맛집과 관광지를 검색합니다.

**가격대 필터링** — 신한카드 데이터의 `건당평균이용금액구간`을 기반으로 최고가부터 저가까지 필터링할 수 있습니다.

**관광지 연계 추천** — 맛집과 함께 인근 관광지 정보도 함께 제공하여 제주 여행 계획에 도움을 줍니다.

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| LLM | Google Gemini 1.5 Flash |
| 임베딩 | `jhgan/ko-sroberta-multitask` (Hugging Face) |
| 벡터 검색 | FAISS (HNSW) |
| 프레임워크 | Streamlit, PyTorch |
| 데이터 | 신한카드 제주도 가맹점 데이터 |
| 배포 | Streamlit Cloud |

---

## 파일 구성

```
├── app.py                 # Streamlit 메인 앱
├── data/
│   ├── store_data.csv     # 제주 가맹점 데이터
│   └── tourist_data.csv   # 제주 관광지 데이터
├── modules/
│   ├── faiss_index.index            # 맛집 FAISS 인덱스
│   ├── faiss_tour_index.index       # 관광지 FAISS 인덱스
│   ├── embeddings_array_file.npy    # 맛집 임베딩
│   └── embeddings_tour_array_file.npy  # 관광지 임베딩
├── requirements.txt
└── README.md
```

---

## 설치 및 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 실행
streamlit run app.py
```

> Gemini API 키를 발급받아 코드 내 `YOUR_API_KEY` 부분에 입력해야 합니다.

---

## 개발 환경

| 항목 | 내용 |
|------|------|
| Python | 3.11.10 |
| OS | Windows / Linux 호환 |
| 주요 라이브러리 | streamlit, faiss-cpu, transformers, torch, google-generativeai, pandas, numpy |

---

## 향후 확장 가능성

- 유저 선호도 학습 기반 추천 고도화
- GPS 연동 실시간 위치 기반 추천
- 리뷰 감성 분석을 결합한 맛집 품질 평가
