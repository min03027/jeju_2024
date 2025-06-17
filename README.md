
# 🧭 제주도 맛집 추천 시스템

Gemini API와 FAISS를 활용한 **상황 맞춤형 제주 맛집 추천 서비스**입니다.  
사용자가 입력한 질문을 분석해 가장 유사한 가게와 관광지를 찾고, LLM을 통해 자연스러운 추천 결과를 제공합니다.

---

## 💡 개발 목적

- 사용자 질문에 맞춘 **개인화된 제주 맛집 추천**  
- 의미 기반 검색(벡터 유사도)과 LLM을 결합한 추천 방식 구현  
- 직관적인 웹 기반 인터페이스 제공 (Streamlit 사용)

---

## 🔧 주요 기능

### 📌 대화형 맛집 추천
- 사용자가 질문을 입력하면 **LLM이 자연스럽게 추천 문장을 생성**
- 가격대, 위치, 분위기 등 다양한 조건을 반영

### 📌 유사도 기반 맛집 탐색
- Ko-SRoBERTa로 질문을 벡터화
- FAISS(HNSW) 방식으로 **가장 유사한 맛집/관광지 11개 검색**

### 📌 필터 기능
- 데이터 내 `건당평균이용금액구간`을 바탕으로 **가격대 필터** 적용  
  (고가, 저가 등)

### 📌 간편한 웹 UI
- Streamlit 기반 채팅 인터페이스
- 사이드바에서 **이모지 기반 가격대 선택** 가능

---

## 🛠 개발 환경

| 항목 | 내용 |
|------|------|
| 개발 언어 | Python 3.11.10 |
| 주요 라이브러리 | `streamlit`, `faiss-cpu`, `transformers`, `torch`, `google-generativeai`, `pandas`, `numpy` |
| 임베딩 모델 | Ko-SRoBERTa |
| LLM | Gemini 1.5 Flash |
| 운영 체제 | Windows / Linux 호환 |

---

## ⚙️ 설치 및 실행 방법

### 1️⃣ 라이브러리 설치

```bash
pip install streamlit faiss-cpu transformers torch pandas numpy google-generativeai
```

### 2️⃣ 실행

```bash
streamlit run app.py
```

---

## 🧠 작동 흐름

```plaintext
사용자 질문 입력
       ↓
질문 의미 분석 (Ko-SRoBERTa)
       ↓
FAISS로 유사 맛집/관광지 검색
       ↓
LLM 프롬프트 구성 및 답변 생성 (Gemini)
       ↓
Streamlit UI로 결과 출력
```

---

## 📷 결과 예시

<img src="예시스크린샷.png" width="600"/>

---

## 📝 파일 구성

| 파일명 | 설명 |
|--------|------|
| `app.py` | 메인 실행 파일 (Streamlit 앱) |
| `store_data.csv` | 제주 지역 가맹점 데이터 |
| `tourist_data.csv` | 제주 관광지 데이터 |
| `README.md` | 프로젝트 설명 파일 |

---

## 💬 향후 확장 가능성

- 유저 선호도 학습 기반 추천 고도화  
- 위치 기반 실시간 추천 (GPS 연동)  
- 리뷰 감성 분석과 결합한 품질 평가 기능 추가

---

## 📌 기타

- 이 프로젝트는 개인 포트폴리오용으로 제작되었습니다.
- Gemini API 키는 사용자가 직접 발급하여 입력해야 작동합니다.
