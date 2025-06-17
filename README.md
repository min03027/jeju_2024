# KAIT 주최 및 신한카드 주관 - 2024 빅콘테스트 생성형AI분야 
제주도 맛집 추천 시스템 (6점)
1. 개발 목적
사용자 맞춤형 제주 맛집을 대화형으로 추천

가격대, 위치, 업종 등 상황에 맞는 개인화된 추천 제공

대용량 데이터 속 유사한 맛집/관광지를 빠르게 탐색하고 LLM을 통해 자연스러운 추천 응답 생성

직관적이고 인터랙티브한 웹 UI 제공

2. 구체적 구현 내용
1. GUI 구성
Streamlit을 활용해 웹 기반 인터페이스 구성

사용자는 사이드바에서 **희망 가격대(이모지 포함)**를 선택

중앙에는 질문 입력 및 응답이 표시되는 채팅 인터페이스 제공

맛집 정보, 추천 결과, 이미지 등 시각적으로 구성

2. 추천 로직 흐름
사용자가 질문을 입력하면, Ko-SRoBERTa 임베딩 모델을 통해 의미를 벡터화

**FAISS(HNSW 방식)**를 통해 유사한 맛집 정보 10건과 관광지 정보 1건을 빠르게 탐색

유사도가 높은 데이터를 기반으로 프롬프트 구성 후, Gemini 1.5 Flash 모델에 입력

LLM이 추천 응답을 자연스럽게 생성해 사용자에게 제공

3. 개인화 필터링 기능
데이터 내 건당평균이용금액구간을 기반으로 가격대 필터링 수행
(예: 고가, 저가, 중저가 등 구간 코드 기준 자동 필터)

사용자의 입력에 따라 적절한 가맹점, 관광지 텍스트 정보를 프롬프트에 포함

4. 응답 생성 및 대화관리
Streamlit 세션을 통해 대화 내역을 저장/초기화 가능

채팅 입력마다 LLM 응답을 실시간으로 생성하여 UI에 출력

3. 개발환경 명시
개발 언어: Python 3.11.10

사용한 라이브러리:

Streamlit (웹 GUI)

transformers (Ko-SRoBERTa 임베딩)

torch, faiss-cpu, pandas, numpy

google.generativeai (Gemini API 활용)

운영체제: Windows / Linux 호환

4. 터미널 명령어
📦 라이브러리 설치
bash
복사
편집
pip install streamlit faiss-cpu transformers torch pandas numpy google-generativeai
▶️ 프로그램 실행
bash
복사
편집
streamlit run app.py
