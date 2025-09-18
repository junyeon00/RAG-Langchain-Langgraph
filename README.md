📘 LangChain + ClovaX 기반 RAG 실습

이 프로젝트는 네이버 ClovaX LLM과 LangChain을 활용하여,
PDF 문서를 기반으로 검색 및 답변을 생성하는 RAG(Retrieval-Augmented Generation) 구조를 실습한 코드입니다.

동일한 기능을 두 가지 방식으로 구현했습니다:

LangChain 기반 버전 → 단순 Retriever + LLM 연결

LangGraph 기반 버전 → StateGraph를 활용한 확장형 워크플로우

🚀 주요 기능
✅ 공통

PDF 파일 읽기 및 파싱 (PyPDF2)

문서 분할 (Chunking) → RecursiveCharacterTextSplitter

문서 임베딩 → ClovaXEmbeddings

벡터 저장소 → InMemoryVectorStore

검색 모듈 → Retriever (.as_retriever())

LangChain Tool → create_retriever_tool로 retriever를 툴로 래핑

ClovaX LLM (ChatClovaX)과 결합해 문서 기반 질의응답 구현

📂 버전별 차이
1. LangChain 버전 (ransom_langchain.py)

Retriever Tool을 직접 LLM에 바인딩하여 질의응답 수행

흐름:

사용자 질문 → LLM → (필요 시 retriever tool 호출) → 답변 반환


단순하고 직관적인 구조

작은 실습 및 프로토타이핑에 적합

2. LangGraph 버전 (ransom_langgraph.py)

StateGraph를 활용하여 노드 기반 워크플로우 구성

llm_node: LLM이 tool call 여부 판단

tool_node: retriever 실행 및 결과 반환

조건부 edge를 통해 LLM → Retriever → 최종 답변의 루프 제어

더 복잡한 에이전트 스타일 동작 확장이 가능

상태(state)를 기반으로 하여 대화 기록을 더 정교하게 관리 가능
