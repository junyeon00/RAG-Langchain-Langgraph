# 📘 LangChain + ClovaX 기반 RAG 실습

이 프로젝트는 네이버 ClovaX LLM과 LangChain을 활용하여, **PDF 문서를 기반으로 검색 및 답변을 생성하는 RAG(Retrieval-Augmented Generation)** 구조를 실습한 코드입니다.

---

## ⚙️ 구현 방식

동일한 기능을 두 가지 방식으로 구현했습니다:

| 버전              | 설명                                                                 |
|-------------------|----------------------------------------------------------------------|
| **LangChain 기반** | 단순 Retriever + LLM 연결 구조                                       |
| **LangGraph 기반** | StateGraph를 활용한 노드 기반 확장형 워크플로우                    |

---

## 🚀 주요 기능 (공통)

- 📄 **PDF 파일 읽기 및 파싱**: `PyPDF2` 사용
- ✂️ **문서 분할 (Chunking)**: `RecursiveCharacterTextSplitter`
- 🧠 **문서 임베딩**: `ClovaXEmbeddings` 사용
- 📦 **벡터 저장소**: `InMemoryVectorStore` 기반
- 🔍 **검색 모듈**: `.as_retriever()`로 retriever 구성
- 🔧 **LangChain Tool**: `create_retriever_tool()`로 retriever 툴화
- 💬 **ClovaX LLM** (`ChatClovaX`)과 결합해 문서 기반 질의응답 구현

---

## 📂 버전별 차이

### 1️⃣ LangChain 버전 (`ransom_langchain.py`)

- **구조**: Retriever Tool을 직접 LLM에 바인딩하여 질의응답 수행
- **흐름**:  
  사용자 질문 → LLM → *(필요 시)* Retriever Tool 호출 → 답변 반환  
- **특징**:  
  - 단순하고 직관적인 구조  
  - 빠르게 실습/프로토타이핑할 때 적합  

---

### 2️⃣ LangGraph 버전 (`ransom_langgraph.py`)

- **구조**: `StateGraph`를 활용하여 노드 기반 워크플로우 구성
- **구성 요소**:
  - `llm_node`: LLM이 tool 호출 여부 판단
  - `tool_node`: Retriever 실행 및 결과 반환
- **흐름 제어**:  
  조건부 edge를 통해 **LLM → Retriever → LLM**의 루프 구현
- **특징**:
  - 복잡한 에이전트 스타일 동작 확장 가능  
  - 대화 상태(`state`) 기반으로 정교한 기록 관리 가능  

---

## ✅ 흐름도

### 1️⃣ LangChain 버전
```mermaid
graph TD
    A[사용자 입력] --> B[LLM 호출 (ChatClovaX)]
    B --> C{Retriever Tool 필요?}
    C -- Yes --> D[Tool 호출 (Retriever 실행)]
    D --> E[Tool 결과를 LLM에 전달]
    E --> F[최종 응답 생성]
    C -- No --> F[LLM이 직접 응답]

위 흐름은 LangChain에서 Retriever Tool을 사용하는 단순 구조를 보여줍니다.

### 1️⃣ LangGraph 버전
```mermaid
graph TD
    A[사용자 입력] --> B[llm_node: LLM Tool 판단]
    B --> C{Tool 호출?}
    C -- Yes --> D[tool_node: Retriever 실행]
    D --> E[retrieved docs 반환]
    E --> B
    C -- No --> F[응답 생성 및 종료]

LangGraph 흐름은 상태 노드 기반의 유연한 반복 구조를 제공합니다.
---
