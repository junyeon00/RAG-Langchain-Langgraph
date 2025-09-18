import os
import time
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_naver import ChatClovaX, ClovaXEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, END

# 환경 변수 로드
load_dotenv()


# 1) PDF 문서 읽기
pdf_path = "**insert document path**"
reader = PdfReader(pdf_path)

documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        documents.append(
            Document(page_content=text, metadata={"source": pdf_path, "page": i + 1})
        )

print(f"원본 문서 수: {len(documents)}")

# 2) 문서 분할 (Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
split_docs = text_splitter.split_documents(documents)
print(f"쪼갠 문서 수: {len(split_docs)}")


# 3) 클로바 임베딩 모델
embeddings = ClovaXEmbeddings(model="clir-emb-dolphin")

# 4) InMemory 벡터스토어 구축
vectorstore = InMemoryVectorStore(embedding=embeddings)

batch_size = 2
for i in range(0, len(split_docs), batch_size):
    batch = split_docs[i : i + batch_size]
    vectorstore.add_documents(batch)
    print(f"{i+len(batch)}/{len(split_docs)} 완료")
    time.sleep(1.0)  # RateLimit 방지

# 5) Retriever & Tool
retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_research",
    description="랜섬웨어 관련 PDF를 기반으로 검색하는 툴",
)

# 6) LLM 초기화 (ClovaX)
chat = ChatClovaX(
    model="HCX-005",
    api_key=os.getenv("CLOVASTUDIO_API_KEY"),
)

# 7) 노드 정의
# =====================================
# LLM 노드: 질문을 받아 retriever 호출 여부 판단
def llm_node(state: MessagesState):
    response = chat.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# Retriever 실행 노드
def tool_node(state: MessagesState):
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    for call in tool_calls:
        if call["name"] == "pdf_research":
            docs_and_scores = vectorstore.similarity_search_with_score(
                call["args"]["query"], k=3
            )
            filtered = [doc for doc, score in docs_and_scores if score > 0.75]

            if not filtered:
                return {
                    "messages": state["messages"] + [
                        AIMessage(content="관련 문서에서 답을 찾지 못했습니다. 직접 답변해 드릴게요.")
                    ]
                }

            # 검색 결과 요약하도록 LLM 호출
            # 참조 내용이 너무 커지면 토큰 수가 급증 -> 현재는 테스트용으로 300자로 제한. 필요 시 늘려도 됨.
            context = "\n\n".join([doc.page_content[:300] for doc in filtered])
            prompt = f"다음 문서를 참고해서 질문에 답변해 주세요.\n\n질문: {call['args']['query']}\n\n문서 내용:\n{context}"
            summary = chat.invoke([HumanMessage(prompt)])

            return {"messages": state["messages"] + [summary]}

# 8. 그래프 구성
graph = StateGraph(MessagesState)

graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("llm")
graph.add_conditional_edges(
    "llm",
    lambda state: "tool"
    if getattr(state["messages"][-1], "tool_calls", None)
    else END,
    {"tool": "tool", END: END},
)
graph.add_edge("tool", END)

app = graph.compile()

# 실행 루프
while True:
    content = input("✅ 질문을 입력하세요 : ")
    if content.lower() == "q":
        break

    input_state = {"messages": [HumanMessage(content)]}
    final_state = app.invoke(input_state)

    # 마지막 AI 메시지만 출력
    last_ai_msg = None
    for m in final_state["messages"]:
        if m.type == "ai":
            last_ai_msg = m

    if last_ai_msg:
        print("\n📌 답변:\n", last_ai_msg.content.strip(), "\n")