import os
import time
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_naver import ChatClovaX
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_naver import ClovaXEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

# 환경 변수 로드
load_dotenv()

# 1) PDF 파일 읽기 
pdf_path = "**insert document path**"
reader = PdfReader(pdf_path)

documents = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        documents.append(Document(page_content=text, metadata={"source": pdf_path, "page": i+1}))

print(f"원본 문서 수: {len(documents)}")

# 2) 문서 분할 (Chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # 한 조각당 최대 길이
    chunk_overlap=50   # 앞뒤 겹치는 부분
)
split_docs = text_splitter.split_documents(documents)
print(f"쪼갠 문서 수: {len(split_docs)}")

# 3) 네이버 클로바 임베딩 모델
embeddings = ClovaXEmbeddings(
    model="clir-emb-dolphin",
)

# 4) InMemoryVectorStore 생성
vectorstore = InMemoryVectorStore(embedding=embeddings)

batch_size = 2  # 5개씩 처리
for i in range(0, len(split_docs), batch_size):
    batch = split_docs[i:i+batch_size]

    # ✅ 벡터스토어에 문서 추가 (내부에서 임베딩 자동 처리)
    vectorstore.add_documents(batch)

    print(f"✅ {i+len(batch)}/{len(split_docs)} 완료")
    time.sleep(1.0)  # 호출 간격 추가 → RateLimit 방지

# 5) Retriever 생성
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    name = "pdf_research",
    description="랜섬웨어 관련 PDF를 기반으로 검색하는 툴"
)

# 6) LLM(**ClovaX**) 초기화
chat = ChatClovaX(
    model="HCX-005",
    api_key=os.getenv("CLOVASTUDIO_API_KEY"),
)

# 7) Retriever 붙인 LLM에 질문 진행
def rag_agent(user_query: str):
    """사용자 질문 → LLM tool_call → retriever 실행 → 최종 답변 생성"""
    # (1) LLM에게 질문
    msg = chat.bind_tools([retriever_tool]).invoke(user_query)

    # (2) LLM이 tool_call 했는지 확인
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_call = msg.tool_calls[0]
        print(f"🔧 Tool 호출: {tool_call}")

        # (3) 실제 retriever 실행
        tool_result = retriever_tool.invoke(tool_call["args"])

        # (4) Tool 결과를 다시 LLM에 전달해서 최종 답변 생성
        final_answer = chat.bind_tools([retriever_tool]).invoke([
            msg,
            {
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call["id"],
            }
        ])
        return final_answer.content
    else:
        # LLM이 직접 답했을 경우
        return msg.content

# 8) 실행
while(1):
    query = input("\n✅질문을 입력하세요 : ")
    if query == 'q':
        break
    answer = StrOutputParser().parse(rag_agent(query))
    print('\n', answer)