import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.documents import Document

# LangChain Core
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChain 공식 모듈
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # 메타에서 개발한 벡터 검색 라이브러리

# LangChain-Naver
from langchain_naver import ClovaXEmbeddings, ChatClovaX

load_dotenv()  # .env 파일 로드

VECTOR_DB_PATH = "faiss_index"
PDF_PATH = "/mnt/c/Users/user/Downloads/ransomware.pdf"

# 1. 벡터 DB 파일이 없으면 생성 후 vector_store 리턴
def create_vector_db():
    # 1-1. 문서 로딩 (PyPDF2로 텍스트 추출 후 LangChain Document로 변환)
    reader = PdfReader(PDF_PATH)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text, metadata={"page": i}))
    print(f"문서의 수: {len(docs)}")

    # 1-2. 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    splits = text_splitter.split_documents(docs)
    print(f"split size: {len(splits)}")

    # 1-3. 임베딩 생성
    embeddings = ClovaXEmbeddings(model="clir-emb-dolphin")

    # 1-4. 벡터 저장소 구축
    vector_store = FAISS.from_documents(splits, embeddings)

    # 1-5. 벡터 DB를 로컬에 저장
    vector_store.save_local(VECTOR_DB_PATH)

    return vector_store

# 2. 메인 로직
if os.path.exists(VECTOR_DB_PATH):
    print("기존 벡터 DB를 로드합니다.")
    embeddings = ClovaXEmbeddings(model="clir-emb-dolphin")
    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("새로운 벡터 DB를 생성합니다.")
    vector_store = create_vector_db()

# 3. Retriever 생성
retriever = vector_store.as_retriever()

# 4. Prompt Template 설정
prompt = PromptTemplate.from_template(
"""당신은 질문-답변(Question-Answering)을 수행하는 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
질문과 관련성이 높은 내용만 답변하고 추측된 내용을 생성하지 마세요. 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question: 
{question} 

#Context: 
{context} 

#Answer:"""
)

# 5. LLM 설정
llm = ChatClovaX(
    model="HCX-005",
    api_key=os.getenv("CLOVASTUDIO_API_KEY"),
)

# 6. Chain 생성
chain = prompt | llm | StrOutputParser()

# 7. 사용자 입력 루프
while True:
    question = input("\n\n당신: ")
    if question.lower() in ["끝", "exit", "quit"]:
        break

    # 문서 검색
    retrieved_docs = retriever.invoke(question)
    print(f"retrieved size: {len(retrieved_docs)}")
    combined_docs = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 체인 실행
    formatted_prompt = {"context": combined_docs, "question": question}
    result = ""
    for chunk in chain.stream(formatted_prompt):
        print(chunk, end="", flush=True)
        result += chunk