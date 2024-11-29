import streamlit as st
import os
from typing import Optional
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

os.environ["openai_api_key"] = ""

st.set_page_config(
    page_title="안녕하세요",
    page_icon="(～￣▽￣)～"
)


## 스트림릿 페이지 제목 설정
st.title("포트폴리오 기반 챗봇 만들기")


## 간단 안내 문구 추가
st.markdown("""
왼쪽 사이드바에서 파일을 선택해주세요.
""")


@st.cache_resource(show_spinner="Loading...") ## 리소스를 초기화하거나 연결하는 데 시간이 걸리는 작업을 캐싱해 효율성을 높입니다. 
def embedding_file(file: str) -> VectorStoreRetriever:
    """문서를 청크 단위로 분할하고 임베딩 모델(text-embedding-ada-002)을 통해 임베딩하여 vector store에 저장합니다. 이후 vector store를 기반으로 검색하는 객체를 생성합니다. 

    Args:
        file (str): pdf 문서 경로

    Returns:
        VectorStoreRetriever: 검색기 
    """
    
    ## 긴 텍스트를 작은 청크로 나누는 데 사용되는 클래스
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(       
        chunk_size=300,         ## 최대 청크 길이 정의
        chunk_overlap=100,      ## 청크 간 겹침 길이 정의
        separators=["\n\n"]     ## 텍스트를 나눌 때 사용할 구분자를 지정 (문단)
    )
    
    ## PDF 파일 불러오기
    loader = PyPDFLoader(f"./files/{file}")
    docs = loader.load_and_split(text_splitter=splitter)
    
    ## Embedding 생성 및 vector store에 저장
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(
        documents=docs,         ## 벡터 저장소에 추가할 문서 리스트
        embedding=embeddings    ## 사용할 임베딩 함수
    )
    
    ## 검색기로 변환: 현재 벡터 저장소를 기반으로 VectorStoreRetriever 객체를 생성하는 기능을 제공
    retriever = vector_store.as_retriever(
        search_type="similarity"    ## 어떻게 검색할 것인지? default가 유사도
    )

    return retriever


## 파일 선택
with st.sidebar:
    chat_clear = None
    
    ## 모델 선택
    model_name = st.selectbox(
        label="모델을 선택해주세요.", 
        placeholder= "Select Your Model", 
        options=["gpt-4o", "gpt-4o-mini"], 
        index=None
    )
    
    ## 파일 선택
    file = st.selectbox(
        label="파일을 선택해주세요.", 
        placeholder= "Select Your File", 
        options=(os.listdir("./files")), 
        index=None
    )

    ## 파일이 선택되면 실행    
    if file:
        retriever = embedding_file(file)        ## file 기반의 retriever 생성
        st.success(f"임베딩이 완료되었습니다.")
        st.info(f"현재 임베딩 된 파일명 : \n\n **{file}**") 
        col1, col2 = st.columns(2)
        chat_clear = col1.button("대화 내용 초기화")
        embed_clear = col2.button("임베딩 초기화")

        ## LLM 모델 생성
        llm = ChatOpenAI(
            temperature=0.1,
            model=model_name
        )          

        ## embed_clear 버튼이 선택되면 초기화
        if embed_clear:
            st.cache_resource.clear()

if "history" not in st.session_state or chat_clear:
    st.session_state["history"] = []   


## 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    context : {context}

    당신은 언제나 고객에게 최선을 다해 답변을 하며 말투는 굉장히 친근합니다. 직업은 전문 상담원입니다. 답변 시, 아래의 규칙을 지켜야만 합니다.
    규칙 1. 주어진 contextjdnft만을 이용하여 답변해야 합니다. 
    규칙 2. 주어진 context에서 답변을 할 수 없다면 "해당 질문은 알아낼 수 없는 질문입니다." 라고 대답하세요.
    규칙 3. 문자열에 A1, A2, A11, A22 등 필요 없는 문자는 제거한 뒤 출력합니다.
    규칙 4. 항상 친절한 말투로 응대합니다.
    """),
    ("human", "{query}")
])

query = st.chat_input("질문을 입력해주세요.")


def format_docs(docs: list[Document]) -> str:
    """문시 리스트에서 텍스트를 추출하여 하나의 문자로 합치는 기능을 합니다. 

    Args:
        docs (list[Document]): 여러 개의 Documnet 객체로 이루어진 리스트

    Returns:
        str: 모든 문서의 텍스트가 하나로 합쳐진 문자열을 반환
    """
    return "\n\n".join(doc.page_content for doc in docs)


def chat_history(
    message: Optional[str] = None, 
    role: Optional[str] = None, 
    show: bool=True
) -> None:
    """채팅 history를 관리하고 출력하는 함수입니다. 

    Args:
        message (Optional[str], optional): 새로운 메세지의 텍스트를 받는 인자
        role (Optional[str], optional): 메세지를 작성한 사용자의 역할을 지정
        show (bool, optional): 대화 히스토리를 화면에 출력할지 여부를 결정
    """
    ## 채팅을 session_state에 저장
    if message and role:
        st.session_state["history"].append(
            {
                "message":message, 
                "role":role
            }
        )
    
    ## message를 보여주는 코드
    if show:
        for m in st.session_state["history"]:
            with st.chat_message(m["role"]):
                st.write(m["message"])
                

def chat_llm(query: str) -> None:
    """사용자가 입력한 query를 기반으로 LLM 과의 대화를 처리하는 기능을 하는 함수입니다. 

    Args:
        query (str): _description_
    """    
    ## 사용자의 질문을 받아서 history에 저장
    chat_history(
        message=query, 
        role="user", 
        show=False
    )
    
    ## Q. chain은 순서대로 진행되는데 어떻게 질문이 먼저 들어오지 않았는데 검색기가 유사한 문서를 찾는 것일까?
    ## A. 애초에 쿼리를 retriever를 호출하면서 동시에 질문을 받는다.
    ## Q. retriever이 어떻게 질문을 받고 유사한 문서를 찾을까? 동작원리가 궁금하다. 
    ## A. 
    chain = {
        "context":retriever | RunnableLambda(format_docs), ## 외부 문서를 검색하는 객체를 불러오기 -> 검색된 문서를 전처리하는 함수
        "query":RunnablePassthrough() ## query를 그대로 전달
        } | prompt | llm
    
    result = chain.invoke(query)
    
    ## LLM의 답변을 history에 저장하며 보여준다. 
    chat_history(
        message=result.content, 
        role = "ai", 
        show=True
    )  
  
    
if query:
    chat_llm(query=query)