from pathlib import Path
import os
import pandas as pd
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_upstage import ChatUpstage, UpstageEmbeddings, UpstageDocumentParseLoader
from langchain.callbacks.base import BaseCallbackHandler
from langchain.globals import set_verbose

load_dotenv()

# --- 설정 ---
st.set_page_config(
    page_title="국토부문서 기반 토지 파생 상품 생성 Agent",
    page_icon="📄",
)

# --- 스트리밍 핸들러 정의 ---
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# --- LLM 세팅 ---
llm = ChatUpstage(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="solar-pro",
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

# --- 문서 임베딩 함수 ---
@st.cache_resource(show_spinner="🔍 문서를 분석 중입니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UpstageDocumentParseLoader(file_path, split="page")
    docs = loader.load()
    embeddings = UpstageEmbeddings(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="solar-embedding-1-large"
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# --- 좌표 CSV 불러오기 ---
@st.cache_data
def load_location_csv():
    df = pd.read_csv("pages/부동산_위치정보.csv", encoding="utf-8", header=None)
    df.columns = ["시도", "시군구", "읍면동", "위도", "경도"]
    return df

location_df = load_location_csv()

# --- 행정동 이름으로 위경도 찾기 ---
def find_coordinates(행정동이름: str):
    match = location_df[location_df["읍면동"].str.contains(행정동이름)]
    if match.empty:
        return None
    return match[["위도", "경도"]].values.tolist()


# --- 행정동 이름만 뽑기 ---
def extract_administrative_districts(text):
    return re.findall(r"\w+동", text)

# --- 응답에서 좌표를 시각화하는 함수 ---
def visualize_coords_from_response(response_text):
    행정동목록 = extract_administrative_districts(response_text)
    coords_list = []

    for name in 행정동목록:
        coords = find_coordinates(name)
        if coords:
            for lat, lon in coords:
                coords_list.append({"lat": lat, "lon": lon})
            st.markdown(f"📍 **{name}** 위치 시간화됨")
        else:
            st.markdown(f"❓ **{name}** 좌표 정보 없음")

    if coords_list:
        df_coords = pd.DataFrame(coords_list)
        df_coords.columns = ["lat", "lon"]
        st.map(df_coords)




# --- 대화 세션 저장 ---
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# --- 프롬프트 템플릿 ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Answer the question using ONLY the following context. 
If you're asked about administrative districts (행정지역), extract them precisely from the text. 
If you don't know the answer just say you don't know. DON'T make anything up.

Context: {context}
""",
        ),
        ("human", "{question}"),
    ]
)

# --- UI 시작 ---
st.title("📄 국토부문서 기반 부동산&토지형 파사상품 생성 Agent")

st.markdown(
    """
이 앱은 국토부 문서를 기반으로 행정지역, 개발 정보 등을 확인하고 AI펀드 매니저가 부동산, 토지형 파사상품을 생성하는 질문하고 채팅 시스템입니다.  
향후 지도 기반 매물 추천 및 파사상품 토큰 거리소로 확장 예정입니다.

작동 로직:
- 행정지역(시/도 시/군/구 음/면/동) 확인 매물 정보 크롤링 -> 행정동 바탕 투자 수익률 계산
- 행정동 바탕 투자 수익률 계산
- 파사상품 만들어줘 -> 파사상품 생성 시작
"""
)

with st.sidebar:
    file = st.file_uploader(
        "📄 고시 문서를 업로드하세요 (PDF 권장)",
        type=["pdf", "txt", "docx"],
    )

# --- 메인 로직 ---
if file:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    retriever = embed_file(file)

    send_message("\ubb38서 \ubd84석 \uc644료! \uad81금한 점을 \ubb3c어보세요 ✨", "ai", save=False)
    paint_history()

    message = st.chat_input("\ubb38서에 대해 질문해보세요...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
            st.markdown(response.content)
            visualize_coords_from_response(response.content)
else:
    st.session_state["messages"] = []

set_verbose(True)  # 디베객 보기
