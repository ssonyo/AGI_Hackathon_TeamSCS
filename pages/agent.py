# 📦 전체 구조: LLM 기반 RAG + LangChain Agent 통합형 + 커스텀 프롬프트

from pathlib import Path
import os
import pandas as pd
import re
import streamlit as st
from dotenv import load_dotenv
from typing import Type
from pydantic import BaseModel, Field
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain_upstage import ChatUpstage, UpstageEmbeddings, UpstageDocumentParseLoader
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools import BaseTool

load_dotenv()

st.set_page_config(page_title="국토부문서 기반 파생상품 생성 Agent", page_icon="📄")

class ChatCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        if "generated" not in st.session_state:
            st.session_state.generated = ""
        st.session_state.generated += token
          # 또는 st.chat_message("ai").write(...) 로 교체 가능
llm = ChatUpstage(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="solar-pro",
    temperature=0.2,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)

@st.cache_resource(show_spinner="📄 문서를 임베딩 중입니다...")
def embed_file(file):
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())

    docs = UpstageDocumentParseLoader(file_path, split="page").load()
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = UpstageEmbeddings(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="solar-embedding-1-large"
    )
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(chunks, cached_embeddings)
    return vectorstore.as_retriever()

@st.cache_data
def load_location_csv():
    df = pd.read_csv("pages/부동산_위치정보.csv", header=None)
    df.columns = ["시도", "시군구", "읍면동", "위도", "경도"]
    return df

location_df = load_location_csv()

# 🧩 BaseTool 기반 커스텀 도구 클래스들

class ExtractDongsArgs(BaseModel):
    text: str = Field(description="행정동 이름이 포함된 텍스트")

class ExtractDongsTool(BaseTool):
    name: str = "ExtractDongs"
    description: str = "텍스트에서 행정동 이름을 추출합니다."
    args_schema: Type[BaseModel] = ExtractDongsArgs

    def _run(self, text: str):
        return re.findall(r"\w+동", text)

class FindCoordsArgs(BaseModel):
    dong: str = Field(description="행정동 이름")

class FindCoordsTool(BaseTool):
    name: str = "FindCoords"
    description: str = "행정동 이름으로 위경도를 조회합니다."
    args_schema: Type[BaseModel] = FindCoordsArgs

    def _run(self, dong: str):
        match = location_df[location_df["읍면동"].str.contains(dong)]
        if match.empty:
            return None
        return match[["위도", "경도"]].values.tolist()

class GetRealEstatesArgs(BaseModel):
    coords: list = Field(description="[(위도, 경도)] 형태의 좌표 리스트")

class GetRealEstatesTool(BaseTool):
    name: str = "GetRealEstates"
    description: str = "주어진 좌표로 주변 매물 정보를 반환합니다."
    args_schema: Type[BaseModel] = GetRealEstatesArgs

    def _run(self, coords: list):
        return [{"lat": lat, "lon": lon, "매물명": "OO오피스텔"} for lat, lon in coords]

class EstimateYieldArgs(BaseModel):
    listings: list = Field(description="매물 목록")

class EstimateYieldTool(BaseTool):
    name: str = "EstimateYield"
    description: str = "매물 목록을 기반으로 수익률을 예측합니다."
    args_schema: Type[BaseModel] = EstimateYieldArgs

    def _run(self, listings: list):
        return f"수익률 예측 결과: 평균 7.2% (총 {len(listings)}개 매물 기준)"

class CreateFasaArgs(BaseModel):
    summary: str = Field(description="요약 내용")

class CreateFasaTool(BaseTool):
    name: str = "CreateFasa"
    description: str = "최종 파생상품 요약을 생성합니다."
    args_schema: Type[BaseModel] = CreateFasaArgs

    def _run(self, summary: str):
        return f"파생상품 요약 생성 완료:\n{summary}"

tools = [
    ExtractDongsTool(),
    FindCoordsTool(),
    GetRealEstatesTool(),
    EstimateYieldTool(),
    CreateFasaTool(),
]

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
문서 내용은 다음과 같습니다:
{context}

사용자 질문:
{question}

당신은 부동산/토지 투자 전문 AI Agent입니다.
행정동 추출 → 위경도 조회 → 매물 정보 확인 → 수익률 계산 → 파생상품 생성의 흐름으로 질문에 답하세요.
필요시 아래 도구들을 사용하세요:
{tool_names}

출력 형식은 반드시 다음을 따르세요:
Thought: ...\nAction: ...\nAction Input: ...\nObservation: ...\n...
Final Answer: ...
❗ Final Answer 이후에는 절대 아무 것도 출력하지 마세요.
"""
)

import re

def parse_and_display_output(text: str):
    """
    LangChain Agent의 출력 텍스트를
    Thought / Action / Observation / Final Answer 단위로 파싱해서 Streamlit UI에 출력
    """
    pattern = r"(Thought:.*?)(?=(Thought:|Final Answer:|$))"
    thought_blocks = re.findall(pattern, text, re.DOTALL)

    for block, _ in thought_blocks:
        # 각 블록에서 구성 요소 추출
        thought_match = re.search(r"Thought:(.*)", block)
        action_match = re.search(r"Action:(.*)", block)
        action_input_match = re.search(r"Action Input:(.*)", block)
        obs_match = re.search(r"Observation:(.*)", block)

        with st.chat_message("ai"):
            if thought_match:
                st.markdown(f"🧠 **Thought:** {thought_match.group(1).strip()}")
            if action_match:
                st.markdown(f"🔧 **Action:** `{action_match.group(1).strip()}`")
            if action_input_match:
                st.markdown(f"📥 **Action Input:** `{action_input_match.group(1).strip()}`")
            if obs_match:
                st.markdown(f"📊 **Observation:** {obs_match.group(1).strip()}")

    # 마지막 Final Answer
    final_match = re.search(r"Final Answer:(.*)", text, re.DOTALL)
    if final_match:
        with st.chat_message("ai"):
            st.markdown(f"✨ **Final Answer:**\n\n{final_match.group(1).strip()}")
            
def run_agent_with_query(query, retriever):
    st.session_state.generated = ""  # 초기화

    context_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in context_docs])

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    formatted_prompt = custom_prompt.format(
        context=context,
        question=query,
        tool_names='[' + ', '.join(tool.name for tool in tools) + ']'
    )

    # ✅ 에이전트 실행
    agent.run(formatted_prompt)

    # ✅ UI 출력 (스트리밍 끝난 후 정제된 출력)
    parse_and_display_output(st.session_state.generated)



st.title("📄 국토부 기반 파생상품 에이전트")

file = st.sidebar.file_uploader("📎 고시 문서를 업로드하세요", type=["pdf", "txt"])

if file:
    retriever = embed_file(file)
    if "generated" not in st.session_state:
        st.session_state.generated = ""
    
    # user_input = st.chat_input("문서에서 무엇을 도와드릴까요?")
    # if user_input:
    #     st.chat_message("user").write(user_input)
    #     with st.spinner("에이전트가 분석 중입니다..."):
    #         run_agent_with_query(user_input, retriever)
    st.sidebar.markdown("### 📊 분석 단계")
    
    if st.sidebar.button("1️⃣ 관련 행정동 추출"):
        with st.spinner("행정동을 추출하고 있습니다..."):
            query = "문서에서 언급된 행정동을 모두 추출해주세요."
            run_agent_with_query(query, retriever)
    
    if st.sidebar.button("2️⃣ 수익률 분석"):
        with st.spinner("수익률을 분석하고 있습니다..."):
            query = "추출된 행정동들의 예상 수익률을 분석해주세요."
            run_agent_with_query(query, retriever)
    
    if st.sidebar.button("3️⃣ 위험도 분석"):
        with st.spinner("위험도를 분석하고 있습니다..."):
            query = "해당 지역의 투자 위험도를 분석해주세요."
            run_agent_with_query(query, retriever)
    
    if st.sidebar.button("4️⃣ 종합 리포트 생성"):
        with st.spinner("종합 리포트를 생성하고 있습니다..."):
            query = "지금까지의 분석을 종합한 투자 리포트를 생성해주세요."
            run_agent_with_query(query, retriever)

    # 메인 영역에 설명 추가
    st.markdown("""
    ### 📌 사용 방법
    1. 사이드바에서 분석하고 싶은 문서를 업로드하세요
    2. 각 분석 단계 버튼을 순서대로 클릭하세요:
        - 행정동 추출: 문서에서 언급된 행정동을 찾습니다
        - 수익률 분석: 각 지역의 예상 수익률을 계산합니다
        - 위험도 분석: 투자 위험 요소를 분석합니다
        - 종합 리포트: 모든 분석 결과를 종합합니다
    """)
