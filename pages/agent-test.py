
from pathlib import Path
import os
import pandas as pd
import re
from typing import Type
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain_upstage import ChatUpstage, UpstageEmbeddings, UpstageDocumentParseLoader
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel, Field
import requests
from langchain_community.tools import BaseTool


load_dotenv()
upstage_api_key = os.environ.get("UPSTAGE_API_KEY")

llm = ChatUpstage(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="solar-pro",
    temperature=0.2,
    streaming=True,
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



class KoreanStockSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="검색하고자 하는 한국 기업명. 예: 삼성전자, 네이버"
    )

class KoreanStockSymbolSearchTool(BaseTool):
    name: str = "KoreanStockSymbolSearchTool"
    description: str = """
    한국 기업의 주식 종목 코드를 찾는 도구입니다.
    기업명을 입력하면 해당 기업의 종목 코드를 반환합니다.
    """
    args_schema: Type[KoreanStockSymbolSearchToolArgsSchema] = KoreanStockSymbolSearchToolArgsSchema

    def _run(self, query):
        headers = {"Authorization": f"Bearer {upstage_api_key}"}
        response = requests.get(
            f"{UPSTAGE_BASE_URL}/search?keyword={query}",
            headers=headers
        )
        return response.json()

class KoreanCompanyOverviewArgsSchema(BaseModel):
    symbol: str = Field(
        description="기업의 종목 코드. 예: 005930 (삼성전자)",
    )

class KoreanCompanyOverviewTool(BaseTool):
    name: str = "KoreanCompanyOverview"
    description: str = """
    한국 기업의 기본 정보를 조회하는 도구입니다.
    종목 코드를 입력하면 기업의 개요 정보를 반환합니다.
    """
    args_schema: Type[KoreanCompanyOverviewArgsSchema] = KoreanCompanyOverviewArgsSchema

    def _run(self, symbol):
        headers = {"Authorization": f"Bearer {upstage_api_key}"}
        response = requests.get(
            f"{UPSTAGE_BASE_URL}/company/{symbol}/overview",
            headers=headers
        )
        return response.json()

agent = initialize_agent(
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ✅ 이걸로 바꾸기
    verbose=True,
    handle_parsing_errors=True,
    tools=[
        KoreanStockSymbolSearchTool(),
        KoreanCompanyOverviewTool(),
        # 필요한 다른 도구들도 추가할 수 있습니다
    ],
)

prompt = "네이버의 주식 정보를 분석해서 투자 가치가 있는지 알려줘"

agent.invoke(prompt) 