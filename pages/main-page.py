# 📦 전체 구조: LLM 기반 RAG + LangChain Agent 통합형 + 커스텀 프롬프트

import json
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
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from fpdf import FPDF

def extract_article_no(url):
    match = re.search(r'articleNo=(\d+)', url)
    if match:
        return match.group(1)
    return None

def parse_table_to_dict(table):
    """<table> 태그를 받아 dict로 변환"""
    info = {}
    rows = table.select("tbody tr")
    for row in rows:
        cells = row.find_all(["th", "td"])
        for i in range(0, len(cells) - 1, 2):
            key = cells[i].get_text(strip=True)
            value = cells[i + 1].get_text(strip=True)
            info[key] = value
    return info

def get_property_details(property_list):
    results = []
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # ← 최신 크롬은 이렇게
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option("detach", True)

    chrome_options.add_experimental_option("excludeSwitches",["enable-logging"])

    service = Service(executable_path=ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, 10)

    try:
    
        for article_no in property_list:
            try:
                url = f"https://land.naver.com/info/printArticleDetailInfo.naver?atclNo={article_no}&atclRletTypeCd=E03&rletTypeCd=E03"
                driver.get(url)
                time.sleep(1)

                soup = BeautifulSoup(driver.page_source, 'html.parser')

                tables = soup.find_all("table", summary=True)
                main_info = {}
                detail_info = {}

                for table in tables:
                    summary = table.get("summary", "")
                    if "매물정보" in summary:
                        main_info = parse_table_to_dict(table)
                        # 매매가를 숫자로 변환
                        if "매매가" in main_info:
                            price_str = main_info["매매가"]
                            # "만원" 제거하고 숫자만 추출
                            price_num = int(re.sub(r'[^0-9]', '', price_str))
                            # 만원 단위를 원 단위로 변환
                            main_info["매매가_원"] = price_num * 10000
                            
                            # 대지면적을 숫자로 변환
                            if "대지면적" in main_info:
                                area_str = main_info["대지면적"]
                                # "㎡" 제거하고 숫자만 추출
                                area_num = int(re.sub(r'[^0-9]', '', area_str))
                                main_info["대지면적_㎡"] = area_num
                                
                                # 단위면적당 가격 계산
                                if area_num > 0:
                                    main_info["단위면적당가격"] = main_info["매매가_원"] // area_num
                    elif "매물세부정보" in summary:
                        detail_info = parse_table_to_dict(table)

                results.append({
                    "article_no": article_no,
                    "main_info": main_info,
                    "detail_info": detail_info
                })

            except Exception as e:
                print(f"{article_no} 매물 상세 추출 실패:", e)
                continue
    finally:
        driver.quit()

    return results


def get_property_list(latitude, longitude):
    st.write("📍 크롤링 시작:", latitude, longitude)
    """특정 위도, 경도의 매물 목록을 가져오는 함수"""
    results = []
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # ← 최신 크롬은 이렇게
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option("detach", True)

    chrome_options.add_experimental_option("excludeSwitches",["enable-logging"])

    service = Service(executable_path=ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        
        
        # 웹드라이버 설정
        wait = WebDriverWait(driver, 10)
        
        # 네이버 부동산 URL 생성
        url = f"https://new.land.naver.com/offices?ms={latitude},{longitude},16&a=TJ&b=A1&e=RETAIL"
        driver.get(url)
        time.sleep(2)  # 페이지 로딩 대기
        
        # 스크롤 컨테이너 찾기
        scroll_container = driver.find_element(By.CSS_SELECTOR, ".item_list.item_list--article")

        # 초기 높이 설정
        prev_height = driver.execute_script("return arguments[0].scrollHeight", scroll_container)
        same_count = 0  # 변화 없는 횟수

        while True:
            # 내부 컨테이너를 스크롤
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_container)
            time.sleep(1.5)

            new_height = driver.execute_script("return arguments[0].scrollHeight", scroll_container)

            if new_height == prev_height:
                same_count += 1
            else:
                same_count = 0

            if same_count >= 5:
                break

            prev_height = new_height

        # 매물 목록 가져오기
        items = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".item_list.item_list--article .item")))
        
        
        for item in items[:12]:  # 최대 12개까지만 처리
            try:
                item.click()
                time.sleep(1)
                
                article_no = extract_article_no(driver.current_url)
                if article_no:
                    results.append(article_no)
                    st.write(f"📝 매물 저장: {article_no}")
                
            except Exception as e:
                st.write(f"매물 처리 중 오류 발생: {e}")
                continue
                
        return results
                
    except Exception as e:
        st.write(f"페이지 처리 중 오류 발생: {e}")
        return results
        
    finally:
        if driver:
            driver.quit()

# # 사용 예시
# latitude = 37.430218  # 예시 위도
# longitude = 127.002425  # 예시 경도
# property_list = get_property_list(latitude, longitude)
# print(f"총 {len(property_list)}개의 매물이 발견되었습니다.")

# a = get_property_details(property_list)
# for i in a:
#     print(i)


        


load_dotenv()

st.set_page_config(page_title="미래형 부동산/토지 파생형 금융상품 생성기", page_icon="🚩")

class ChatCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        if "generated" not in st.session_state:
            st.session_state.generated = ""
        st.session_state.generated += token

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
    
    # ✅ chunk 크기 축소
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    # ✅ 4000 토큰 이하만 필터링
    filtered_chunks = [doc for doc in chunks if len(doc.page_content) <= 2800]

    embeddings = UpstageEmbeddings(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="solar-embedding-1-large"
    )
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(filtered_chunks, cached_embeddings)
    return vectorstore.as_retriever()

@st.cache_data
def load_location_csv():
    df = pd.read_csv("pages/부동산_위치정보.csv", header=None)
    df.columns = ["시도", "시군구", "읍면동", "위도", "경도"]
    return df

location_df = load_location_csv()

# Tool 클래스 정의
class ExtractDongsArgs(BaseModel):
    text: str = Field(description="행정동 이름이 포함된 텍스트")

class ExtractDongsTool(BaseTool):
    name: str = "ExtractDongs"
    description: str = "텍스트에서 읍면동 이름만 추출합니다."
    args_schema: Type[BaseModel] = ExtractDongsArgs

    def _run(self, text: str):
        cleaned = re.sub(r'\w+(시|군|구)', '', text)
        return re.findall(r"\w+동", cleaned)

class EstimateYieldArgs(BaseModel):
    text: str = Field(description="문서 내용이 포함된 텍스트")

class EstimateYieldTool(BaseTool):
    name: str = "EstimateYield"
    description: str = "문서내용을 기반으로 수익률을 예측합니다."
    args_schema: Type[BaseModel] = EstimateYieldArgs

    def _run(self, text: str):
        # 개발 상태 프리미엄 계수 추출
        premium_factors = []
        
        # 공사완료 고시
        if "공사완료" in text or "입주" in text or "분양" in text:
            premium_factors.append(1.4)
            
        # 정비구역 고시 확정
        if "지정고시" in text or "개발예정" in text or "정비구역" in text:
            premium_factors.append(1.2)
            
        # 개발 미지정
        if not premium_factors:
            premium_factors.append(1.0)
            
        # 평균 프리미엄 계수 계산
        avg_premium = sum(premium_factors) / len(premium_factors)
        
        return f"개발 상태 프리미엄 계수: {avg_premium:.1f} (총 {len(premium_factors)}개 기준)"

tools = [ExtractDongsTool(), EstimateYieldTool()]

analysis_prompt = PromptTemplate(
    input_variables=["context", "question", "tool_names"],
    template="""
문서 내용은 다음과 같습니다:
{context}

사용자 질문:
{question}

당신은 부동산 투자 전문가입니다. 다음 중 필요한 작업만 수행하세요:

1. 행정동 추출
- 문서에서 언급된 읍면동만 추출 (ExtractDongsTool 사용)
- 시군구는 제외하고 읍면동만 추출

2. 수익률 분석
- 고시내용 기반으로 계발계수 예측 (EstimateYieldTool 사용)
- 예측된 계발계수를 추출
- 계발계수 예측의 근거 추출

도구 목록:
{tool_names}

출력 형식:
Thought: ...
Action: ...
Action Input: ...
Observation: ...
Final Answer: ...
"""
)

report_prompt = PromptTemplate(
    input_variables=["context", "question", "tool_names", "recommendation"],
    template="""
문서 내용은 다음과 같습니다:
{context}

사용자 질문:
{question}

추천 매물:
{recommendation}

당신은 부동산/토지형 파생상품 생성 전문 펀드매니저 AI Agent이며, 기관 투자자에게 제안할 **정제된 부동산/토지형 파생상품 포트폴리오**를 작성합니다.

보고서는 다음 지침을 따르세요:

---

📌 **보고서 목적**  
- 사용자의 투자 조건과 국토부 고시 등 공공문서를 분석하여 부동산/토지 기반 파생상품을 기획  
- 기관 투자자(자산운용사, 증권사, 리츠 등)에게 제안 가능한 수준의 정제된 리서치 보고서 작성

📌 **작성 지침**  
- 마크다운 문법 없이 문장 중심으로 작성  
- 문단/소제목을 활용해 구조적이고 설득력 있게  
- 정책, 시장 동향이 상품 설계에 미치는 영향을 설명  
- 수익 흐름, 리스크 분석 및 완화 방안 포함  
- 전문적이지만 실용적 언어 사용

📄 **보고서 구성**

1. 보고서 제목  
2. 투자 개요 및 조건 요약  
3. 공공문서 기반 시장 분석  
4. 수익 모델 및 파생상품 구조  
    - 매물 정보 recommendation
    - 파생상품 구조 (매물 비율)
    - 수익률 (EstimateYieldTool 사용)
5. 리스크 요인 및 완화 방안  
6. 종합 평가 및 투자 제언  


1. 행정동 추출
- 문서에서 언급된 읍면동만 추출 (ExtractDongsTool 사용)
- 시군구는 제외하고 읍면동만 추출

2. 수익률 분석
- 고시내용 기반으로 계발계수 예측 (EstimateYieldTool 사용)
- 예측된 계발계수를 추출
- 계발계수 예측의 근거 추출


도구 목록:
{tool_names}

---
Final Answer: ...


"""
)

def parse_and_display_output(text: str):
    pattern = r"(Thought:.*?)(?=(Thought:|Final Answer:|$))"
    thought_blocks = re.findall(pattern, text, re.DOTALL)

    for block, _ in thought_blocks:
        thought_match = re.search(r"Thought:(.*)", block)
        action_match = re.search(r"Action:(.*)", block)
        action_input_match = re.search(r"Action Input:(.*)", block)
        obs_match = re.search(r"Observation:(.*)", block)

        with st.chat_message("ai"):
            if thought_match:
                st.markdown(f"🧠 **분석:** {thought_match.group(1).strip()}")
            if action_match:
                st.markdown(f"🔧 **작업:** `{action_match.group(1).strip()}`")
            if action_input_match:
                st.markdown(f"📥 **입력:** `{action_input_match.group(1).strip()}`")
            if obs_match:
                st.markdown(f"📊 **결과:** {obs_match.group(1).strip()}")

    final_match = re.search(r"Final Answer:(.*)", text, re.DOTALL)
    if final_match:
        with st.chat_message("ai"):
            st.markdown(f"✨ **최종 결과:**\n\n{final_match.group(1).strip()}")

def run_agent_with_query(query, retriever):
    st.session_state.generated = ""
    context_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in context_docs])

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
    )

    formatted_prompt = analysis_prompt.format(
        context=context,
        question=query,
        tool_names='[' + ', '.join(tool.name for tool in tools) + ']'
    )

    agent.run(formatted_prompt)
    parse_and_display_output(st.session_state.generated)

def run_agent_with_query2(query, retriever, recommendation):
    st.session_state.generated = ""
    context_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in context_docs])

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
    )

    formatted_prompt = report_prompt.format(
        context=context,
        question=query,
        tool_names='[' + ', '.join(tool.name for tool in tools) + ']',
        recommendation=recommendation
    )

    agent.run(formatted_prompt)
    
    # 보고서 형식으로 출력
    with st.container():
        st.markdown("---")
        st.markdown("## 📑 투자 분석 보고서")
        st.markdown("---")
        
        # 보고서 내용을 섹션별로 구분하여 표시
        report_sections = st.session_state.generated.split("\n\n")
        for section in report_sections:
            if section.strip():
                st.markdown(f"### {section.strip()}")
                st.markdown("---")
    
    # 원본 분석 내용도 함께 표시
    st.markdown(st.session_state.generated)

# 메인 UI
st.title("국토부 문서 기반 미래형 토지/부동산 파생상품 생성 에이전트")
# 사용 방법 안내
st.markdown("""
### 📌 사용 방법
1. 분석할 문서를 업로드하세요
2. '행정동 추출' 버튼으로 문서에서 언급된 행정동을 찾거나
3. 직접 행정동을 입력하고 '매물 분석' 버튼을 눌러 매물 정보와 수익률을 확인하세요
4. AI Agent 가 추천하는 미래형 토지/부동산 파생 금융상품의 포트폴리오를 확인해보세요
5. 아래 링크에서 고시를 확인해보세요. https://www.eum.go.kr/web/gs/gv/gvGosiList.jsp?listSize=10&pageNo=2&zonenm=&startdt=&enddt=&chrgorg=&selSggCd=11&select2=1100000000&select_3=&gosino=&gosichrg=&prj_nm=&prj_cat_cd=&geul_yn=Y&gihyung_yn=&silsi_yn=&mobile_yn=
""")



file = st.file_uploader("📎 분석할 문서를 업로드하세요", type=["pdf", "txt"])

if file:
    retriever = embed_file(file)
    if "generated" not in st.session_state:
        st.session_state.generated = ""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("1️⃣ 행정동 추출"):
            with st.spinner("행정동을 추출하고 있습니다..."):
                query = "문서에서 언급된 행정동을 모두 추출해주세요."
                run_agent_with_query(query, retriever)
                st.toast("✅ 행정동 추출 완료!")

    with col2:
        dong = st.text_input("분석할 행정동을 입력하세요", placeholder="예: 미아동")
        if st.button("2️⃣ 매물 분석") and dong:
            with st.spinner(f"{dong} 매물을 분석중입니다..."):
                # 위경도 조회
                match = location_df[location_df["읍면동"] == dong]
                if not match.empty:
                    lat = float(match.iloc[0]["위도"])
                    lon = float(match.iloc[0]["경도"])
                    
                    # 매물 크롤링
                    property_list = get_property_list(lat, lon)
                    property_details = get_property_details(property_list)
                    # 단위면적당가격 기준으로 정렬하여 상위 5개 매물 추출
                    if property_details:
                        sorted_properties = sorted(
                            property_details,
                            key=lambda x: x["main_info"].get("단위면적당가격", 0),
                            reverse=True
                        )[:5]
                        property_details = sorted_properties
                        st.write(f"📊 총 {len(property_details)}개의 매물이 선정되었습니다.")
                    else:
                        st.warning("⚠️ 해당 지역에서 매물을 찾을 수 없습니다.")
                        property_details = []
                    # 결과 저장
                    st.session_state.latest_properties = property_details
                    st.session_state.current_dong = dong
        else:
            st.error("해당 행정동을 찾을 수 없습니다.")
        if st.button("3️⃣ 수익률 분석") and dong:
            with st.spinner(f"{dong} 수익률을 분석중입니다..."):
            # 수익률 분석
                query = f"{dong}의 계발계수를 바탕으로 수익률을 분석해주세요."
                run_agent_with_query(query, retriever)
        st.toast("✅ 분석 완료!")
        

    # 매물 정보 표시
    if "latest_properties" in st.session_state:
        st.markdown(f"### 📋 {st.session_state.current_dong} 선정 매물 정보")
        for idx, prop in enumerate(st.session_state.latest_properties):
            with st.expander(f"📌 매물 {idx+1}"):
                st.json(prop)
    if st.button("4️⃣ 파생상품 포트폴리오 생성"):
        property_details = st.session_state.latest_properties
        match = location_df[location_df["읍면동"] == dong]
        with st.spinner("포트폴리오를 생성중입니다..."):
            # LLM을 통한 포트폴리오 최적화
            query = f"""
            {dong}의 매물 정보를 바탕으로 최적의 토지형 파생상품 포트폴리오를 구성해주세요.
        
            매물의 각각 비중은 27 26 25 24 23%로 잡아주세요
            파생상품 구조는 매물 비율을 바탕으로 잡아주세요
            
            다음 사항을 고려하여 포트폴리오를 구성해주세요:
            1. 리스크 분산: 지역적, 시장적 리스크를 고려한 분산 투자
            2. 수익률 극대화: 단위면적당 가격, 개발 잠재력 등을 고려한 수익성 분석
            3. 지역 특성: 인프라, 교통, 상업시설 등 지역 개발 잠재력
            4. 투자 기간: 단기/중기/장기 투자 전략에 따른 포트폴리오 구성
            5. 유동성: 매매 활성도, 거래량 등을 고려한 유동성 분석
            """
            st.markdown(f"### 📊 {dong} 파생상품 포트폴리오")
            
            portfolio_analysis = run_agent_with_query2(query, retriever, property_details)