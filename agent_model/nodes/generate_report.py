import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

load_dotenv()

chat = ChatUpstage(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="solar-pro"
)

embeddings = UpstageEmbeddings(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="embedding-query"
)

# LangSmith 추적 가능한 Retriever 구성
def get_retriever_runnable(vectordb_path: str):
    db = Chroma(
        persist_directory=vectordb_path,
        embedding_function=embeddings
    )
    retriever = db.as_retriever()
    return RunnableLambda(lambda q: retriever.get_relevant_documents(q))


def generate_report_node(state: dict) -> dict:
    query = state.get("query", "특별한 조건 없음")
    vectordb_path = state.get("vectordb_path", None)
    yield_result = state.get("yield_result", "")
    risk_result = state.get("risk_result", "")

    context = ""
    if vectordb_path:
        retriever_runnable = get_retriever_runnable(vectordb_path)
        retrieved_docs = retriever_runnable.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    당신은 기관 투자자를 위한 부동산 파생상품을 설계·제안하는 AI 리서치 어드바이저입니다.  
    다음 조건과 문서를 바탕으로 투자 제안 보고서를 작성해주세요.

    ---

    📌 **보고서 목적**  
    - 사용자의 투자 조건과 국토부 고시 등 공공문서를 분석하여 부동산/토지 기반 파생상품을 기획  
    - 기관 투자자(자산운용사, 증권사, 리츠 등)에게 제안 가능한 수준의 정제된 리서치 보고서를 작성

    ---

    📌 **작성 지침**  
    - 마크다운 문법(`*`, `#`, `-`) 없이 문장 중심으로 작성  
    - 문단/소제목을 활용해 **명확한 구조**와 **투자 설득력**을 갖춘 문서 작성  
    - 단순 설명이 아닌 **상품화 가능한 구조 제안**을 포함  
    - **정책, 제도, 시장 동향**이 상품 설계에 어떻게 반영되는지 서술  
    - **파생상품의 기본 구조**와 수익 흐름, 리스크 완화 방안이 명확히 드러나야 함  
    - 전문성은 유지하되, 투자자에게 익숙한 표현으로 간결하고 실용적으로 작성

    ---

    📄 **보고서 구성**

    1. 보고서 제목  
    - 투자 대상, 입지, 전략 키워드를 반영한 제목

    2. 투자 개요 및 조건 요약  
    - 사용자 입력 조건 요약  
    - 투자 타깃 자산(토지, 부동산 등)에 대한 정의 및 전반적 투자 목적

    3. 공공문서 기반 시장 분석  
    - 국토부 고시 등 문서로부터 추출된 개발 계획, 규제, 수요 전망, 공급 상황 등을 요약  
    - 입지 경쟁력과 수급 분석 포함

    4. 수익 모델 및 파생상품 구조  
    - 예상 수익률 및 구성요소(임대수익, 매각차익 등)  
    - 파생상품 구조 설명: 예) 임대료 연동형, 수익 공유형, 만기형, 조건부 상환형  
    - 운용 구조 또는 SPV 활용 여부 등 포함

    5. 리스크 요인 및 완화 방안  
    - 금리, 환율, 공공정책 변경, 공급 과잉 등의 리스크 분석  
    - 각 리스크에 대한 헤지 전략 또는 구조적 보완책 제시

    6. 종합 평가 및 투자 제언  
    - 투자 적합성 판단  
    - 기대 수익률과 리스크 대비 수익성 평가  
    - 투자 시 고려사항 및 투자자 유의사항 정리

    ---

    📥 **입력 정보 요약**

    [사용자 조건]  
    {query}

    [수익률 분석 결과]  
    {yield_result}

    [리스크 분석 결과]  
    {risk_result}

    [공공 문서 요약 내용]  
    {context if context else "(공공 문서 내용 없음)"}
    """

    response = chat.invoke([HumanMessage(content=prompt)])
    final_report = response.content.strip()
    state["final_report"] = final_report

    return state


if __name__ == "__main__":
    test_state = {
        "query": "강남구, 오피스텔, 신축, 보증금 1억, 월세 90만원",
        "yield_result": "연 수익률은 약 4.8%로 추정됩니다. 월세 수익과 관리비를 반영했습니다.",
        "risk_result": "해당 지역은 금리 인상에 따른 투자 리스크가 존재하며, 공급 증가도 우려됩니다.",
        "vectordb_path": "./chroma_db"
    }

    output_state = generate_report_node(test_state)
    print("\n=== Output (Report) 🎆🎉📈 ===")
    print(output_state["final_report"])