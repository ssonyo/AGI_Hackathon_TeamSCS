import os
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage

chat = ChatUpstage(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="solar-pro"
)

embeddings = UpstageEmbeddings(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="embedding-query"
)

# rag 는 전략 좀 더 정해지면 따로 module 로 구체화해서 import
def retrieve_context(query: str, vectordb_path: str, strategy: str = "simple") -> str:
    db = Chroma(persist_directory=vectordb_path, embedding_function=embeddings)
    if strategy == "simple":
        docs = db.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])
    return ""

def generate_report_node(state: dict) -> dict:
    query = state.get("query", "특별한 조건 없음")
    vectordb_path = state["vectordb_path"]
    yield_result = state.get("yield_result", "")
    risk_result = state.get("risk_result", "")

    context = retrieve_context(query, vectordb_path)

    prompt = f"""
당신은 부동산 파생상품 설계 전문가입니다.
다음 정보를 바탕으로 투자 제안 보고서를 작성해주세요.

1. 사용자 조건:
{query}

2. 수익률 분석 결과:
{yield_result}

3. 리스크 분석 결과:
{risk_result}

4. 관련 문서 정보:
{context}

각 항목을 조화롭게 종합하여 보고서 형태로 작성해주세요.
    """

    response = chat.invoke([HumanMessage(content=prompt)])
    state["final_report"] = response.content.strip()
    return state


if __name__ == "__main__":
    os.environ["UPSTAGE_API_KEY"] = "your-key-here"

    test_state = {
        "query": "강남구, 오피스텔, 신축, 보증금 1억, 월세 90만원",
        "yield_result": "연 수익률은 약 4.8%로 추정됩니다. 월세 수익과 관리비를 반영했습니다.",
        "risk_result": "해당 지역은 금리 인상에 따른 투자 리스크가 존재하며, 공급 증가도 우려됩니다.",
        "vectordb_path": "./chroma_db"
    }

    output_state = generate_report_node(test_state)
    print("\n=== Output (Report) 🎆🎉📈 ===")
    print(output_state["final_report"])
