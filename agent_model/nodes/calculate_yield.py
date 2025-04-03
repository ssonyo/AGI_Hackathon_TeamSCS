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
    retriever = db.as_retriever(search_kwargs={"k": 10})  # ✅ Top-k를 10으로 확대

    # LangSmith에서 추적되는 Runnable 형태
    return RunnableLambda(lambda q: retriever.get_relevant_documents(q))

def calculate_yield_node(state: dict) -> dict:
    query = state.get("query", "특별한 조건 없음")
    vectordb_path = state.get("vectordb_path")

    context = ""
    if vectordb_path:
        retriever_runnable = get_retriever_runnable(vectordb_path)
        retrieved_docs = retriever_runnable.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
당신은 부동산 금융 전문가입니다.

아래 문서에서 명시적인 수익률 관련 수치가 없다면,
가상의 투자금, 수익 구조, 수익률을 상식과 문서 기반 추론에 따라 제시해주세요.
문서에 등장하는 용적률, 세대수, 주택면적, 용도지구 등과 같은 정보도 함께 고려해 
적절한 수익 시뮬레이션을 만들어주시면 됩니다.

※ 단, 결과는 실제 투자 자문이 아닌 시뮬레이션 예시임을 전제로 작성해주세요.

---

[문서 내용]
{context}

[사용자 쿼리 조건]
{query}
"""

    response = chat.invoke([HumanMessage(content=prompt)])
    state["yield_result"] = response.content.strip()
    return state



if __name__ == "__main__":
    test_state = {
        "query": "파생상품 기획하고 수익률 계산해줘.",
        "vectordb_path": "./chroma_db"
    }

    output_state = calculate_yield_node(test_state)
    print("\n=== Yield rate % ===")
    print(output_state["yield_result"])