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
다음 문서의 내용을 바탕으로 연 수익률을 계산해주세요.
가능하다면 수치와 근거를 포함해서 설명해주세요.

문서 내용:
{context}

사용자 쿼리 조건:
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