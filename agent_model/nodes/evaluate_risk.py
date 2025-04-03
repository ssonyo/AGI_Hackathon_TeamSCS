# rag 종류 변경 고려해보면 좋을듯.
# 개인적으론 시간 좀 걸려도 퀄리티잇는 결과 뽑아야하니까 refine documents 방식이 조을것같아요!

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

# LangSmith 추적 가능한 Retriever 정의
def get_retriever_runnable(vectordb_path: str):
    db = Chroma(
        persist_directory=vectordb_path,
        embedding_function=embeddings
    )
    retriever = db.as_retriever()
    return RunnableLambda(lambda q: retriever.get_relevant_documents(q))

def evaluate_risk_node(state: dict) -> dict:
    query = state.get("query", "특별한 조건 없음")
    vectordb_path = state["vectordb_path"]

    retriever_runnable = get_retriever_runnable(vectordb_path)
    retrieved_docs = retriever_runnable.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
당신은 부동산 리스크 분석 전문가입니다.
다음 문서 내용을 바탕으로 투자 위험 요소를 분석하고 평가해주세요.
정책 변화, 금리, 입지 리스크 등을 중심으로 설명해주세요.

[문서 내용]:
{context}

[사용자 쿼리 조건]:
{query}
    """

    response = chat.invoke([HumanMessage(content=prompt)])
    state["risk_result"] = response.content.strip()
    return state


if __name__ == "__main__":
    test_state = {
        "query": "강남구, 오피스텔, 신축 위주",
        "vectordb_path": "./chroma_db"
    }

    output_state = evaluate_risk_node(test_state)
    print("\n=== Risk Evaluation Result ===")
    print(output_state["risk_result"])
