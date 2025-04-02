from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage
import os

chat = ChatUpstage(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="solar-pro"
)

def calculate_yield_node(state: dict) -> dict:
    document_text = state["document_text"]
    query = state.get("query", "특별한 조건 없음")

    prompt = f"""
당신은 부동산 금융 전문가입니다.
다음 문서의 내용을 바탕으로 연 수익률을 계산해주세요.
가능하다면 수치와 근거를 포함해서 설명해주세요.

문서 내용:
{document_text}

사용자 쿼리 조건:
{query}
    """

    response = chat.invoke([HumanMessage(content=prompt)])
    state["yield_result"] = response.content.strip()
    return state



if __name__ == "__main__":

    test_state = {
        "document_text": """
        이 문서는 강남구 신축 오피스텔 투자 제안서입니다.
        보증금 1억, 월세 90만원 조건이며, 매입가는 5억입니다.
        관리비는 월 10만원입니다.
        """,
        "query": "강남구 오피스텔 기준으로 수익률 계산"
    }

    output_state = calculate_yield_node(test_state)
    print("\n=== Yield rate %% ===")
    print(output_state["yield_result"])