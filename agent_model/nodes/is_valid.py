from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()


chat = ChatUpstage(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="solar-pro"
)

def is_valid_node(state: dict) -> dict:
    document_text = state["document_text"]

    prompt = f"""
당신은 부동산 투자분석 전문가입니다.
다음 문서가 부동산 가치 평가 또는 투자 분석에 유효한 문서인지 판단해주세요.
유효하다면 "유효", 아니라면 "무효"라고만 답해주세요.

문서 내용:
{document_text}
    """

    response = chat.invoke([HumanMessage(content=prompt)])
    is_valid = "유효" in response.content

    state["is_valid"] = is_valid
    return state


if __name__ == "__main__":
    test_state = {
        "document_text": "이 보고서는 강남구 토지의 공시지가와 어쩌구를 보여줌."
    }
    output_state = is_valid_node(test_state)
    print("\n=== Is Document Valid? ===")
    print("[answer] Valid OOOOO" if output_state["is_valid"] else "[answer] Invalid XXXXX")
