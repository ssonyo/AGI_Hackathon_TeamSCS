import os
from agent_model.graph import app #graph.py 에서 컴파일
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

input_state = {
    "file_path": "agent_model/sample.pdf", 
    "query": "첨부한 고시 문서를 바탕으로 적절한 부동산 파생상품을 분석·설계하고, 투자 판단에 필요한 전문 보고서를 작성해주세요."
}

final_state = app.invoke(input_state)

print("\n=== Final Report 🥳 ===\n")
print(final_state.get("final_report", "[error] 뭔가 잘못됨...ㅠㅠ]"))
pprint(final_state)
