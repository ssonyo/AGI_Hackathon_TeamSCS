import os
from graph import app #graph.py 에서 컴파일
from pprint import pprint

input_state = {
    "file_path": "sample.pdf", 
    "query": "강남구 오피스텔, 보증금 1억, 월세 90만원 기준 수익률 및 위험 평가"
}

final_state = app.invoke(input_state)

print("\n=== Final Report 🥳 ===\n")
print(final_state.get("final_report", "[error] 뭔가 잘못됨...ㅠㅠ]"))
pprint(final_state)
