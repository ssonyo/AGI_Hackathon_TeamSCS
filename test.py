# pip install -qU langchain-core langchain-upstage

import os
from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage
 
api_key = os.getenv("UPSTAGE_API_KEY")
chat = ChatUpstage(api_key=api_key, model="solar-pro")
 
messages = [
    HumanMessage(
        content="너 부동산 투자좀 아니?"
    )
]
 
response = chat.invoke(messages)
print(response)