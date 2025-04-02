import os
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def embed_docs_node(state: dict) -> dict:
    '''
    문서 split하고 embedding, vectorDB에 적재하는 모듈
    '''
    document_text = state["document_text"]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([document_text])

    embeddings = UpstageEmbeddings(
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="embedding-query"
    )

    persist_dir = "./chroma_db"
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()

    state["vectordb_path"] = persist_dir
    return state

if __name__ == "__main__":

    test_state = {
        "document_text": """
        이 보고서는 강남구 신축 오피스텔의 가격 추이와 월세 수익률을 바탕으로 
        중장기 투자 전략을 제시합니다. 해당 지역의 인프라 확충 계획 및 수요 증가 전망을 포함하고 있습니다.
        아 저도 강남구 신축 오피스텔을 참 사고싶네요.
        """
    }

    output_state = embed_docs_node(test_state)
    print("\n=== Chroma Vector DB path ===")
    print(output_state["vectordb_path"])


    # 빼서 확인
    embeddings = UpstageEmbeddings(api_key=os.getenv("UPSTAGE_API_KEY"), model="embedding-query")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    print("# of docs:", db._collection.count())


    # 유사 문서 검색
    results = db.similarity_search("강남구 오피스텔", k=2)
    print("\n=== Similar documents: ===")
    for doc in results:
        print(doc.page_content[:200])