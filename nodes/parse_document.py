import os
from langchain_upstage import UpstageDocumentParseLoader
from langchain_core.documents import Document

def parse_document_node(state: dict) -> dict:
    file_path = state["file_path"]
    loader = UpstageDocumentParseLoader(file_path, ocr="force")
    pages = loader.load()

    full_text = "\n\n".join([page.page_content for page in pages])
    state["document_text"] = full_text
    return state


if __name__ == "__main__":
    # os.environ["UPSTAGE_API_KEY"] = "your-key-here" 
    test_state = {
        "file_path": "samplefile.pdf" 
    }
    output_state = parse_document_node(test_state)
    print("\n=== Parsed Document ===\n")
    print(output_state["document_text"][:500])