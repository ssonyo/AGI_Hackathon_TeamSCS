import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth

load_dotenv()

# 폰트 등록
FONT_PATH = os.path.join(os.path.dirname(__file__), "../fonts/NanumGothic.ttf")
pdfmetrics.registerFont(TTFont("NanumGothic", FONT_PATH))

# LLM 설정
chat = ChatUpstage(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="solar-pro"
)

def create_file_node(state: dict) -> dict:
    raw_text = state.get("final_report", "")
    
    # 형식 정돈 프롬프트
    refine_prompt = f"""
다음 보고서 내용을 깔끔한 보고서 형식으로 재작성해주세요.

요구사항:
- 제목은 가장 위에 한 줄로 굵게 작성해주세요.
- 각 항목(예: 수익률 분석, 리스크 분석 등)은 [1. 수익률 분석]과 같은 소제목 형식으로 써주세요.
- 각 문단 사이에는 한 줄 공백을 넣어 가독성을 높여주세요.
- 줄바꿈은 자연스럽게 처리해주세요.
- 전체적으로 보고서처럼 전문적인 느낌이 나도록 정돈해주세요.

보고서 원문:
{raw_text}
"""
    
    response = chat.invoke([{"role": "user", "content": refine_prompt}])
    refined_text = response.content.strip()

    # PDF 저장 경로
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "final_report.pdf")

    # PDF 생성
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin
    c.setFont("NanumGothic", 14)

    textobject = c.beginText(margin, y)
    textobject.setFont("NanumGothic", 14)

    max_width = width - 2 * margin
    line_spacing = 22

    for paragraph in refined_text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            textobject.textLine("")  # 문단 공백
            continue

        words = paragraph.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            if stringWidth(test_line, "NanumGothic", 14) < max_width:
                line = test_line
            else:
                textobject.textLine(line)
                line = word
        if line:
            textobject.textLine(line)

        textobject.textLine("")

        if textobject.getY() < margin:
            c.drawText(textobject)
            c.showPage()
            textobject = c.beginText(margin, height - margin)
            textobject.setFont("NanumGothic", 14)

    c.drawText(textobject)
    c.save()

    state["final_report_path"] = output_path
    return state

# ✅ 테스트 코드 (__main__)
if __name__ == "__main__":
    sample_state = {
        "query": "서울 강남구에 위치한 신축 오피스텔 투자",
        "final_report": (
            "서울 강남구에 위치한 신축 오피스텔 투자는 높은 임대 수요와 안정적인 수익률을 기대할 수 있습니다.\n"
            "본 보고서는 수익률 분석, 리스크 분석, 관련 시장 동향 등을 종합하여 투자 제안을 제공합니다.\n\n"
            "1. 수익률 분석:\n"
            "- 예상 월세 수익은 100만원 수준이며 연 수익률은 약 4.8%입니다.\n"
            "- 주변 시세와 비교하여 경쟁력 있는 임대료 수준입니다.\n\n"
            "2. 리스크 분석:\n"
            "- 금리 인상에 따른 대출 이자 부담 증가 가능성\n"
            "- 공급 과잉에 따른 공실 리스크\n\n"
            "3. 결론:\n"
            "강남구 신축 오피스텔은 안정적인 수익을 기대할 수 있으며, 중장기적으로도 우량한 투자처로 판단됩니다."
        )
    }

    result = create_file_node(sample_state)
    print(f"\n✅ PDF 생성 완료: {result['final_report_path']}")
