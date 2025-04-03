# ✅ Future Real Estate/Land Derivative Financial Product Generator (AI Agent-based)

## 🔗 Project Introduction

This project combines **public policy documents + real-time real estate information** to create a system where  
**AI Agents design actual investment products and analyze returns**.  
It particularly aims to be a future-oriented financial service that considers **Web3-based tokenization and derivative product design structures**.

---

## 🚀 Key Technologies

| Technology                               | Description                                                                             |
| ---------------------------------------- | --------------------------------------------------------------------------------------- |
| **LangChain Agent**                      | Tool-based AI Agent for analyzing policy documents and estimating returns               |
| **RAG (Retrieval Augmented Generation)** | Embedding → Q&A based on Ministry of Land PDF documents                                 |
| **Upstage LLM (Solar-Pro)**              | High-performance Korean-specialized LLM for document analysis and automation            |
| **Upstage Embedding + FAISS**            | Vector embedding and search optimization for RAG construction                           |
| **Selenium-based Crawler**               | Collects real property information (price, area, urban planning) from Naver Real Estate |
| **Geo-based Analysis**                   | Property search and mapping based on regional coordinates                               |
| **Rule-based ROI Calculator**            | Return calculation applying policy status-based 'premium coefficients' as rules         |
| **Streamlit**                            | Interactive analysis + portfolio report UI configuration                                |

---

## 📦 Feature Summary

| Feature                            | Description                                                                 |
| ---------------------------------- | --------------------------------------------------------------------------- |
| Document Upload                    | Upload Ministry of Land policy documents (PDF)                              |
| Administrative District Extraction | Automatic extraction of policy-targeted districts from document content     |
| Location-based Property Collection | Crawling and refining real properties in extracted districts                |
| Return Analysis                    | ROI calculation applying premium coefficients based on policy status        |
| Derivative Product Generation      | LLM-based portfolio composition (including property weights)                |
| Research Report Generation         | AI Agent automatically creates comprehensive analysis reports for investors |

---

## 💡 Project Features and Attempts

### ✅ 1. Upstage-based Korean-specialized AI + RAG Integration

- Excellent document analysis performance based on Solar-Pro
- Optimized Korean document RAG construction with UpstageDocumentParseLoader and Embedding

### ✅ 2. LangChain Agent + Custom Tool for Return Prediction Structure

- `ExtractDongsTool`: Administrative district extraction
- `EstimateYieldTool`: Development status-based premium coefficient estimation
- Action chain configuration through PromptTemplate-based tool calls

### ✅ 3. Rule-based Return Calculation Formula Application

- Coefficient assignment based on policy status (construction completion/redevelopment area/unspecified)
- Actual return calculation combining purchase price, rental possibility, etc.

### ✅ 4. Real-time Connection between Real Estate Properties and Policy Documents

- Real-time property information crawling from Naver Real Estate
- Property scoring and composition based on policy benefit status in policy documents

### ✅ 5. User-friendly Interface based on Streamlit

- Interactive button flow providing analysis → property collection → return analysis → report
- Portfolio composition including actual weight structure

---

## 🌍 Future Expansion Directions

- Web3-based derivative product tokenization + DAO-based exchange expansion
- Introduction of investor risk management-based rating model (LLM + statistics)
- Providing automated financial product design API by learning various regional documents

---

## 📄 Usage Instructions

1. Run Streamlit:
   streamlit run app.py

2. Document upload → Analysis → Property check → Return analysis → Derivative product composition

3. (Optional) Portfolio diversification possible by repeatedly learning new regional policy documents

---

## 🙌 Team Introduction

| Team & Batch | Name           | Role                                                                                                                         |
| ------------ | -------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| DS 24th      | Park Jeongyang | Ministry of Land policy document crawling and RAG core principles explanation, LangChain structure design                    |
| DS 25th      | Kim Soyun      | Prompt enhancement, output diversification attempts, LangChain structure design                                              |
| DE 25th      | Baek Junho     | Selenium automation, LangChain Agent system architecture design and implementation, Streamlit implementation, AWS deployment |
| DS 26th      | Cho Seokhee    | Prompt enhancement, LangSmith connection, validation structure design                                                        |

---

## 📌 Reference Links

- [Ministry of Land Policy Information](https://www.eum.go.kr/web/gs/gv/gvGosiList.jsp)
- [Naver Real Estate](https://land.naver.com/)
- [Derivative Financial Products based on Real Estate Price Index Paper](https://www.smallake.kr/wp-content/uploads/2015/12/20151219_224054.pdf)
