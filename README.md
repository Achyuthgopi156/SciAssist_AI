# SCIAssist AI

SCIAssist AI is an AI-powered academic assistant that leverages Retrieval-Augmented Generation (RAG), Vision-Language Models (LLaMA 3.2), and FAISS to deliver accurate, context-aware responses from research documents and scientific PDFs.

## 🚀 Features

- 🔍 Intelligent Search and Retrieval using FAISS and embedding models
- 📄 PDF parsing with support for text and visual elements
- 💬 Natural language Q&A using Transformer-based models
- 📊 (Partial) support for graphical data extraction
- 🌐 Web-based interface using Streamlit
- ⚙️ Backend integration with Flask and FastAPI

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Flask, FastAPI
- **AI Models:** LLAMA 3.2, BERT, GPT-style Transformers
- **Embeddings & Search:** FAISS
- **PDF Parsing:** PyMuPDF, pdf2image, OCR
- **Database:** PostgreSQL (optional for metadata), FAISS index
- **Deployment:** Streamlit Cloud

## 📥 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SCIAssist.git
   cd SCIAssist


Create and activate virtual environment:

python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

Install the required packages:

pip install -r requirements.txt

Run the Streamlit app:

    streamlit run app.py

🎯 How It Works

    Upload PDF: User uploads a scientific document via the Streamlit interface.

    Parse & Embed: Text and (some) visual content are extracted, converted into embeddings using LLaMA 3.2 and stored in a FAISS index.

    Query Handling: User submits a question; the backend searches the FAISS index for relevant chunks.

    Generate Answer: Retrieved content is fed into a Transformer model to generate an accurate, context-aware answer.

    Response Delivery: The final answer is shown in the UI with reference to the original document snippet.

🚧 Known Limitations

    🧠 LLaMA 3.2's visual embedding support may not fully interpret complex graphs/charts.

    🛠️ Future improvements include adding CLIP or Flamingo for better visual-text integration.

🔗 Live Demo

Try the deployed version here:
➡️ https://sciassistai-hl43wrrkuj62llbdcvhqev.streamlit.app/
