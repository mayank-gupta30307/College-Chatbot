# 🎓 College Chatbot

This project is an AI-powered chatbot designed to answer questions about a college using Retrieval-Augmented Generation (RAG). Instead of relying only on an LLM's knowledge, it searches through college documents and uses the relevant information to generate accurate, context-aware responses.

The goal of this project was to build a chatbot that can assist students by answering questions related to academics, admissions, fees, contact information, regulations, question papers, and other college resources.

---

## Features

* 💬 Natural language conversations
* 📄 Answers based on college documents
* 🔍 Semantic search using vector embeddings
* 🤖 Google Gemini for response generation
* 🧠 LangGraph-based workflow
* 🎤 Voice input support
* 🔊 Text-to-Speech responses
* 🌐 Simple Streamlit interface

---

## Built With

* Python
* Streamlit
* LangGraph
* LangChain
* Google Gemini
* ChromaDB
* Hugging Face Embeddings
* PyTorch
* BeautifulSoup
* PyPDF
* SpeechRecognition
* Edge-TTS

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/mayank-gupta30307/College-Chatbot.git
cd College-Chatbot
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project directory.

```env
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

---

## Running the Project

```bash
streamlit run app.py
```

Once the application starts, open the URL displayed in your terminal.

---

## How It Works

1. College documents are collected and processed.
2. Documents are split into smaller chunks.
3. Embeddings are generated using Hugging Face models.
4. The embeddings are stored in ChromaDB.
5. When a user asks a question, the most relevant chunks are retrieved.
6. Gemini uses those chunks as context to generate the final response.
7. The chatbot displays the answer and can also read it aloud.

---

## Project Structure

```
College-Chatbot/
│
├── app.py
├── RAG.py
├── tts.py
├── requirements.txt
├── .env
│
├── Data/
├── Vectorstores/
└── README.md
```

## Author

**Mayank Gupta**

