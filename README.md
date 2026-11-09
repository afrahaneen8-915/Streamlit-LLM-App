# AI PDF Question Answering System

## Project Overview
The AI PDF Question Answering System is an interactive web application built using Streamlit, LangChain, ChromaDB, and Ollama. It allows users to upload PDF documents and ask questions based on their content.

## Features
- Upload multiple PDF documents.
- Extract text from PDF files.
- Split extracted text into smaller chunks.
- Generate semantic embeddings using Sentence Transformers.
- Store embeddings in ChromaDB.
- Retrieve the top 5 relevant chunks for each question.
- Generate answers using Ollama with Llama 3.2.
- Maintain conversation history.
- Clear chat history.
- Switch between uploaded PDF documents.
- Display retrieved chunks and response generation time.

## Technologies Used
- Python
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers
- Ollama
- Llama 3.2
- PyPDF

## Project Structure
```text
Streamlit_LLM_App/
│
├── app.py
├── pdf_qa.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── sample.pdf
│   ├── queries.csv
│   ├── answers.csv
│   └── uploaded_files/
│
├── chroma_db/
│
└── assets/
    └── screenshots/
```

## Installation and Execution

1. Install Python and Ollama.
2. Download the Llama 3.2 model:

```bash
ollama pull llama3.2
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

5. Open the Streamlit application in your browser.
6. Upload PDF documents and start asking questions.

## Output
The application generates answers based on the uploaded PDF documents and displays the relevant retrieved text chunks along with response generation time.

## Conclusion
This project demonstrates how Large Language Models and Retrieval-Augmented Generation (RAG) can be used to build an interactive document question-answering application.
