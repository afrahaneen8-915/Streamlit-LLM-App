
import streamlit as st
import os
import time
import hashlib

from pathlib import Path
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Question Answering System")
st.write("Upload your PDF and ask questions to get intelligent answers.")

os.makedirs("dataset/uploaded_files", exist_ok=True)
os.makedirs("chroma_db", exist_ok=True)


# LOAD EMBEDDING MODEL
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )


# LOAD OLLAMA MODEL
@st.cache_resource
def load_llm():
    return ChatOllama(
        model="llama3.2:latest",
        temperature=0
    )


# SESSION STATE
if "documents" not in st.session_state:
    st.session_state.documents = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# SIDEBAR
st.sidebar.title("⚙️ PDF Settings")
st.sidebar.write("Upload and manage your documents.")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Process PDF Documents"):

    if not uploaded_files:
        st.sidebar.warning("Please upload a PDF first.")

    else:
        embeddings = load_embeddings()

        for uploaded_file in uploaded_files:
            try:
                pdf_bytes = uploaded_file.getvalue()

                if not pdf_bytes.startswith(b"%PDF"):
                    st.sidebar.error("Invalid PDF file.")
                    continue

                file_id = hashlib.sha256(
                    pdf_bytes
                ).hexdigest()[:12]

                if file_id in st.session_state.documents:
                    st.sidebar.info(
                        f"{uploaded_file.name} is already processed."
                    )
                    continue

                safe_name = Path(uploaded_file.name).name

                file_path = os.path.join(
                    "dataset",
                    "uploaded_files",
                    f"{file_id}_{safe_name}"
                )

                with open(file_path, "wb") as file:
                    file.write(pdf_bytes)

                reader = PdfReader(file_path)

                documents = []

                for page_number, page in enumerate(
                    reader.pages, start=1
                ):
                    page_text = page.extract_text() or ""

                    if page_text.strip():
                        documents.append(
                            Document(
                                page_content=page_text,
                                metadata={
                                    "source": uploaded_file.name,
                                    "page": page_number
                                }
                            )
                        )

                if not documents:
                    st.sidebar.error(
                        f"No extractable text in {uploaded_file.name}."
                    )
                    continue

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150
                )

                chunks = splitter.split_documents(documents)

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=f"pdf_{file_id}",
                    persist_directory=os.path.join(
                        "chroma_db", file_id
                    )
                )

                st.session_state.documents[file_id] = {
                    "name": uploaded_file.name,
                    "pages": len(reader.pages),
                    "chunks": len(chunks),
                    "vectorstore": vectorstore
                }

                st.sidebar.success(
                    f"Processed: {uploaded_file.name}"
                )

            except Exception as error:
                st.sidebar.error(
                    f"Processing error: {error}"
                )


# DOCUMENT SELECTION
if st.session_state.documents:

    st.sidebar.divider()
    st.sidebar.subheader("📂 Available Documents")

    document_ids = list(st.session_state.documents.keys())

    selected_id = st.sidebar.selectbox(
        "Choose a PDF",
        document_ids,
        format_func=lambda x:
            st.session_state.documents[x]["name"]
    )

    selected_document = st.session_state.documents[selected_id]

    st.sidebar.write(
        f"Pages: {selected_document['pages']}"
    )

    st.sidebar.write(
        f"Total chunks: {selected_document['chunks']}"
    )

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.subheader(f"📄 {selected_document['name']}")

    st.write(
        f"Total pages: {selected_document['pages']} | "
        f"Total chunks: {selected_document['chunks']}"
    )

    # DISPLAY CHAT HISTORY
    for chat in st.session_state.chat_history:

        if chat["document_id"] == selected_id:

            with st.chat_message("user"):
                st.write(chat["question"])

            with st.chat_message("assistant"):
                st.write(chat["answer"])

                st.caption(
                    f"Response time: {chat['time']:.2f} seconds"
                )

                with st.expander("View Retrieved Chunks"):
                    for index, chunk in enumerate(
                        chat["chunks"], start=1
                    ):
                        st.markdown(f"**Chunk {index}**")
                        st.write(chunk.page_content)

                        st.caption(
                            f"Page: {chunk.metadata.get('page', 'N/A')}"
                        )

    # QUESTION INPUT
    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching document and generating answer..."
            ):

                try:
                    start_time = time.perf_counter()

                    vectorstore = selected_document["vectorstore"]

                    retrieved_chunks = (
                        vectorstore.similarity_search(
                            question,
                            k=5
                        )
                    )

                    context = "\n\n".join(
                        [
                            f"Page {chunk.metadata.get('page', 'N/A')}:\n"
                            f"{chunk.page_content}"
                            for chunk in retrieved_chunks
                        ]
                    )

                    llm = load_llm()

                    prompt = f"""
You are a helpful PDF question answering assistant.

Answer using ONLY the information in the document context.

If the answer is not available in the context,
clearly say that the document does not provide enough information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear and simple answer.
"""

                    response = llm.invoke(prompt)

                    answer = response.content

                    elapsed_time = (
                        time.perf_counter() - start_time
                    )

                    st.write(answer)

                    st.caption(
                        f"⏱️ Response generation time: "
                        f"{elapsed_time:.2f} seconds"
                    )

                    with st.expander("View Retrieved Chunks"):
                        for index, chunk in enumerate(
                            retrieved_chunks, start=1
                        ):
                            st.markdown(f"**Chunk {index}**")
                            st.write(chunk.page_content)

                            st.caption(
                                f"Page: {chunk.metadata.get('page', 'N/A')}"
                            )

                    st.session_state.chat_history.append({
                        "document_id": selected_id,
                        "question": question,
                        "answer": answer,
                        "time": elapsed_time,
                        "chunks": retrieved_chunks
                    })

                except Exception as error:
                    st.error(f"Error generating answer: {error}")

else:

    st.info(
        "👈 Upload a PDF from the sidebar and click "
        "'Process PDF Documents' to get started."
    )

    st.markdown("""
    ### ✨ Application Features

    - Upload multiple PDF documents.
    - Extract PDF text.
    - Generate semantic embeddings.
    - Store embeddings in ChromaDB.
    - Retrieve the top 5 relevant chunks.
    - Generate answers using Ollama and Llama 3.2.
    - View conversation history.
    - Clear chat history.
    - Display response generation time.
    """)
