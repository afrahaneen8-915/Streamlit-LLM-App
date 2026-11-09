
import os
import shutil
import pandas as pd

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama


# ---------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------

PDF_PATH = "dataset/sample.pdf"
QUERY_PATH = "dataset/queries.csv"
CHROMA_PATH = "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:latest"

print("PDF QUESTION ANSWERING SYSTEM")
print("--------------------------------")


# ---------------------------------------
# 2. EXTRACT TEXT FROM PDF
# ---------------------------------------

print("\nStep 1: Extracting PDF text...")

reader = PdfReader(PDF_PATH)

page_documents = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    if text.strip():
        page_documents.append(
            Document(
                page_content=text,
                metadata={"page": page_number}
            )
        )

if not page_documents:
    raise ValueError("No extractable text found in the PDF.")

extracted_text = "\n\n".join(
    doc.page_content for doc in page_documents
)

with open("dataset/extracted_text.txt", "w", encoding="utf-8") as file:
    file.write(extracted_text)

print("PDF pages:", len(reader.pages))
print("Pages extracted:", len(page_documents))


# ---------------------------------------
# 3. SPLIT TEXT INTO CHUNKS
# ---------------------------------------

print("\nStep 2: Splitting text into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(page_documents)

for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"chunk_{index + 1}"

print("Total chunks:", len(chunks))

chunks_df = pd.DataFrame([
    {
        "Chunk_ID": chunk.metadata["chunk_id"],
        "Page": chunk.metadata["page"],
        "Text": chunk.page_content
    }
    for chunk in chunks
])

chunks_df.to_csv("dataset/chunks.csv", index=False)


# ---------------------------------------
# 4. GENERATE EMBEDDINGS
# ---------------------------------------

print("\nStep 3: Loading embedding model...")

embedding_model = SentenceTransformer(MODEL_NAME)

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding model loaded successfully!")

# Generate embeddings to verify the model.
sample_embedding = embedding_model.encode(chunks[0].page_content)

print("Embedding dimension:", len(sample_embedding))


# ---------------------------------------
# 5. STORE EMBEDDINGS IN CHROMADB
# ---------------------------------------

print("\nStep 4: Creating ChromaDB vector store...")

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="pdf_education_collection",
    persist_directory=CHROMA_PATH
)

print("Documents stored in ChromaDB:", len(chunks))


# ---------------------------------------
# 6. LOAD OLLAMA MODEL
# ---------------------------------------

print("\nStep 5: Connecting to Ollama...")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0
)

print("Ollama model connected:", OLLAMA_MODEL)


# ---------------------------------------
# 7. LOAD USER QUESTIONS
# ---------------------------------------

print("\nStep 6: Loading questions...")

queries = pd.read_csv(QUERY_PATH)

print("Total questions:", len(queries))


# ---------------------------------------
# 8. RETRIEVE TOP 5 CHUNKS
# ---------------------------------------

print("\nStep 7: Performing question answering...")

results = []

for index, row in queries.iterrows():

    query_id = row["Query_ID"]
    question = row["Question"]

    print(f"\nProcessing {query_id}: {question}")

    retrieved_docs = vectorstore.similarity_search_with_score(
        question,
        k=5
    )

    retrieved_texts = []
    retrieved_scores = []

    for document, distance in retrieved_docs:
        retrieved_texts.append(
            document.page_content
        )
        retrieved_scores.append(
            round(float(distance), 4)
        )

    context = "\n\n".join(retrieved_texts)

    # -----------------------------------
    # 9. ANSWER WITH RETRIEVED CONTEXT
    # -----------------------------------

    rag_prompt = f"""
You are an educational question answering assistant.

Answer the question using ONLY the information
provided in the context below.

If the answer is not available in the context,
clearly say that the PDF does not provide enough
information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    rag_response = llm.invoke(rag_prompt)
    answer_with_retrieval = rag_response.content

    # -----------------------------------
    # 10. ANSWER WITHOUT RETRIEVAL
    # -----------------------------------

    normal_prompt = f"""
Answer the following question using your
general knowledge.

QUESTION:
{question}

ANSWER:
"""

    normal_response = llm.invoke(normal_prompt)
    answer_without_retrieval = normal_response.content

    results.append({
        "Query_ID": query_id,
        "Question": question,
        "Retrieved_Chunks": "\n\n".join(retrieved_texts),
        "Retrieval_Scores": str(retrieved_scores),
        "Answer_With_Retrieval": answer_with_retrieval,
        "Answer_Without_Retrieval": answer_without_retrieval
    })

    print("Top chunks retrieved:", len(retrieved_docs))
    print("RAG answer generated successfully!")


# ---------------------------------------
# 11. SAVE ANSWERS TO CSV
# ---------------------------------------

print("\nStep 8: Saving results...")

answers_df = pd.DataFrame(results)

answers_df.to_csv(
    "dataset/answers.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n--------------------------------")
print("PDF QUESTION ANSWERING COMPLETED")
print("--------------------------------")

print("PDF pages:", len(reader.pages))
print("Total chunks:", len(chunks))
print("Total questions:", len(queries))
print("Answers generated:", len(results))
print("Top 5 chunks retrieved per question")

print("\nFiles generated:")
print("1. dataset/extracted_text.txt")
print("2. dataset/chunks.csv")
print("3. dataset/answers.csv")
print("4. chroma_db/")

print("\nProject executed successfully!")
