import os
from concurrent.futures import ThreadPoolExecutor
import torch
from tqdm import tqdm
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CURR_DIR = os.path.dirname(__file__)
PDF_DIR = os.path.join(CURR_DIR , "Data" , "PDFs")
VECTORSTORE = os.path.join(CURR_DIR , "Vectorstores" , "QuestionPapers")


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings" : True , "batch_size" : 64})


db = Chroma(
    persist_directory=VECTORSTORE,
    embedding_function=embeddings
)


existing = set()

try:
    data = db.get()
    for md in data["metadatas"]:
        existing.add(md["source"])
except:
    pass


def load_single_pdf(pdf):

    if not pdf.lower().endswith(".pdf"):
        return None
    if pdf in existing:
        return None
    
    path = os.path.join(PDF_DIR, pdf)

    try:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return Document(
            page_content="\n".join(pages),
            metadata={
                "source": pdf
            }
        )

    except Exception as e:
        print(f"Skipping {pdf}")
        print(e)
        return None


def load_pdfs():
    pdfs = os.listdir(PDF_DIR)
    docs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        iterator = executor.map(load_single_pdf , pdfs)
        for doc in tqdm(iterator , total=len(pdfs) , desc="Loading PDFs"):
            if doc is not None:
                docs.append(doc)
    return docs


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250
)


def split_documents(documents):
    chunks = splitter.split_documents(documents)
    return chunks


def add_chunks(chunks):
    BATCH = 128
    for i in tqdm(range(0, len(chunks) , BATCH) , desc="Embedding"):
        batch = chunks[i:i+BATCH]
        db.add_documents(batch)





if __name__ == "__main__":
    print("Loading PDFs")
    docs = load_pdfs()
    print(f"\nLoaded {len(docs)} new PDFs.")

    print("Chunking")
    chunks = split_documents(docs)
    print(f"\nCreated {len(chunks)} chunks.")

    print("Embedding")
    add_chunks(chunks)
    print("Finished")