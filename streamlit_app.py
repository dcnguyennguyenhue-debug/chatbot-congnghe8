import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

st.title("Chatbot môn Công Nghệ 8")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATA_PATH = "data"

@st.cache_resource
def load_db():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    return db

db = load_db()

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

question = st.text_input("Nhập câu hỏi")

if question:

    docs = db.similarity_search(question, k=3)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Bạn là trợ lý môn Công Nghệ 8.

    Chỉ trả lời dựa trên nội dung sau:
    {context}

    Nếu không có trong nội dung trên, trả lời:
    'Nội dung này không có trong tài liệu môn học.'

    Câu hỏi: {question}
    """

    answer = llm.predict(prompt)

    st.write(answer)
