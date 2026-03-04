from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DATA_PATH = "data/"

def build_db():
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

db = build_db()
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

@app.route("/")
def home():
    return """
    <h2>Chatbot nội bộ môn Công Nghệ 8</h2>
    <input id='msg' style='width:70%'>
    <button onclick='send()'>Gửi</button>
    <div id='chat'></div>
    <script>
    async function send(){
        let m = document.getElementById('msg').value;
        let r = await fetch('/chat',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({message:m})
        });
        let d = await r.json();
        document.getElementById('chat').innerHTML += "<p><b>HS:</b>"+m+"</p><p><b>Bot:</b>"+d.answer+"</p>";
    }
    </script>
    """

@app.route("/chat", methods=["POST"])
def chat():
    question = request.json["message"]

    docs = db.similarity_search(question, k=3)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Bạn là trợ lý môn Công Nghệ 8.
    Chỉ trả lời dựa trên nội dung sau:

    {context}

    Nếu không có trong nội dung trên, trả lời:
    "Nội dung này không có trong tài liệu môn học."

    Câu hỏi: {question}
    """

    answer = llm.predict(prompt)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run()
