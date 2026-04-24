from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from zhipuai import ZhipuAI

# ================= 🔧 基础配置 =================
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "你的真实Key")
CHAT_MODEL = "glm-4-air"   # 换成你账户实际支持的模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "test_doc.pdf")

# ================= 📦 构建知识库 =================
@st.cache_resource
def build_knowledge_base(pdf_path, mtime):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embeddings_model = ZhipuAIEmbeddings(api_key=ZHIPU_API_KEY, model="embedding-3")
    vector_store = Chroma.from_documents(
        chunks,
        embeddings_model,
        persist_directory=os.path.join(BASE_DIR, "chroma_db")
    )
    return vector_store

# ================= 💬 检索与回答 =================
def ask_question(vector_store, question):
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    你是一个大疆(DJI)产品的金牌客服。请严格根据下面提供的【参考文档】回答用户的【问题】。
    如果文档中没有相关信息，请直接回答"对不起，说明书中没有提及这部分内容"，不要自己捏造。

    【参考文档】：
    {context}

    【问题】：
    {question}
    """

    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

# ================= 🎨 前端 =================
st.set_page_config(page_title="Pocket 3 智能客服", page_icon="🚁")
st.title("🚁 大疆 Pocket 3 专属智能客服")

if not os.path.exists(PDF_PATH):
    st.error(f"找不到 PDF 文件：{PDF_PATH}")
    st.stop()

mtime = os.path.getmtime(PDF_PATH)
my_database = build_knowledge_base(PDF_PATH, mtime)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("你想查阅关于 Pocket 3 的什么信息？（例如：电量低会自动关机吗？）"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("AI 客服正在疯狂翻阅说明书..."):
            answer = ask_question(my_database, prompt)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})