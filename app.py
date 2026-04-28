from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from zhipuai import ZhipuAI

# ================= 🔧 基础配置 =================
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "你的真实Key")
CHAT_MODEL = "glm-4.7"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "clean_manual.md")

# ================= 📦 构建知识库 =================
@st.cache_resource
def build_knowledge_base(md_path, mtime):
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    md_chunks = md_splitter.split_text(markdown_text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = char_splitter.split_documents(md_chunks)

    embeddings_model = ZhipuAIEmbeddings(api_key=ZHIPU_API_KEY, model="embedding-3")
    vector_store = Chroma.from_documents(
        chunks,
        embeddings_model,
        persist_directory=os.path.join(BASE_DIR, "chroma_db"),
    )
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )
    return ensemble_retriever


# ================= 💬 检索与回答（不再重写问题） =================
def ask_question(retriever, current_question):
    client = ZhipuAI(api_key=ZHIPU_API_KEY, timeout=30.0, max_retries=1)

    st.sidebar.info(f"🔍 实际搜索内容：{current_question}")

    # ✅ 用传进来的 retriever
    docs = retriever.invoke(current_question)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = f"""
    你是一个大疆(DJI)产品的金牌客服。请严格根据下面提供的【参考文档】回答用户的【问题】。

    【参考文档】：
    {context}

    【问题】：
    {current_question}
    """

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 调用失败：{type(e).__name__}: {e}"


# ================= 🎨 前端 =================
st.set_page_config(page_title="Pocket 3 智能客服", page_icon="🚁")
st.title("🚁 大疆 Pocket 3 专属智能客服")

if not os.path.exists(MD_PATH):
    st.error(f"找不到 Markdown 文件：{MD_PATH}")
    st.stop()

mtime = os.path.getmtime(MD_PATH)
my_retriever = build_knowledge_base(MD_PATH, mtime)

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
            answer = ask_question(my_retriever, prompt)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})