from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from zhipuai import ZhipuAI

# ================= 🔧 基础配置 =================
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "你的真实Key")
CHAT_MODEL = "glm-4.7"   # 换成你账户实际支持的模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "clean_manual.md")

# ================= 📦 构建知识库 =================
@st.cache_resource
def build_knowledge_base(mtime):
    loader = TextLoader(MD_PATH, encoding="utf-8")
    docs = loader.load()

    text_splitter = MarkdownHeaderTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embeddings_model = ZhipuAIEmbeddings(api_key=ZHIPU_API_KEY, model="embedding-3")
    vector_store = Chroma.from_documents(
        chunks,
        embeddings_model,
        persist_directory=os.path.join(BASE_DIR, "chroma_db")
    )
    return vector_store

def get_standalone_question(client, history, current_question):
    """结合历史记录，把含糊的追问改写成独立的问题"""
    if not history:
        return current_question
    
    # 将历史记录格式化为文本
    history_text = ""
    for msg in history[-5:]: # 只取最近5轮，防止太长
        role = "用户" if msg["role"] == "user" else "AI"
        history_text += f"{role}: {msg['content']}\n"

    rewrite_prompt = f"""
    根据以下对话历史和用户的新问题，将其改写为一个不需要上下文也能听懂的【独立完整问题】。
    
    【对话历史】：
    {history_text}
    
    【用户新问题】：{current_question}
    
    请直接输出改写后的问题，不要说任何废话。
    """
    
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

# ================= 💬 检索与回答 =================
def ask_question(vector_store, history, current_question):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    
    # 1. 【新增】重写问题，处理“它”、“那这个”等代词
    standalone_q = get_standalone_question(client, history, current_question)
    st.sidebar.info(f"🔍 实际搜索内容：{standalone_q}") # 在侧边栏显示，方便调试观察

    # 2. 用重写后的问题去搜数据库
    docs = vector_store.similarity_search(standalone_q, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. 组装最终回答的 Prompt
    final_prompt = f"""
    你是一个大疆(DJI)产品的金牌客服。请严格根据下面提供的【参考文档】回答用户的【问题】。
    
    【参考文档】：
    {context}

    【问题】：
    {standalone_q}
    """

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

# ================= 🎨 前端 =================
st.set_page_config(page_title="Pocket 3 智能客服", page_icon="🚁")
st.title("🚁 大疆 Pocket 3 专属智能客服")

if not os.path.exists(MD_PATH):
    st.error(f"找不到 PDF 文件：{MD_PATH}")
    st.stop()

mtime = os.path.getmtime(MD_PATH)
my_database = build_knowledge_base(MD_PATH, mtime)

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
            history_for_rewrite = st.session_state.messages[:-1]
            answer = ask_question(my_database, history_for_rewrite, prompt)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})