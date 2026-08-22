# frontend/app.py —— SmartKB 2.0 知识库问答前端
import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"   # 后端地址

st.set_page_config(page_title="SmartKB 2.0", page_icon="📚")
st.title("📚 SmartKB 2.0 知识库问答系统")
st.caption("基于本地大模型（Ollama + Qwen2.5-7B）的 RAG 问答")

# 输入框
question = st.text_input("请输入你的问题：", placeholder="例如：怎么微调大模型？")

# 提问按钮
if st.button("提问", type="primary"):
    if not question.strip():
        st.warning("请输入问题后再提问")
    else:
        with st.spinner("正在检索知识库并生成回答..."):
            try:
                resp = requests.post(API_URL, json={"question": question}, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.subheader("💬 回答")
                st.write(data["answer"])
                                st.subheader("📎 引用来源")
                for i, cite in enumerate(data["citations"], 1):
                    cite_short = cite if len(cite) <= 100 else cite[:100] + "..."
                    st.markdown(f"{i}. {cite_short}")
            except Exception as e:
                st.error(f"请求失败：{e}")
                st.info("请检查后端服务是否运行正常")
