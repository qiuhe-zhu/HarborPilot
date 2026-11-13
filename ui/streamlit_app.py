import streamlit as st
import httpx

st.set_page_config(page_title="HarborPilot 航运智能助手", page_icon="🚢")
st.title("🚢 HarborPilot 航运智能助手")
st.caption("基于 RAG + 本地 Qdrant 的智能问答 Demo")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 查看引用来源"):
                for s in msg["sources"]:
                    st.write(f"- {s['source']} (相似度: {s['score']})")

if prompt := st.chat_input("请输入您的问题（例如：洋山港区在哪里？）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索资料并思考..."):
            try:
                response = httpx.post(
                    "http://localhost:8000/ask",
                    json={"question": prompt},
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                
                answer = data["answer"]
                sources = data["sources"]
                latency = data["latency_ms"]
                
                st.markdown(answer)
                st.caption(f"⏱️ 耗时 {latency} ms | 🆔 Trace: {data['trace_id'][:8]}")
                
                with st.expander("📚 查看引用来源"):
                    for s in sources:
                        st.write(f"- {s['source']} (相似度: {s['score']})")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"请求失败，请检查 FastAPI 服务是否正在运行。错误信息：{e}")
