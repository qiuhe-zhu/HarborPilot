# 🚢 HarborPilot 航运智能助手

基于 C++/Python 的航运/海事场景 RAG (检索增强生成) 智能问答系统。

## 📖 项目简介 (Problem)
航运、海事法规和港口公开信息分散，大语言模型直接回答容易产生幻觉。HarborPilot 通过 RAG 架构，构建了一个可以**精准引用公开资料**的智能助手，并提供了完整的本地评测闭环。

## 🏗️ 系统架构 (Architecture)
- **前端**: Streamlit (聊天交互 + 引用来源展示)
- **API 网关**: FastAPI (接收请求、拼装上下文)
- **RAG 核心**: 向量检索 (Qdrant Local) + Prompt 约束防幻觉
- **Embedding**: 硅基流动 `BAAI/bge-m3` (1024维)
- **LLM**: DeepSeek API (`deepseek-chat`)

## 🚀 快速开始 (Quickstart)

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv .venv
# Windows 激活: .\.venv\Scripts\Activate.ps1
# Mac/Linux 激活: source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
