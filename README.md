# 基于RAG的多工具智能Agent问答系统
> 基于Gradio开发的交互式智能Agent Web平台

## ✨ 功能介绍
- 📄 RAG私有文档问答：Ollama本地部署文本向量化模型，支持上传PDF知识库检索问答
- 🧠 对话记忆：LangGraph维护多轮上下文会话
- 🧪 安全Python代码沙盒：白名单内置函数，支持算法验证、数值计算
- 🌤 天气查询：对接和风天气API
- 🌐 文本翻译：调用DeepSeek大模型翻译
- 📝 工具调用日志面板：可视化Agent工具调用链路，支持一键重置会话

## 🛠 技术栈
Python + LangChain + LangGraph + Gradio + Ollama + Requests（RAG、多API集成）

## 🚀 运行前置
1. 安装 Ollama，拉取 `nomic-embed-text` 模型
2. 准备DeepSeek API Key、和风天气Key，配置到key.env
3. 安装依赖
```bash
pip install -r requirements.txt
