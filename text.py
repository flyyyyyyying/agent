import os
import gradio as gr
import requests
import io
import sys
import builtins
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
import pdfplumber

load_dotenv(".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
QWHEATHER_KEY = os.getenv("QWHEATHER_KEY")
QW_HOST = os.getenv("QW_HOST")
BASE_URL = "https://api.deepseek.com"

# LLM 继续使用 DeepSeek API
llm = ChatOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0
)

# 嵌入模型：Ollama本地 nomic-embed-text
embeddings = None
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("✅ Ollama Embedding 连接成功")
except Exception as e:
    print(f"❌ Ollama Embedding 初始化失败：{e}")

FAISS_INDEX_DIR = "./faiss_index"
vector_db = None
retriever = None
tool_log = ""
memory = MemorySaver()
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# ========== 天气工具 ==========
@tool
def get_weather(city: str) -> str:
    """查询城市未来3天天气，用户问天气的时候调用这个工具
    Args:
        city: 城市名称，例如 金坛、常州、西安
    """
    global tool_log
    try:
        headers = {"X-QW-Api-Key": QWHEATHER_KEY}
        loc_url = f"https://{QW_HOST}/geo/v2/city/lookup?location={city}"
        loc_resp = requests.get(loc_url, headers=headers, timeout=8)
        if loc_resp.status_code != 200:
            return f"城市查询失败，http码：{loc_resp.status_code}"
        loc_data = loc_resp.json()
        if loc_data.get("code") != "200":
            return f"找不到城市：{city}，code={loc_data.get('code')}"
        loc_id = loc_data["location"][0]["id"]

        weather_url = f"https://{QW_HOST}/v7/weather/3d?location={loc_id}"
        w_resp = requests.get(weather_url, headers=headers, timeout=8)
        if w_resp.status_code != 200:
            return f"天气接口请求失败，http码：{w_resp.status_code}"
        w_data = w_resp.json()
        if w_data.get("code") != "200":
            return f"天气查询失败，code={w_data.get('code')}"

        daily_list = w_data.get("daily", [])
        if not daily_list:
            return "没有获取到天气预报数据"
        res = []
        for day in daily_list:
            fxDate = day.get("fxDate")
            textDay = day.get("textDay")
            tempMin = day.get("tempMin")
            tempMax = day.get("tempMax")
            res.append(f"{fxDate}：{textDay}，{tempMin}℃ ~ {tempMax}℃")
        out = f"{city}未来三天天气：" + "；".join(res)
        tool_log += f"\n【调用工具：get_weather】\n城市：{city}\n返回：{out}\n-------------------------\n"
        return out
    except Exception as e:
        err = f"天气查询异常：{str(e)}"
        tool_log += f"\n【调用工具：get_weather】\n城市：{city}\n报错：{err}\n-------------------------\n"
        return err

# ========== Python代码沙盒 ==========
@tool
def python_code_runner(code: str) -> str:
    """
    执行Python代码，用于复杂数学计算、数据统计、算法验证等任务。
    Args:
        code: 完整可运行的Python代码，结果尽量存入result变量方便读取
    """
    global tool_log
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    allowed_builtins = {
        "abs", "sum", "max", "min", "range", "len",
        "list", "dict", "float", "int", "round", "pow",
        "sorted", "zip", "enumerate"
    }
    safe_builtins = {}
    for name in allowed_builtins:
        if hasattr(builtins, name):
            safe_builtins[name] = getattr(builtins, name)

    safe_globals = {"__builtins__": safe_builtins}
    safe_locals = {"result": None}

    try:
        exec(code, safe_globals, safe_locals)
        output = new_stdout.getvalue()
        res_val = safe_locals.get("result")
        msg = []
        if output.strip():
            msg.append(f"打印输出：\n{output}")
        if res_val is not None:
            msg.append(f"result变量值：{res_val}")
        if not msg:
            msg.append("代码执行完成，无输出")
        ret = "\n".join(msg)
        tool_log += f"\n【调用工具：python_code_runner】\n=====代码=====\n{code}\n=====结果=====\n{ret}\n-------------------------\n"
        return ret
    except Exception as e:
        err_msg = f"代码执行异常：{str(e)}"
        tool_log += f"\n【调用工具：python_code_runner】\n=====代码=====\n{code}\n=====报错=====\n{err_msg}\n-------------------------\n"
        return err_msg
    finally:
        sys.stdout = old_stdout

# ========== 翻译工具 ==========
@tool
def translate(text: str, target_lang: str) -> str:
    """文本翻译工具，支持中文、英文、日语、韩语互译
    Args:
        text: 需要翻译的原文
        target_lang: 目标语言，例如：中文 / English / 日语 / 韩语
    """
    global tool_log
    translate_llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=BASE_URL,
        model="deepseek-chat",
        temperature=0
    )
    resp = translate_llm.invoke(f"""你是专业翻译，只输出翻译结果，不要多余解释。
原文：{text}
翻译成：{target_lang}""")
    out = resp.content
    tool_log += f"\n【调用工具：translate】\n原文：{text}\n目标语言：{target_lang}\n翻译结果：{out}\n-------------------------\n"
    return out

tools = [get_weather, python_code_runner, translate]

# ========== PDF RAG加载 【FAISS + pdfplumber】 ==========
def load_pdf_to_vector(file_path: str):
    global vector_db, retriever
    if embeddings is None:
        return "❌ Ollama Embedding 未就绪，无法加载PDF，请先启动Ollama"
    try:
        docs = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(page_content=text))
        if len(docs) == 0:
            return "⚠️ 没有提取到任何文字，PDF是图片扫描版，无法RAG"

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        # FAISS构建向量库
        vector_db = FAISS.from_documents(split_docs, embeddings)
        vector_db.save_local(FAISS_INDEX_DIR)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        return "✅ PDF加载完成！FAISS向量库构建成功，可以问答简历内容"
    except Exception as e:
        return f"❌ 加载失败：{str(e)}"

system_prompt = """你是集成私有知识库、天气查询、Python代码执行、文本翻译的综合智能Agent。
规则：
1. 如果用户的问题和已经上传的PDF文档内容相关，优先使用检索到的文档上下文作为依据回答；
2. 查天气调用get_weather；需要计算、统计、算法验证调用python_code_runner；翻译文本调用translate；
3. 如果只是单纯问天气、单纯计算、单纯翻译，不需要强行关联文档；
4. 如果知识库不可用，直接告知用户无法读取PDF；
5. 文档里没有的信息不要编造；
上下文：{context}
"""

agent = create_agent(llm, tools, checkpointer=memory)

def chat(user_query: str):
    global tool_log
    context = ""
    skip_keywords = ["天气", "计算", "等于", "+", "-", "*", "/", "翻译", "译成"]
    if retriever is not None and embeddings is not None:
        need_rag = not any(k in user_query for k in skip_keywords)
        if need_rag:
            docs = retriever.invoke(user_query)
            context = "\n".join([d.page_content for d in docs])
    tool_log = ""
    resp = agent.invoke(
        {"messages": [("system", system_prompt.format(context=context)), ("human", user_query)]},
        config=config
    )
    ans = resp["messages"][-1].content
    return ans, tool_log

# 清空会话（重置记忆+清空日志）
def clear_session():
    global thread_id, config, tool_log
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    tool_log = ""
    return "", "", "✅ 会话已清空，记忆全部清除"

# ========== Gradio界面 ==========
with gr.Blocks(title="综合智能Agent平台（FAISS稳定版）") as demo:
    gr.Markdown("# 🧰 综合智能Agent平台 | RAG + 对话记忆 + 代码沙盒 + 天气 + 翻译")
    gr.Markdown("""
    ✅ 功能：🧠对话记忆、📄PDF私有知识库RAG（Ollama本地Embedding + FAISS）、🧪Python代码沙盒、🌤天气查询、🌐翻译
    💡 记忆测试示例：
    1. 帮我写一个求最长递增子序列的Python代码，放到result变量
    2. 把它改成二分优化版本
    """)
    with gr.Row():
        pdf = gr.File(label="上传PDF私有知识库（可选）", type="filepath")
        status = gr.Textbox(label="知识库状态", interactive=False)
        clear_btn = gr.Button("🗑️ 清空会话（清除记忆+日志）")
    pdf.upload(load_pdf_to_vector, [pdf], [status])

    q = gr.Textbox(label="提问", placeholder="示例：帮我写LIS代码 / 分析这份简历 / 常州天气")
    btn = gr.Button("提交")
    ans = gr.Markdown(label="最终回答")
    log_box = gr.Textbox(label="🔍 工具调用日志（看源码、调用记录）", lines=12, interactive=False)

    btn.click(chat, [q], [ans, log_box])
    clear_btn.click(clear_session, outputs=[q, ans, log_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
