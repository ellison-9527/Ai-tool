# core/tools.py
import os
import subprocess
import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

@tool
def fetch_url(url: str) -> str:
    """抓取指定 URL 网页内容并返回纯文本。用于获取在线文档、文章、博客等网页内容。"""
    try:
        # 使用同步的 httpx 保持简单健壮
        resp = httpx.get(
            url,
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.splitlines() if line.strip()]
        return "\n".join(lines)[:8000]
    except Exception as e:
        return f"抓取失败: {str(e)}"

@tool
def bash_execute(command: str) -> str:
    """执行本地 shell 命令并返回输出。可运行 python, node, ls 等。"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if not output.strip():
            output = f"(命令已执行，退出码: {result.returncode})"
        return output[:8000]
    except subprocess.TimeoutExpired:
        return "执行超时（120秒限制）"
    except Exception as e:
        return f"执行错误: {e}"

# 初始化 Tavily 联网搜索工具
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_search = None
if tavily_api_key:
    tavily_search = TavilySearch(max_results=3, api_key=tavily_api_key)