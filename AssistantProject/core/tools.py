# core/tools.py
import os
import subprocess
import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    if os.getenv("TAVILY_API_KEY"):
        tavily_search = TavilySearch()
    else:
        tavily_search = None
except Exception:
    tavily_search = None

@tool
def fetch_url(url: str) -> str:
    """抓取网页内容。"""
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]): tag.decompose()
        return "\n".join([line for line in soup.get_text(separator="\n", strip=True).splitlines() if line.strip()])[:8000]
    except Exception as e: return f"抓取失败: {e}"


@tool
def bash_execute(command: str) -> str:
    """
    【系统工具：执行本地 shell 命令】
    执行基础终端命令。
    【🔴 严厉警告】：
    1. 绝对禁止使用 `echo` 等命令去“伪造”、“Mock”或“欺骗”任务输出！如果你无法真实执行某个测试，请直接告诉用户环境不支持。
    2. 禁止执行会导致阻塞的长服务或后台服务（如 nohup python ... &），本环境不支持开启 Web 服务器。
    """
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
    except Exception as e:
        return f"执行失败: {str(e)}"

@tool
def read_local_file(file_path: str) -> str:
    """
    【系统工具：读取本地文件】
    当需要阅读某个本地代码、Markdown 或配置文件时使用。
    必须传入相对于项目根目录的相对路径，例如：skills/get-time/references/api_doc.md
    """
    safe_path = os.path.abspath(os.path.join(PROJECT_ROOT, file_path))
    if not safe_path.startswith(PROJECT_ROOT): return f"⚠️ 越权拒绝。"
    try:
        with open(safe_path, 'r', encoding='utf-8') as f: return f.read()
    except Exception as e: return f"❌ 读取失败：{e}"


@tool
def execute_python_script(script_path: str, args: str = "") -> str:
    """
    【系统核心工具：执行本地 Python 脚本】
    🔴 严厉警告：绝对禁止使用此工具运行包含 GUI 界面（如 tkinter, PyQt）或需要控制台交互（如 input()）的脚本！
    如果必须运行，请先检查环境是否支持。
    """
    safe_path = os.path.abspath(os.path.join(PROJECT_ROOT, script_path))
    if not safe_path.startswith(PROJECT_ROOT):
        return f"⚠️ 权限拒绝：禁止执行项目外部脚本。"

    if not os.path.exists(safe_path):
        return f"❌ 脚本未找到：{script_path}"

    # ==========================================
    # 【智能拦截器】：扫描代码内容，防止卡死后台
    # ==========================================
    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
            if "input(" in code_content or "tkinter" in code_content or "PyQt" in code_content:
                return "⚠️ [系统拦截] 检测到脚本包含交互输入(input)或图形界面(GUI)。当前沙盒环境无显示器且无法提供键盘输入。请直接告诉用户代码已生成完毕，并让用户在自己的终端里手动执行命令：`python " + script_path + "`"
    except Exception:
        pass

    try:
        print(f"🤖 [系统工具调用] 正在执行脚本: {script_path} {args}")
        cmd = ["python", safe_path]
        if args:
            cmd.extend(args.split())

        # 将超时时间降至 15 秒，防止线程长期阻塞导致前端崩溃
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        output = ""
        if result.stdout:
            output += f"【STDOUT】\n{result.stdout}\n"
        if result.stderr:
            output += f"【STDERR】\n{result.stderr}\n"

        if result.returncode == 0:
            return f"✅ 执行成功！\n{output}"
        else:
            return f"⚠️ 执行异常 (退出码 {result.returncode})：\n{output}"

    except subprocess.TimeoutExpired:
        return f"⏳ 脚本执行超时 (超过 15 秒被强制终止)。这通常是因为脚本陷入了死循环或在等待操作。"
    except Exception as e:
        return f"❌ 脚本执行失败：{str(e)}"


@tool
def write_local_file(file_path: str, content: str) -> str:
    """
    【核心工具：写入本地文件 (OpenCode)】
    当你需要创建新技能包(SKILL.md)、编写项目代码、保存文件时，直接调用此工具。
    必须传入相对于项目根目录的相对路径，例如：skills/db-optimizer/SKILL.md
    如果文件夹不存在，系统会自动为你创建！
    """
    safe_path = os.path.abspath(os.path.join(PROJECT_ROOT, file_path))
    if not safe_path.startswith(PROJECT_ROOT): return f"⚠️ 越权拒绝。"
    try:
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"✅ 文件已成功写入物理路径：{file_path}"
    except Exception as e:
        return f"❌ 文件写入失败：{str(e)}"