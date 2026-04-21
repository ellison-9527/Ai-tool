# core/tools.py
import os
import subprocess
import httpx
import sys
import asyncio
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from AssistantProject.core.logger import logger

load_dotenv()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    if os.getenv("TAVILY_API_KEY"):
        tavily_search = TavilySearch()
    else:
        tavily_search = None
except Exception as e:
    logger.warning(f"⚠️ Tavily Search 未加载: {e}")
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
    except Exception as e: 
        logger.error(f"❌ 抓取URL失败 {url}: {e}")
        return f"抓取失败: {e}"


@tool
async def bash_execute(command: str) -> str:
    """
    【系统工具：执行本地 shell 命令】
    执行基础终端命令。
    【🔴 严厉警告】：
    1. 绝对禁止使用 `echo` 等命令去“伪造”、“Mock”或“欺骗”任务输出！如果你无法真实执行某个测试，请直接告诉用户环境不支持。
    2. 禁止执行会导致阻塞的长服务或后台服务（如 nohup python ... &），本环境不支持开启 Web 服务器。
    """
    # 【安全补丁】：拦截高危命令
    dangerous_keywords = ["rm -rf /", "mkfs", "format", ":(){ :|:& };:"]
    if any(keyword in command for keyword in dangerous_keywords):
        logger.warning(f"🛡️ 拦截高危命令: {command}")
        return "⚠️ 拒绝执行：该命令触碰安全红线被系统拦截。"

    try:
        logger.info(f"💻 [Shell执行 (Async)] {command}")
        
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            # 加入 120 秒超时控制
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        except asyncio.TimeoutError:
            process.kill()
            logger.error(f"⏳ Shell执行超时: {command}")
            return f"⏳ 命令执行超时 (超过 120 秒被强制终止)。"

        stdout_str = stdout.decode('utf-8', errors='replace')
        stderr_str = stderr.decode('utf-8', errors='replace')
        
        output = stdout_str
        if process.returncode != 0 and stderr_str:
            output += f"\n[stderr]\n{stderr_str}"
        if not output.strip():
            output = f"(命令已执行，退出码: {process.returncode})"
            
        return output[:8000]
        
    except Exception as e:
        logger.error(f"❌ Shell执行失败: {e}")
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
    except Exception as e: 
        logger.error(f"❌ 读取文件失败 {file_path}: {e}")
        return f"❌ 读取失败：{e}"


@tool
async def execute_python_script(script_path: str, script_args: str = "", **kwargs) -> str:
    """
    【系统核心工具：执行本地 Python 脚本】
    🔴 严厉警告：绝对禁止使用此工具运行包含 GUI 界面（如 tkinter, PyQt, pygame）或需要控制台交互（如 input()）的脚本！
    如果你需要运行会弹出界面的游戏或图形化程序，**必须且只能使用 `run_background_program` 工具！**
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
            if "input(" in code_content or "tkinter" in code_content or "PyQt" in code_content or "pygame" in code_content or "curses" in code_content:
                logger.warning(f"🛡️ 拦截含GUI/交互脚本: {script_path}")
                return "⚠️ [系统拦截] 此脚本包含交互输入或GUI界面，使用当前工具会卡死系统。请改用专用的后台启动工具 `run_background_program` 运行它！"
    except Exception:
        pass

    try:
        logger.info(f"🤖 [执行脚本 (Async)] {script_path} {script_args} (忽略额外参数: {kwargs})")
        cmd = [sys.executable, safe_path]
        if script_args:
            cmd.extend(script_args.split())

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            # 将超时时间限制在 25 秒
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=25)
        except asyncio.TimeoutError:
            process.kill()
            logger.error(f"⏳ 脚本执行超时: {script_path}")
            return f"⏳ 脚本执行超时 (超过 25 秒被强制终止)。这通常是因为脚本陷入了死循环或在等待操作。"

        stdout_str = stdout.decode('utf-8', errors='replace')
        stderr_str = stderr.decode('utf-8', errors='replace')

        output = ""
        if stdout_str:
            output += f"【STDOUT】\n{stdout_str}\n"
        if stderr_str:
            output += f"【STDERR】\n{stderr_str}\n"

        if process.returncode == 0:
            return f"✅ 执行成功！\n{output}"
        else:
            return f"⚠️ 执行异常 (退出码 {process.returncode})：\n{output}"

    except Exception as e:
        logger.error(f"❌ 脚本执行失败 {script_path}: {e}")
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
        logger.info(f"📝 [文件写入] {file_path}")
        return f"✅ 文件已成功写入物理路径：{file_path}"
    except Exception as e:
        logger.error(f"❌ 文件写入失败 {file_path}: {e}")
        return f"❌ 文件写入失败：{str(e)}"

@tool
async def run_background_program(command: str, **kwargs) -> str:
    """
    【全新大管家工具：独立后台运行界面/游戏程序】
    如果用户要求你“打开”、“启动”、“运行”一个游戏、GUI程序（如 tkinter, pygame）或耗时服务，必须使用本工具。
    它会在用户的 Windows 电脑上单独弹出一个新的控制台窗口运行该程序，并且**立刻返回结果不阻塞你的思考**。
    例如运行刚才写好的游戏：run_background_program(command="python games/snake_game.py")
    """
    try:
        # 在 Windows 环境下，使用 CREATE_NEW_CONSOLE 可以在用户桌面上弹出一个新窗口
        # 这对于运行 pygame 等交互式游戏体验极佳
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE
            # 追加 pause 使得终端在运行结束或报错时不立刻关闭，让用户能看到报错
            command = f"{command} & pause"

        logger.info(f"🚀 [后台独立启动] 命令: {command}")
        
        # 使用 Popen 后台挂起，不等待结果直接返回
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=PROJECT_ROOT,
            creationflags=creationflags
        )
        
        return f"✅ 成功！已在用户的桌面上启动独立后台进程 (PID: {process.pid})。现在您可以告诉用户去游玩/查看了！"
    except Exception as e:
        logger.error(f"❌ 后台启动失败 {command}: {e}")
        return f"❌ 启动失败：{str(e)}"