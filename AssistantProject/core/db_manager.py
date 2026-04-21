# core/db_manager.py
import sqlite3
import os
import json
from datetime import datetime
from AssistantProject.core.logger import logger

# 将数据库文件存放在项目的 data 目录下
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chat_history.db"))


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """初始化数据库表结构"""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            # 创建会话表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    updated_at DATETIME,
                    history_json TEXT,
                    state_messages_json TEXT
                )
            ''')
            # 为 updated_at 添加索引以加速排序
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)')
            conn.commit()
            logger.info("✅ 数据库表结构初始化成功")
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")


def get_all_sessions():
    """获取所有历史会话列表，按时间倒序"""
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT session_id, title FROM sessions ORDER BY updated_at DESC')
            rows = cursor.fetchall()
            # 返回格式: ["对话标题 (session_id)", ...] 以便在 Gradio 下拉框显示
            return [f"{row[1]} ({row[0]})" for row in rows]
    except Exception as e:
        logger.error(f"❌ 获取所有会话失败: {e}")
        return []


def load_session(session_display_name):
    """根据下拉框选中的名字加载会话数据"""
    if not session_display_name:
        return [], [], None

    try:
        # 提取括号里的真实 session_id
        session_id = session_display_name.split('(')[-1].strip(')')
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT history_json, state_messages_json FROM sessions WHERE session_id = ?', (session_id,))
            row = cursor.fetchone()

        if row:
            history = json.loads(row[0]) if row[0] else []
            state_messages = json.loads(row[1]) if row[1] else []
            return history, state_messages, session_id
    except Exception as e:
        logger.error(f"❌ 加载会话 {session_display_name} 失败: {e}")
        
    return [], [], None


def save_session(session_id, title, history, state_messages):
    """保存或更新会话"""
    try:
        history_str = json.dumps(history, ensure_ascii=False)
        state_messages_str = json.dumps(state_messages, ensure_ascii=False)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with get_connection() as conn:
            cursor = conn.cursor()
            # 使用标准的 ON CONFLICT 进行 UPSERT 操作
            cursor.execute('''
                INSERT INTO sessions (session_id, title, updated_at, history_json, state_messages_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title=excluded.title,
                    updated_at=excluded.updated_at,
                    history_json=excluded.history_json,
                    state_messages_json=excluded.state_messages_json
            ''', (session_id, title, now, history_str, state_messages_str))
            conn.commit()
    except Exception as e:
        logger.error(f"❌ 保存会话 {session_id} 失败: {e}")


def delete_session(session_display_name):
    """删除会话"""
    if not session_display_name: 
        return
    try:
        session_id = session_display_name.split('(')[-1].strip(')')
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
            conn.commit()
            logger.info(f"🗑️ 会话 {session_id} 已删除")
    except Exception as e:
        logger.error(f"❌ 删除会话 {session_display_name} 失败: {e}")


# 模块加载时自动建表
init_db()