# core/db_manager.py
import sqlite3
import os
import json
from datetime import datetime

# 将数据库文件存放在项目的 data 目录下
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chat_history.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """初始化数据库表结构"""
    conn = get_connection()
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
    conn.commit()
    conn.close()


def get_all_sessions():
    """获取所有历史会话列表，按时间倒序"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT session_id, title FROM sessions ORDER BY updated_at DESC')
    rows = cursor.fetchall()
    conn.close()
    # 返回格式: ["对话标题_id", ...] 以便在 Gradio 下拉框显示
    return [f"{row[1]} ({row[0]})" for row in rows]


def load_session(session_display_name):
    """根据下拉框选中的名字加载会话数据"""
    if not session_display_name:
        return [], [], None

    # 提取括号里的真实 session_id
    session_id = session_display_name.split('(')[-1].strip(')')

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT history_json, state_messages_json FROM sessions WHERE session_id = ?', (session_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        history = json.loads(row[0]) if row[0] else []
        state_messages = json.loads(row[1]) if row[1] else []
        return history, state_messages, session_id
    return [], [], None


def save_session(session_id, title, history, state_messages):
    """保存或更新会话"""
    conn = get_connection()
    cursor = conn.cursor()

    history_str = json.dumps(history, ensure_ascii=False)
    state_messages_str = json.dumps(state_messages, ensure_ascii=False)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 使用 REPLACE 语法，存在则更新，不存在则插入
    cursor.execute('''
        REPLACE INTO sessions (session_id, title, updated_at, history_json, state_messages_json)
        VALUES (?, ?, ?, ?, ?)
    ''', (session_id, title, now, history_str, state_messages_str))

    conn.commit()
    conn.close()


def delete_session(session_display_name):
    """删除会话"""
    if not session_display_name: return
    session_id = session_display_name.split('(')[-1].strip(')')

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()


# 模块加载时自动建表
init_db()