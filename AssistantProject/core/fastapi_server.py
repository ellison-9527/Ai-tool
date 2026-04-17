# core/fastapi_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from AssistantProject.core.db_manager import get_all_sessions, save_session, load_session

app = FastAPI(title="AssistantPro DB Service")

# 定义数据模型
class ChatSession(BaseModel):
    session_id: str
    messages: List[dict]
    title: str = "新对话"

@app.get("/sessions")
def list_sessions():
    """接口 1：获取所有历史会话"""
    try:
        return {"status": "success", "data": get_all_sessions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/save")
def save_chat_session(session: ChatSession):
    """接口 2：保存或更新会话"""
    try:
        save_session(session.session_id, session.messages)
        return {"status": "success", "message": f"Session {session.session_id} saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)