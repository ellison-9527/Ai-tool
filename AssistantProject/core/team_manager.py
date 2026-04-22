import os
import json
from AssistantProject.core.logger import logger

TEAMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "teams"))

def ensure_teams_dir():
    if not os.path.exists(TEAMS_DIR):
        os.makedirs(TEAMS_DIR)

def get_teams():
    ensure_teams_dir()
    teams = []
    for file in os.listdir(TEAMS_DIR):
        if file.endswith('.json'):
            teams.append(file.replace('.json', ''))
    return teams

def get_team_config(team_id):
    ensure_teams_dir()
    filepath = os.path.join(TEAMS_DIR, f"{team_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取团队配置失败: {e}")
            return None
    return None

def save_team(team_id, team_name, nodes):
    """
    nodes: list of dict [{"name": "产品经理", "prompt": "..."}]
    """
    ensure_teams_dir()
    filepath = os.path.join(TEAMS_DIR, f"{team_id}.json")
    data = {
        "team_id": team_id,
        "team_name": team_name,
        "nodes": nodes
    }
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return f"✅ 团队 [{team_name}] 保存成功！"
    except Exception as e:
        logger.error(f"保存团队配置失败: {e}")
        return f"❌ 团队保存失败: {e}"

def delete_team(team_id):
    filepath = os.path.join(TEAMS_DIR, f"{team_id}.json")
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            return f"🗑️ 团队 [{team_id}] 已删除！"
        except Exception as e:
            return f"❌ 团队删除失败: {e}"
    return "团队不存在。"
