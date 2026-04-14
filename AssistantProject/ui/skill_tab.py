# ui/skill_tab.py
import gradio as gr
import json
import os

# 【核心修复1】引入 LangChain 原生模块和你在 agent.py 封装好的引擎
from AssistantProject.core.agent import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

CONFIG_DIR = "data"
SKILLS_CONFIG_PATH = os.path.join(CONFIG_DIR, "skills.json")

PRESET_SKILLS = {
    "👨‍💻 代码审查专家": {
        "description": "帮助分析代码，找出潜在的 Bug、安全漏洞并提供优化建议。",
        "prompt": "你现在是一位拥有 20 年经验的资深架构师和代码审查专家。\n请仔细阅读用户提供的代码，并从以下几个维度进行分析：\n1. 潜在的 Bug 或逻辑错误\n2. 性能优化建议\n3. 代码规范与可读性\n4. 安全漏洞\n\n请务必给出具体的修改示例和重构代码。"
    },
    "✍️ 小红书爆款写手": {
        "description": "根据主题生成符合小红书调性的爆款文案。",
        "prompt": "你是一个精通小红书爆款逻辑的文案大师。\n\n你的文案特点是：\n- 标题吸睛（含有情绪价值和痛点）\n- 善用 Emoji 🌟\n- 排版清晰（多分段、多用列表）\n- 语气亲切（多用“姐妹们”、“绝绝子”等词汇）\n\n请根据用户给出的主题，生成一篇带标签的完整图文笔记文案。"
    }
}


def load_skills():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(SKILLS_CONFIG_PATH):
        with open(SKILLS_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(PRESET_SKILLS, f, ensure_ascii=False, indent=2)
        return PRESET_SKILLS.copy()

    try:
        with open(SKILLS_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return PRESET_SKILLS.copy()


def save_skills_to_disk():
    with open(SKILLS_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current_skills, f, ensure_ascii=False, indent=2)


current_skills = load_skills()


def get_skill_choices():
    return list(current_skills.keys())


def load_skill_detail(skill_name):
    if not skill_name or skill_name not in current_skills:
        return "", ""
    skill = current_skills[skill_name]
    return skill.get("description", ""), skill.get("prompt", "")


def save_skill(old_name, new_name, desc, prompt):
    if not new_name.strip():
        return gr.update(), "⚠️ Skill 名称不能为空！"
    if old_name and old_name != new_name and old_name in current_skills:
        del current_skills[old_name]

    current_skills[new_name] = {"description": desc, "prompt": prompt}
    save_skills_to_disk()
    return gr.update(choices=get_skill_choices(), value=new_name), f"✅ 技能 [{new_name}] 已成功保存到本地！"


def delete_skill(skill_name):
    if skill_name in current_skills:
        del current_skills[skill_name]
        save_skills_to_disk()
        return gr.update(choices=get_skill_choices(), value=None), "", "", "🗑️ 已成功删除"
    return gr.update(), "", "", "⚠️ 未找到该技能"


def generate_skill_by_ai(requirement, target_model):
    """调用 LLM 根据简短需求自动生成完整的 System Prompt"""
    if not requirement.strip():
        return gr.update(), gr.update(), "⚠️ 请输入您的需求！"

    sys_prompt = """你是一个专业的Prompt提示词架构师。
用户的输入将是一个非常简短的对AI角色的需求（比如“帮我写一个法律顾问”）。
你的任务是为这个角色设计出结构清晰、专业的高质量系统提示词(System Prompt)。
你必须严格按照以下 JSON 格式输出，不要包含任何 Markdown 标记（如```json），也不要加任何废话：
{
    "name": "专家名称(带Emoji，如 👨‍⚖️ 资深法务顾问)",
    "description": "简短的一句话描述它的功能",
    "prompt": "极其详尽的系统提示词内容（包含：角色背景、核心能力、任务设定、沟通风格、输出规范等）"
}
"""
    try:
        gr.Info(f"🧠 正在使用 {target_model} 构思角色设定中...")

        # 【核心修复2】复用聊天页面同款底层引擎，彻底杜绝 Connection error
        llm = get_llm(target_model=target_model, max_token=2048, temperature=0.7)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"需求：{requirement}")
        ]

        # 调用 LangChain 的 invoke 方法直接获取结果
        response = llm.invoke(messages)
        res_text = response.content.strip()

        if res_text.startswith("```json"):
            res_text = res_text[7:]
        if res_text.startswith("```"):
            res_text = res_text[3:]
        if res_text.endswith("```"):
            res_text = res_text[:-3]

        data = json.loads(res_text.strip())
        return data.get("name", ""), data.get("description", ""), data.get("prompt", "")
    except Exception as e:
        return "", "", f"⚠️ AI 生成失败，原因: {str(e)}。您可以重试或手动编写。"


def create_skill_tab():
    gr.Markdown("### 🧩 Skill 技能配置中心")
    gr.Markdown(
        "在这里管理和编写你的角色提示词 (Prompt Skills)，打造不同领域的专属智能体。这里的配置会自动保存到 `data/skills.json`。")

    choices = get_skill_choices()
    default_name = choices[0] if choices else ""
    default_desc, default_prompt = load_skill_detail(default_name)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### 📚 已有技能列表")
            skill_list = gr.Radio(choices=choices, label="选择要编辑的 Skill", value=default_name, interactive=True)
            with gr.Row():
                new_btn = gr.Button("✨ 手动新建", variant="secondary", size="sm")
                delete_btn = gr.Button("🗑️ 删除选中", variant="stop", size="sm")
            status_box = gr.Textbox(label="操作状态", lines=2, interactive=False)

        with gr.Column(scale=6):
            with gr.Accordion("🪄 懒人专属：让 AI 帮你写 Skill", open=True):
                gr.Markdown("只需要输入你的一句话想法，大模型将自动为你生成详尽的系统提示词（Prompt）。")
                with gr.Row():
                    ai_req_input = gr.Textbox(label="你期望的 AI 角色需求",
                                              placeholder="例如：帮我写一个专门负责解答劳动法问题的资深律师，要求语气严谨专业。",
                                              scale=3)

                    # 【核心修复3】将前端展示标签统一，杜绝视觉冲突
                    skill_model_dropdown = gr.Dropdown(
                        choices=[
                            ("智谱 GLM-4-Flash", "glm-4-flash"),
                            ("通义千问 Qwen-Max", "qwen-max"),
                            ("MiniMax 2.5", "abab6.5s-chat")
                        ],
                        label="🧠 辅助生成模型",
                        value="glm-4-flash",
                        scale=1
                    )
                    ai_gen_btn = gr.Button("🧠 一键生成设定", variant="primary", scale=1)

            gr.Markdown("#### ✍️ 技能详情设定 (生成后可在此微调)")

            with gr.Row():
                skill_name_input = gr.Textbox(label="Skill 名称", placeholder="给你的专家起个名字...", scale=2,
                                              value=default_name)
                skill_desc_input = gr.Textbox(label="描述 (一句话介绍用途)", placeholder="例如：负责审核 Python 代码",
                                              scale=3, value=default_desc)

            skill_prompt_input = gr.Code(label="Skill 核心逻辑 (系统提示词 System Prompt)", language="markdown",
                                         lines=15, value=default_prompt)

            with gr.Row():
                save_btn = gr.Button("💾 保存并应用此配置", variant="primary")

    # --- 交互绑定 ---
    skill_list.change(fn=load_skill_detail, inputs=[skill_list], outputs=[skill_desc_input, skill_prompt_input]).then(
        fn=lambda x: x, inputs=[skill_list], outputs=[skill_name_input])

    new_btn.click(fn=lambda: (None, "", "", ""), inputs=[],
                  outputs=[skill_list, skill_name_input, skill_desc_input, skill_prompt_input])
    save_btn.click(fn=save_skill, inputs=[skill_list, skill_name_input, skill_desc_input, skill_prompt_input],
                   outputs=[skill_list, status_box])
    delete_btn.click(fn=delete_skill, inputs=[skill_list],
                     outputs=[skill_list, skill_name_input, skill_desc_input, status_box])

    ai_gen_btn.click(
        fn=generate_skill_by_ai,
        inputs=[ai_req_input, skill_model_dropdown],
        outputs=[skill_name_input, skill_desc_input, skill_prompt_input]
    )