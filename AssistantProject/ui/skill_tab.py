# ui/skill_tab.py
import gradio as gr
import re
import yaml
from AssistantProject.core.skill_manager import (
    get_skill_choices, load_skill_detail, save_skill, delete_skill
)
from AssistantProject.core.agent import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


def ui_save_skill(old_name, new_name, desc, prompt, script_files, ref_files):
    success, msg = save_skill(old_name, new_name, desc, prompt, script_files, ref_files)
    # 保存后清空文件上传框，防止重复上传
    return gr.update(choices=get_skill_choices(), value=new_name if success else old_name), msg, None, None


def ui_delete_skill(skill_name):
    success, msg = delete_skill(skill_name)
    return gr.update(choices=get_skill_choices(), value=None), "", "", msg


def generate_skill_by_ai(requirement, target_model):
    if not requirement.strip(): return gr.update(), gr.update(), "⚠️ 请输入需求"
    sys_prompt = """你是一位顶级的 AI Agentic Skill 架构师。你需要根据用户需求，编写详尽的工业级 SKILL.md。
【严重警告】：绝对不要使用 JSON 格式！直接输出纯文本！

请按照以下结构输出：
---
name: 小写英文字母-中划线 (如 py-executor)
description: 触发条件描述，稍微激进一点。
enable: true
---
# [技能大标题]
## 概述与核心逻辑
## 强制工作流 (Workflow)
## 输出规范与示例
## 关联文件说明 (如果该技能需要外挂 python 脚本，请在这里告诉大模型应该去读取脚本执行)
"""
    try:
        gr.Info("🧠 正在生成，请耐心等待...")
        llm = get_llm(target_model, max_token=4096, temperature=0.7)
        res = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=f"需求：{requirement}")])
        res_text = res.content.strip()
        res_text = re.sub(r'^```[a-zA-Z]*\n', '', res_text)
        res_text = re.sub(r'```$', '', res_text).strip()

        name = "auto-skill"
        description = "自动生成的技能"
        prompt_body = res_text
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', res_text, re.DOTALL)
        if match:
            try:
                metadata = yaml.safe_load(match.group(1)) or {}
                name = metadata.get("name", name)
                description = metadata.get("description", description)
                prompt_body = match.group(2).strip()
            except Exception:
                pass
        return name, description, prompt_body
    except Exception as e:
        return "", "", f"⚠️ 失败: {e}"


def create_skill_tab():
    gr.Markdown("### 🧩 Skill 技能扩展包管理 (装备库模式)")
    choices = get_skill_choices()
    default_name = choices[0] if choices else None
    default_desc, default_prompt = load_skill_detail(default_name)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### 📚 本地技能包列表")
            skill_list = gr.Radio(choices=choices, label="已安装包", value=default_name, interactive=True)
            with gr.Row():
                new_btn = gr.Button("✨ 手动新建", variant="secondary", size="sm")
                delete_btn = gr.Button("🗑️ 删除选中", variant="stop", size="sm")
            status_box = gr.Textbox(label="操作状态", lines=2, interactive=False)

        with gr.Column(scale=6):
            with gr.Accordion("🪄 懒人专属：AI 辅助生成", open=False):
                with gr.Row():
                    ai_req = gr.Textbox(label="需求描述", scale=4)
                    model_sel = gr.Dropdown(
                        choices=[("通义千问 Qwen-Max", "qwen-max"), ("智谱 GLM-4-Flash", "glm-4-flash")],
                        value="qwen-max", scale=1)
                    ai_gen_btn = gr.Button("🧠 生成", variant="primary", scale=1)

            gr.Markdown("#### ✍️ 扩展包与装备配置")
            with gr.Row():
                name_in = gr.Textbox(label="包名 (如 get-time)", scale=2, value=default_name if default_name else "")
                desc_in = gr.Textbox(label="触发逻辑描述", scale=3, value=default_desc)

            prompt_in = gr.Code(label="SKILL.md 正文 (操作手册)", language="markdown", lines=15, value=default_prompt)

            # 【核心新增】给大模型挂载实体装备的上传接口！
            gr.Markdown("##### 🧰 为该技能挂载外部装备 (可选)")
            with gr.Row():
                scripts_upload = gr.File(label="🐍 上传代码脚本 (存入 scripts/)", file_count="multiple", type="filepath")
                refs_upload = gr.File(label="📄 上传专有文档 (存入 references/)", file_count="multiple", type="filepath")

            save_btn = gr.Button("💾 封装全套包并保存至 skills/ 目录", variant="primary")

    skill_list.change(fn=load_skill_detail, inputs=[skill_list], outputs=[desc_in, prompt_in]).then(fn=lambda x: x,
                                                                                                    inputs=[skill_list],
                                                                                                    outputs=[name_in])
    new_btn.click(fn=lambda: (None, "", "", ""), outputs=[skill_list, name_in, desc_in, prompt_in])

    # 绑定保存事件，带上文件上传参数
    save_btn.click(
        fn=ui_save_skill,
        inputs=[skill_list, name_in, desc_in, prompt_in, scripts_upload, refs_upload],
        outputs=[skill_list, status_box, scripts_upload, refs_upload]
    )
    delete_btn.click(fn=ui_delete_skill, inputs=[skill_list], outputs=[skill_list, name_in, desc_in, status_box])
    ai_gen_btn.click(fn=generate_skill_by_ai, inputs=[ai_req, model_sel], outputs=[name_in, desc_in, prompt_in])