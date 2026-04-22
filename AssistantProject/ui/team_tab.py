import gradio as gr
from AssistantProject.core.team_manager import get_teams, get_team_config, save_team, delete_team

def create_team_tab():
    gr.Markdown("### 👥 团队编排中心 (Team Orchestrator)")
    gr.Markdown("在这里，你可以像搭积木一样，自定义一条**流水线团队**。团队中的每个节点将按顺序接力完成工作。")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("#### 1. 团队选择与基础信息")
            team_list = gr.Dropdown(choices=get_teams(), label="选择已有团队", interactive=True)
            refresh_btn = gr.Button("🔄 刷新列表", size="sm")
            
            gr.Markdown("---")
            team_id_in = gr.Textbox(label="团队 ID (唯一标识，英文/数字)", placeholder="例如: dev_team")
            team_name_in = gr.Textbox(label="团队显示名称", placeholder="例如: 核心开发团队")
            
            with gr.Row():
                new_btn = gr.Button("✨ 新建空团队", variant="secondary")
                delete_btn = gr.Button("🗑️ 删除该团队", variant="stop")
                
            status_box = gr.Textbox(label="操作状态", interactive=False)

        with gr.Column(scale=7):
            gr.Markdown("#### 2. 定义流水线节点 (按顺序执行)")
            gr.Markdown("在下方表格中依次添加节点。例如：第一行写「产品经理」，第二行写「程序员」。系统会自动按顺序连线。")
            
            # 使用 Dataframe 管理节点列表
            # 列：节点名称(name), 角色提示词(prompt)
            nodes_df = gr.Dataframe(
                headers=["节点名称 (Name)", "系统提示词 (System Prompt)", "授权环境 (逗号分隔, 如: visual_server)"],
                datatype=["str", "str", "str"],
                col_count=(3, "fixed"),
                row_count=1,
                interactive=True,
                wrap=True,
                type="array"
            )
            
            save_btn = gr.Button("💾 保存完整团队编排", variant="primary")

    # --- 逻辑绑定 ---
    def refresh_list():
        return gr.update(choices=get_teams())

    refresh_btn.click(fn=refresh_list, inputs=[], outputs=[team_list])

    def load_team(t_id):
        if not t_id:
            return "", "", [["", "", ""]]
        config = get_team_config(t_id)
        if config:
            df_data = []
            for node in config.get("nodes", []):
                allowed = ",".join(node.get("allowed_mcps", []))
                df_data.append([node.get("name", ""), node.get("prompt", ""), allowed])
            if not df_data:
                df_data = [["", "", ""]]
            return config.get("team_id", t_id), config.get("team_name", ""), df_data
        return t_id, "", [["", "", ""]]

    team_list.change(fn=load_team, inputs=[team_list], outputs=[team_id_in, team_name_in, nodes_df])

    def clear_form():
        return None, "", "", [["", "", ""]]
    
    new_btn.click(fn=clear_form, inputs=[], outputs=[team_list, team_id_in, team_name_in, nodes_df])

    def save_action(t_id, t_name, df):
        if not t_id or not t_name:
            return "❌ 请填写团队 ID 和名称"
        
        nodes = []
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            df = df.values.tolist()
            
        for row in df:
            name = row[0] if len(row) > 0 else ""
            prompt = row[1] if len(row) > 1 else ""
            allowed = row[2] if len(row) > 2 else ""
            
            if str(name).strip() and str(prompt).strip():
                allowed_list = [s.strip() for s in str(allowed).split(",") if s.strip()]
                nodes.append({"name": str(name).strip(), "prompt": str(prompt).strip(), "allowed_mcps": allowed_list})
                
        if not nodes:
            return "❌ 至少需要定义一个有效的节点！"
            
        msg = save_team(t_id, t_name, nodes)
        return msg

    save_btn.click(fn=save_action, inputs=[team_id_in, team_name_in, nodes_df], outputs=[status_box]).then(
        fn=refresh_list, inputs=[], outputs=[team_list]
    )

    def delete_action(t_id):
        if not t_id: return "❌ 请先选择要删除的团队"
        msg = delete_team(t_id)
        return msg

    delete_btn.click(fn=delete_action, inputs=[team_id_in], outputs=[status_box]).then(
        fn=clear_form, inputs=[], outputs=[team_list, team_id_in, team_name_in, nodes_df]
    )
