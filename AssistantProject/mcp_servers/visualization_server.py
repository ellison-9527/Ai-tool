# mcp_servers/visualization_server.py
import os
import uuid
import time
import matplotlib.pyplot as plt
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DataVisualizer")

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "workspace", "charts"))
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def cleanup_old_charts(directory, max_age_hours=72):
    """【智能保洁】清理超过指定时间(默认3天)的废弃临时图表"""
    try:
        now = time.time()
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                filepath = os.path.join(directory, filename)
                # 如果文件的修改时间早于 (当前时间 - 小时数*3600秒)，则删之
                if os.path.getmtime(filepath) < now - max_age_hours * 3600:
                    os.remove(filepath)
                    print(f"🗑️ 已自动清理过期临时图表: {filename}")
    except Exception as e:
        print(f"⚠️ 自动清理失败: {e}")


@mcp.tool()
def generate_bar_chart(title: str, x_labels: list[str], y_values: list[float], x_axis_name: str = "",
                       y_axis_name: str = "") -> str:
    """生成一张精美的柱状图 (Bar Chart) 并保存到本地。"""
    # 每次生成前，顺手打扫一下卫生
    cleanup_old_charts(SAVE_DIR)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_labels, y_values, color='skyblue', edgecolor='black')
    plt.title(title, fontsize=16)
    plt.xlabel(x_axis_name, fontsize=12)
    plt.ylabel(y_axis_name, fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 2), va='bottom', ha='center')

    plt.tight_layout()

    filename = f"bar_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.abspath(os.path.join(SAVE_DIR, filename))
    plt.savefig(filepath, dpi=150)
    plt.close()

    abs_path = filepath.replace("\\", "/")
    return f"图表已成功生成！请严格使用以下 Markdown 语法向用户展示该图片：\n![{title}](/file={abs_path})"


@mcp.tool()
def generate_pie_chart(title: str, labels: list[str], values: list[float]) -> str:
    """生成一张饼图 (Pie Chart) 并保存到本地。"""
    # 每次生成前，顺手打扫一下卫生
    cleanup_old_charts(SAVE_DIR)

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title(title, fontsize=16)

    filename = f"pie_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.abspath(os.path.join(SAVE_DIR, filename))
    plt.savefig(filepath, dpi=150)
    plt.close()

    abs_path = filepath.replace("\\", "/")
    return f"图表已成功生成！请严格使用以下 Markdown 语法向用户展示该图片：\n![{title}](/file={abs_path})"


if __name__ == "__main__":
    mcp.run(transport='stdio')