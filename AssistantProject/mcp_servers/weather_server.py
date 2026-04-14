# mcp_servers/weather_server.py
from mcp.server.fastmcp import FastMCP

# 1. 初始化一个 MCP 服务器，命名为 "WeatherService"
mcp = FastMCP("WeatherService")


# 2. 使用 @mcp.tool() 装饰器将 Python 函数暴露给大模型
@mcp.tool()
def get_weather(city: str) -> str:
    """
    【必须要有清晰的 docstring】
    获取指定中国城市的天气信息。当用户询问天气时调用此工具。
    """
    print(f"[天气服务被触发] 正在查询城市: {city}")

    # 这里为了演示，我们使用 Mock（假）数据。
    # 真实开发中，你可以在这里用 requests 调用心知天气、和风天气的 API
    mock_weather_db = {
        "北京": "晴朗，气温 22°C，微风，适合外出。",
        "上海": "多云转小雨，气温 25°C，湿度 70%，建议带伞。",
        "广州": "雷阵雨，气温 28°C，闷热。",
        "深圳": "晴，气温 30°C，紫外线强，注意防晒。"
    }

    # 模糊匹配一下
    for key, value in mock_weather_db.items():
        if key in city:
            return value

    return f"抱歉，暂时没有查到【{city}】的天气数据，目前只支持北上广深。"


if __name__ == "__main__":
    # 3. 启动服务器，默认使用 stdio（标准输入输出）协议进行跨进程通信
    mcp.run()