import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger():
    # 确保日志目录存在
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "assistant.log")

    logger = logging.getLogger("assistant")
    # 如果已经有 handler，不重复添加
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出 (每个文件最大 10MB，保留 5 个备份)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
