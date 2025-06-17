# ===============================================================================
# 文件路径: lib/logger.py
# 描述: 一个基础的日志工具，以满足旧框架的导入需求。
# ===============================================================================
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    """
    创建一个简单的日志记录器，将日志信息打印到控制台。
    这个简化版本不写入文件，只用于满足导入和基本打印功能。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if not logger.handlers:
        # 创建一个输出到控制台的handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        
        # 定义日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # 添加handler
        logger.addHandler(ch)
        
    # 为了防止日志信息向上传播到root logger
    logger.propagate = False
    
    return logger 