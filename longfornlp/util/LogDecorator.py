import os
import time
import traceback
import logging
import logging.handlers
from functools import wraps


FILE_LOG_FMT = "[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s"
FILE_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def init_logger(log_filename, log_level="DEBUG"):
    
    # 创建一个日志器logger并设置其日志级别
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # 创建级别为DEBUG的按时间轮询日志分割处理器，记录所有的日志
    # “midnight”: Roll over at midnight
    # interval 是指等待多少个单位when的时间后，Logger会自动重建文件
    # backupCount 是保留日志个数。默认的0是不会自动删除掉日志
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_filename,
        when="midnight",
        interval=1,
        backupCount=0,
        encoding="UTF-8",
    )
    file_handler.setLevel(logging.DEBUG)

    # 创建级别为ERROR的日志处理器
    f_handler = logging.FileHandler("logs/error.log", encoding="UTF-8")
    f_handler.setLevel(logging.ERROR)
    
    # 创建格式器，加到日志处理器handler中
    file_handler.setFormatter(
        logging.Formatter(fmt=FILE_LOG_FMT, datefmt=FILE_DATE_FMT)
    )
    f_handler.setFormatter(
        logging.Formatter(fmt=FILE_LOG_FMT, datefmt=FILE_DATE_FMT)
    )

    # 这个判断是重复写日志问题，仅handlers列表为空才添加
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(f_handler)
    
    return logger


# log_file = "logs/default.log"
# logger = LogDecorator.init_logger(log_file)  # 日志封装类

# def log_filter(func):
#     """接口日志装饰器"""
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         rsp = None
#         start = 1000 * time.time()
#         logger.info(f"=============  Begin: {func.__name__}  =============")
#         try:
#             rsp = func(*args, **kwargs)
#         except Exception as e:
#             logger.error(traceback.format_exc())
#         end = 1000 * time.time()
#         logger.info(f"Time consuming: {end - start}ms")
#         logger.info(f"=============   End: {func.__name__}   =============\n")
#         return rsp
#     return wrapper

# @log_filter
# def main():
#     while 1:
#         print(input())
