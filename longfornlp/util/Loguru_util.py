from loguru import logger

"""
Python 中更优雅的日志记录方案 loguru
https://cuiqingcai.com/7776.html
"""


# 定位到log日志文件
log_path = os.path.join('./', 'logs')

if not os.path.exists(log_path):
    os.mkdir(log_path)

log_path_info = os.path.join(log_path, f'{time.strftime("%Y-%m-%d")}_info.log')
log_path_warning = os.path.join(log_path, f'{time.strftime("%Y-%m-%d")}_warning.log')
log_path_error = os.path.join(log_path, f'{time.strftime("%Y-%m-%d")}_error.log')

"""日志简单配置 文件区分不同级别的日志
rotation 参数：可以实现一天输出一个日志文件，或者文件太大了自动分隔日志文件
    rotation="200 MB"：每个 log 文件达到200M就会自动进行日志分隔
    rotation='00:00'：每天 0 点新创建一个 log 文件输出
    rotation='1 week'：每隔一周创建一个 log 文件
retention 参数：配置日志的最长保留时间
    retention='10 days'：日志文件最长保留 10 天
compression 配置：文件的压缩格式
    compression='zip'：使用 zip 文件格式保存，节省存储空间
"""
logger.add(log_path_info, rotation="200 MB", encoding='utf-8', enqueue=True, level='INFO', retention='10 days')
logger.add(log_path_warning, rotation="200 MB", encoding='utf-8', enqueue=True, level='WARNING', retention='10 days')
logger.add(log_path_error, rotation="200 MB", encoding='utf-8', enqueue=True, level='ERROR', retention='10 days')

# logger.info(f"xxx{func.__name__}")
# logger.error(traceback.format_exc())  # 错误日志 repr(e)

# Traceback 记录，使用 Loguru 提供的装饰器进行 Traceback 的记录

# @logger.catch
# def my_function(x, y, z):
#     # An error? It's caught anyway!
#     return 1 / (x + y + z)
