"""
Python3-定时任务四种实现方式:
1>循环+sleep；
2>线程模块中Timer类；
3>schedule模块；
4>定时框架：APScheduler

实现系统监测功能
1：定时或者定点监测CPU与内存使用率；
2：将时间，CPU，内存使用情况保存到日志文件；

参考：https://blog.51cto.com/huangyg/2367088
"""

# psutil:获取系统信息模块，可以获取CPU，内存，磁盘等的使用情况
import psutil
import time
import datetime
import schedule
import time
from threading import Timer
from apscheduler.schedulers.blocking import BlockingScheduler


"""
最简单使用方式：sleep
使用while+sleep就可以实现
"""


# logfile：监测信息写入文件
def MonitorSystem(logfile=None):
    # 获取cpu使用情况
    cpuper = psutil.cpu_percent()
    # 获取内存使用情况：系统内存大小，使用内存，有效内存，内存使用率
    mem = psutil.virtual_memory()
    # 内存使用率
    memper = mem.percent
    # 获取当前时间
    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} cpu:{cpuper}%, mem:{memper}%"
    print(line)
    if logfile:
        logfile.write(line)


# def loopMonitor():
#     while True:
#         MonitorSystem()
#         # 3s检查一次
#         time.sleep(3)
# loopMonitor()


"""
threading模块中的Timer
timer最基本理解就是定时器，我们可以启动多个定时任务，这些定时器任务是异步执行，所以不存在等待顺序执行问题。
"""


# logfile：监测信息写入文件
def MonitorSystem2(logfile=None):
    # 获取cpu使用情况
    cpuper = psutil.cpu_percent()
    # 获取内存使用情况：系统内存大小，使用内存，有效内存，内存使用率
    mem = psutil.virtual_memory()
    # 内存使用率
    memper = mem.percent
    # 获取当前时间
    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} cpu:{cpuper}%, mem:{memper}%"
    print(line)
    if logfile:
        logfile.write(line)
    # 启动定时器任务，每三秒执行一次
    Timer(3, MonitorSystem2).start()


# MonitorSystem2()


"""
调度模块：schedule
schedule是一个第三方轻量级的任务调度模块，可以按照秒，分，小时，日期或者自定义事件执行时间；
安装方式：pip install schedule

特征
1. 一种易于使用的API，用于调度作业。
2. 非常轻巧，没有外部依赖性。
3. 出色的测试覆盖率。
4. 在Python 2.7、3.5和3.6上测试
"""


def tasklist():
    # 清空任务
    schedule.clear()
    # 创建一个按秒间隔执行任务
    schedule.every(1).seconds.do(MonitorSystem)
    # schedule.every(1).hour.do(MonitorSystem)
    # 执行10S
    while True:
        schedule.run_pending()
        time.sleep(1)


# tasklist()


"""
任务框架APScheduler
APScheduler是Python的一个定时任务框架，用于执行周期或者定时任务，
可以基于日期、时间间隔，及类似于Linux上的定时任务crontab类型的定时任务；
该该框架不仅可以添加、删除定时任务，还可以将任务存储到数据库中，实现任务的持久化，使用起来非常方便。
安装方式：pip install apscheduler

apscheduler组件及简单说明：

1>triggers（触发器）：触发器包含调度逻辑，每一个作业有它自己的触发器
2>job stores（作业存储）:用来存储被调度的作业，默认的作业存储器是简单地把作业任务保存在内存中,支持存储到MongoDB，Redis数据库中
3> executors（执行器）：执行器用来执行定时任务，只是将需要执行的任务放在新的线程或者线程池中运行
4>schedulers（调度器）：调度器是将其它部分联系在一起,对使用者提供接口，进行任务添加，设置，删除。
参考：https://blog.csdn.net/wsxx1020/article/details/78521564
"""


def dojob():
    # 创建调度器：BlockingScheduler
    scheduler = BlockingScheduler()
    # 添加任务,时间间隔2S
    scheduler.add_job(MonitorSystem, "interval", seconds=2, id="test_job1")
    scheduler.start()


# dojob()


"""
最后选择：
老猫简单总结上面四种定时定点任务实现：
1：循环+sleep方式适合简答测试，
2：timer可以实现定时任务，但是对定点任务来说，需要检查当前时间点；
3：schedule可以定点定时执行，但是需要在循环中检测任务，而且存在阻塞；
4：APScheduler框架更加强大，可以直接在里面添加定点与定时任务；
综合考虑，使用APScheduler框架，实现简单，只需要直接创建任务，并将添加到调度器中即可。
"""
