import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler


"""
任务框架APScheduler
APScheduler是Python的一个定时任务框架，用于执行周期或者定时任务，
可以基于日期、时间间隔，及类似于Linux上的定时任务crontab类型的定时任务；
该该框架不仅可以添加、删除定时任务，还可以将任务存储到数据库中，实现任务的持久化，使用起来非常方便。

安装方式：pip install apscheduler

apscheduler组件及简单说明：
1>triggers（触发器）：    触发器包含调度逻辑，每一个作业有它自己的触发器
2>job stores（作业存储）：用来存储被调度的作业，默认的作业存储器是简单地把作业任务保存在内存中,支持存储到MongoDB，Redis数据库中
3>executors（执行器）：   执行器用来执行定时任务，只是将需要执行的任务放在新的线程或者线程池中运行
4>schedulers（调度器）：  调度器是将其它部分联系在一起,对使用者提供接口，进行任务添加，设置，删除。

参考：https://blog.csdn.net/wsxx1020/article/details/78521564
"""

def service():
    pass

def dojob():
    # 创建调度器：BlockingScheduler
    scheduler = BlockingScheduler()
    # 添加任务,时间间隔2S
    scheduler.add_job(service, "interval", seconds=2, id="test_job1")
    scheduler.start()


class BaseScheduler(object):

    default_interval = 2
    default_second = 30
    default_month = "1-3,7-9"
    default_day = "1"
    default_hour = "1"
    executor = BackgroundScheduler(timezone="Asia/Shanghai")

    def __init__(self):
        super(BaseScheduler, self).__init__()

    @property
    def get_first_run_time(self):
        return datetime.datetime.now() + datetime.timedelta(seconds=+60)

    def service(self):
        pass

    def interval_start(self):
        kwargs = dict()
        kwargs['minutes'] = self.default_interval
        kwargs['start_date'] = self.get_first_run_time

        self.executor.add_job(self.service, 'interval', max_instances=1, **kwargs)
        self.executor.start()

    def cron_start(self):
        kwargs = dict()
        kwargs['month'] = self.default_month
        kwargs['day'] = self.default_day
        kwargs['hour'] = self.default_hour
        kwargs['second'] = self.default_second
        kwargs['start_date'] = self.get_first_run_time

        self.executor.add_job(self.service, 'cron', max_instances=1, **kwargs)
        self.executor.start()

    def date_start(self):
        kwargs = dict()
        kwargs['run_date'] = '2009-11-06 16:30:05'

        self.executor.add_job(self.service, 'date', max_instances=1, **kwargs)
        self.executor.start()
