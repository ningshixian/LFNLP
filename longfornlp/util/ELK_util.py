# encoding:utf-8
# Author:Richie
# Date:2019/5/13
import json
import threading
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer

"""记录日志到 paas 网关？"""


class ELK_Logging:
    def __init__(self, kafka_ip_list, AppID):
        self.kafka_ip_list = kafka_ip_list
        self.AppID = AppID

    def producer(self, DetailInfo, Level="INFO", **kwargs):
        assert Level in ["INFO", "FATAL", "ERROR", "WARNING", "DEBUG"]
        producer = KafkaProducer(
            value_serializer=lambda v: v.encode("utf-8"),
            retries=5,
            bootstrap_servers=self.kafka_ip_list,
        )

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S,") + "%03d" % (
            int(datetime.now().strftime("%f")) // 1000
        )
        print(now)
        # AppID | Module | DateTime | Level | RequestID | TraceID | RequestIP | UserIP | ServerIP | ServerPort | ProcessID | Thread | Location | DetailInfo
        default_json = {
            "AppID": self.AppID,
            "Module": None,
            "DateTime": now,
            "Level": "INFO",
            "RequestID": None,
            "TraceID": None,
            "RequestIP": None,
            "UserIP": None,
            "ServerIP": None,
            "ServerPort": None,
            "ProcessID": None,
            "Thread": None,
            "Location": None,
            "DetailInfo": "_",
        }
        print(default_json)
        for options in kwargs.keys():
            assert options in default_json.keys()
            default_json[options] = kwargs[options]
        default_json["DetailInfo"] = DetailInfo
        default_json["Level"] = Level
        msg = ""
        for raw in default_json.keys():
            if default_json[raw] != None:
                msg += str(default_json[raw]) + "|"
            else:
                msg += "|"
        print(msg[:-1])

        producer.send("testapplog_nlp_multi_label_service", msg[:-1], partition=0)
        producer.close()


if __name__ == "__main__":
    logg = ELK_Logging(
        [
            "10.231.128.34:9092",
            "10.231.128.35:9092",
            "10.231.128.36:9092",
            "10.231.128.37:9092",
            "10.231.128.38:9092",
        ],
        "nlp_multi_label_service",
    )
    for i in range(10):
        logg.producer("测试一下中文", "INFO", RequestID=i)
