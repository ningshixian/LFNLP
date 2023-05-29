#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/20 16:39
# @Author  : zhm
# @File    : TimeNormalizer.py
# @Software: PyCharm
import pickle
import regex as re
import arrow
import json
import sys, os
import difflib
from utils.Time_NLP.StringPreHandler import StringPreHandler
from utils.Time_NLP.TimePoint import TimePoint
from utils.Time_NLP.TimeUnit import TimeUnit


# 时间表达式识别的主要工作类
class TimeNormalizer:
    def __init__(self, isPreferFuture=True):
        self.isPreferFuture = isPreferFuture
        self.map_dict = {
            "中旬": "15号",
            "傍晚": "午后",
            "五一": "劳动节",
            "白天": "早上",
            "礼拜": "星期",
            "：": ":",
            "明后天": "后天",
            # "等会": "稍晚",
            # "等下": "稍晚",
            # "待会": "稍晚",
            # "等一下": "稍晚",
            # "过会": "稍晚",
            # "过会儿": "稍晚",
            # "一会": "稍晚",
            # "马上": "稍晚",
        }
        self.map_dict_temp = {}  # 用于还原字符串
        self.pattern, self.holi_solar, self.holi_lunar = self.init()

    def save_diff(self, old, new):
        # TODO:获取两个字符串之间的不同子串，用于还原字符串
        s = difflib.SequenceMatcher(None, new, old)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            # print('%6s a[%d:%d] b[%d:%d]' % opcode)  # opcode=('equal', 0, 2, 0, 2)
            if tag != "equal" and new[i1:i2].isdigit():
                self.map_dict_temp[new[i1:i2]] = old[j1:j2]

    # 这里对一些不规范的表达做转换
    def _filter(self, input_query):
        # # 这里将汉字表示的数字转化为用阿拉伯数字表示的数字
        # input_query = StringPreHandler.numberTranslator(input_query)

        # 这里将‘3月3’转换为‘3月3号’
        rule = r"[0-9]{1,2}月[0-9]{1,2}"
        pattern = re.compile(rule)
        match = pattern.search(input_query)
        if match != None:
            index = input_query.find("月")
            rule = u"日|号"
            pattern = re.compile(rule)
            match = pattern.search(input_query[index:])
            if match == None:
                rule = u"[0-9]月[0-9]+"
                pattern = re.compile(rule)
                match = pattern.search(input_query)
                if match != None:
                    end = match.span()[1]
                    input_query = input_query[:end] + "号" + input_query[end:]
        # else:
        #     # 这里对于下个周末这种做转化 把个给移除掉
        #     input_query = input_query.replace("个", "")

        for k, v in self.map_dict.items():
            if k in input_query:
                input_query = input_query.replace(k, v)
                self.map_dict_temp[v] = k
        return input_query

    def init(self):
        fpath = os.path.dirname(__file__) + "/resource/reg.pkl"
        try:
            with open(fpath, "rb") as f:
                pattern = pickle.load(f)
        except:
            with open(os.path.dirname(__file__) + "/resource/regex.txt", "r", encoding="utf-8") as f:
                content = f.read()
            p = re.compile(content)
            with open(fpath, "wb") as f:
                pickle.dump(p, f)
            with open(fpath, "rb") as f:
                pattern = pickle.load(f)
        with open(os.path.dirname(__file__) + "/resource/holi_solar.json", "r", encoding="utf-8") as f:
            holi_solar = json.load(f)
        with open(os.path.dirname(__file__) + "/resource/holi_lunar.json", "r", encoding="utf-8") as f:
            holi_lunar = json.load(f)
        return pattern, holi_solar, holi_lunar

    def parse(self, target, timeBase=arrow.now()):
        """
        TimeNormalizer的构造方法，timeBase取默认的系统当前时间
        :param timeBase: 基准时间点
        :param target: 待分析字符串
        :return: 时间单元数组
        """
        self.isTimeSpan = False
        self.invalidSpan = False
        self.timeSpan = ""
        self.target = target
        self.target = self._filter(target)
        self.timeBase = arrow.get(timeBase).format("YYYY-M-D-H-m-s")
        # print(arrow.now())
        # print(timeBase)
        # print(self.timeBase)
        self.nowTime = timeBase
        self.oldTimeBase = self.timeBase
        # self.__preHandling()  # 可能会改变句子！
        self.timeToken, self._timeToken = self.__timeEx()
        # print([x.time.format("YYYY-MM-DD HH:mm:ss") for x in self.timeToken])
        # print([x for x in self._timeToken])
        dic = {}
        res = self.timeToken
        self.isTimeSpan = False  # 忽略时间段

        if self.isTimeSpan:
            if self.invalidSpan:
                dic["error"] = "no time pattern could be extracted."
            else:
                result = {}
                dic["type"] = "timedelta"
                dic["timedelta"] = self.timeSpan
                # print(dic['timedelta'])
                index = dic["timedelta"].find("days")

                days = int(dic["timedelta"][: index - 1])
                result["year"] = int(days / 365)
                result["month"] = int(days / 30 - result["year"] * 12)
                result["day"] = int(days - result["year"] * 365 - result["month"] * 30)
                index = dic["timedelta"].find(",")
                time = dic["timedelta"][index + 1 :]
                time = time.split(":")
                result["hour"] = int(time[0])
                result["minute"] = int(time[1])
                result["second"] = int(time[2])
                dic["timedelta"] = result
        else:
            if len(res) == 0:
                dic["error"] = "no time pattern could be extracted."
            elif len(res) == 1:
                dic["type"] = "timestamp"
                dic["timestamp"] = res[0].time.format("YYYY-MM-DD HH:mm:ss")
                for k, v in self.map_dict_temp.items():  # 还原字符串 '15号' → '中旬'
                    self._timeToken[0] = self._timeToken[0].replace(k, v)
                dic["_timestamp"] = self._timeToken[0]
            else:
                dic["type"] = "timespan"
                dic["timespan"] = [x.time.format("YYYY-MM-DD HH:mm:ss") for x in res]
                # dic["timespan"] = [res[0].time.format("YYYY-MM-DD HH:mm:ss"), res[1].time.format("YYYY-MM-DD HH:mm:ss")]
                for k, v in self.map_dict_temp.items():  # 还原字符串 '15号' → '中旬'
                    self._timeToken = [x.replace(k, v) for x in self._timeToken]
                    # self._timeToken[0] = self._timeToken[0].replace(k, v)
                    # self._timeToken[1] = self._timeToken[1].replace(k, v)
                dic["_timespan"] = self._timeToken[:]
        return json.dumps(dic)

    def __preHandling(self):
        """
        待匹配字符串的清理空白符和语气助词以及大写数字转化的预处理
        :return:
        """
        self.target = StringPreHandler.delKeyword(self.target, u"\\s+")  # 清理空白符
        self.target = StringPreHandler.delKeyword(self.target, u"[的]+")  # 清理语气助词
        # self.target = StringPreHandler.numberTranslator(self.target)  # 中文数字转化

    def __timeEx(self):
        """
        :param target: 输入文本字符串
        :param timeBase: 输入基准时间
        :return: TimeUnit[]时间表达式类型数组
        """
        startline = -1
        endline = -1
        rpointer = 0
        temp = []

        match = self.pattern.finditer(self.target)
        for m in match:
            startline = m.start()
            if startline == endline:
                rpointer -= 1
                temp[rpointer] = temp[rpointer] + m.group()
            else:
                temp.append(m.group())
            endline = m.end()
            rpointer += 1
        res = []
        # 时间上下文： 前一个识别出来的时间会是下一个时间的上下文，用于处理：周六3点到5点这样的多个时间的识别，第二个5点应识别到是周六的。
        contextTp = TimePoint()
        # print(self.timeBase)
        # print('temp',temp)

        for i in range(0, rpointer):
            # 这里是一个类嵌套了一个类
            res.append(TimeUnit(temp[i], self, contextTp))
            # res[i].tp.tunit[3] = -1
            contextTp = res[i].tp
            # print(self.nowTime.year)
            # print(contextTp.tunit)
        res = self.__filterTimeUnit(res)

        return res, temp

    def __filterTimeUnit(self, tu_arr):
        """
        过滤timeUnit中无用的识别词。无用识别词识别出的时间是1970.01.01 00:00:00(fastTime=0)
        :param tu_arr:
        :return:
        """
        if (tu_arr is None) or (len(tu_arr) < 1):
            return tu_arr
        res = []
        for tu in tu_arr:
            if tu.time.timestamp != 0:
                res.append(tu)
        return res
