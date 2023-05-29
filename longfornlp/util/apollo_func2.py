# -*- encoding: utf-8 -*-
"""
@Author  : Wu Wenjie
@Contact : 406959268@qq.com
@Software: PyCharm
@File    : _apollo_config.py
@Time    : 2021/8/18 16:59
@License : (C) Copyright 2019, by Wu Wenjie
@Desc    :

"""
import json
from concurrent.futures import ThreadPoolExecutor
import requests
from config.LogConfig import logger
from util.RsaUtil import RsaUtil
from Args_util import ARGS


# todo: 起一个线程就行，线程池没用
_THREAD_POOL = ThreadPoolExecutor(max_workers=20)


class ApolloCfg(object):
    def __init__(self,
                 apollo_config: dict = None,
                 env: str = "sit",
                 private_key_path: str = None,
                 api_key: str = None):
        """
        :param data: dict，apollo上拉下来的配置，格式为“SECTION.KEY=XXX”
        :param apollo_config: dict，apollo配置，必传字段{uri，app_id，cluster_name，token，name_space}
        :param env: str，默认为sit
        :param private_key: str，apollo私钥，用于解密
        """
        if env not in ("sit", "prod", "pre"):
            raise KeyError("{} for env".format(env))
        self._env = env
        self._private_key = None
        self._api_key = api_key
        self._apollo_config = apollo_config
        self._stop = False
        self._notification_map = {self._apollo_config["name_space"]: -1}
        self._data = None
        self._rsa_util = RsaUtil(apollo_private_key_path=private_key_path)
        self._init_from_apollo(self._apollo_config)
        self.start_listen()

    def _init_from_apollo(self, apollo_config):
        # apollo获取配置
        url = ("{server_url}/configs/{app_id}/{cluster_name}+{token}/"
               "{name_space}".format(server_url=apollo_config["uri"],
                                     app_id=apollo_config["app_id"],
                                     cluster_name=apollo_config["cluster_name"],
                                     token=apollo_config["token"],
                                     name_space=apollo_config["name_space"]
                                     ))

        res = requests.get(url=url)

        self._data = json.loads(res.text).get('configurations', {})
        self._decrypt()

    def stop_listen(self):
        self._stop = True

    def get_env(self):
        return self._env

    def get_data(self):
        return self._data

    def get(self, key: str, fallback=None):
        if key not in self._data and fallback is not None:
            return fallback
        else:
            return self._data[key]

    def getint(self, key: str, fallback=None):
        return int(self.get(key, fallback))

    def items(self):
        return [(x, self._data[x]) for x in self._data]

    def _heart_listen(self):
        while not self._stop:
            notifications = [{
                'namespaceName': x,
                'notificationId': self._notification_map[x]
            } for x in self._notification_map]
            self.listen_params = {
                'appId': self._apollo_config["app_id"],
                'cluster': self._apollo_config["cluster_name"],
                'notifications': json.dumps(notifications)
            }
            try:
                _heart_url = "{server_url}/notifications/v2".format(server_url=self._apollo_config["uri"])
                r = requests.get(url=_heart_url, params=self.listen_params, timeout=1)
            # todo: 通用的异常处理，如果apollo重启了，工程不会挂
            except requests.exceptions.ReadTimeout:
                pass
            else:
                if r.status_code == 200:
                    logger.info("配置发生变更，尝试重载配置...")
                    configs = r.json()
                    for config in configs:
                        self._notification_map[config["namespaceName"]] = config["notificationId"]
                        self._init_from_apollo(self._apollo_config)
                    logger.info("配置重载成功!")
                else:
                    logger.error("配置变更发生错误，响应消息为：{}".format(r.text))

    def start_listen(self):
        _THREAD_POOL.submit(self._heart_listen)

    # todo: 改成封装公司官方的解密方法
    def _decrypt(self):
        for x in self._data:
            try:
                self._data[x] = self._rsa_util.decrypt_by_private_key(self._data[x])
            except Exception:
                continue


# CFG = ApolloCfg(apollo_config={"uri": ARGS.apollo_uri,
#                                "app_id": ARGS.apollo_app_id,
#                                "cluster_name": ARGS.apollo_cluster,
#                                "token": ARGS.apollo_token,
#                                "name_space": ARGS.apollo_namespace},
#                 api_key=ARGS.apollo_key,
#                 env=ARGS.environment,
#                 private_key_path=ARGS.apollo_private_key_path)


# def test():
#     global CFG
#     while True:
#         print(CFG.getint("SERVER.port"))