import os
import numpy as np
import requests
import json


def get_info_from_apollo(config, NAME_SPACE, password_key):
    CONFIG_SERVER_URL = config.apollo.APOLLO_HOST
    APPID = config.apollo.APPID
    CLUSTER_NAME = config.apollo.CLUSTER
    TOKEN = config.apollo.TOKEN
    decrypt_url = config.apollo.DECRYPT_HOST
    api_key = config.apollo.API_KEY
    
    # 从apollo获取NAME_SPACE的配置信息
    url = (
        "{config_server_url}/configfiles/json/{appId}/{clusterName}+{token}/"
        "{namespaceName}".format(
            config_server_url=CONFIG_SERVER_URL,
            appId=APPID,
            clusterName=CLUSTER_NAME,
            token=TOKEN,
            namespaceName=NAME_SPACE,
        )
    )

    res = requests.get(url=url, timeout=10)
    es_index_info = json.loads(res.text)  # dict

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    
    # apollo获取解密后的密码
    headers = {
        "Content-Type": "application/json",
        "X-Gaia-API-Key": api_key,
    }  # X-Gaia-API-Key为PaaS平台上申请的对应key

    with open("/etc/apollo/apollo_private_key", "r") as f:
        PRIVATE_KEY = f.read()

    body = {
        "privateKey": PRIVATE_KEY,
        "cipherText": [es_index_info[password_key]],
    }

    res = requests.post(url=decrypt_url, headers=headers, data=json.dumps(body))
    es_index_info[password_key] = json.loads(res.text)[0]

    return es_index_info


