import json
import requests


def longxin(lx_url, msg):
    try:
        headers = {"Content-Type": "application/json"}
        user_list = ["ningshixian"]
        myobj = json.dumps({"users": user_list, "message": msg})
        x = requests.post(lx_url, data=myobj, headers=headers, timeout=1)
        x = x.json()
        if x["code"] == 200:
            print("【{}】龙信通知: {}".format(env, msg))
    except Exception as e:
        print(e)


def dingding(content, url):
    """钉钉-告警推送"""
    header = {
        "Content-Type": "application/json",
        "Charset": "UTF-8"
    }
    msg = {
        "msgtype": "markdown",  # "text"
        "markdown": {
            "title":"钉钉报警", # 发消息时必须得带有"报警"关键词
            "text": content
        },
        "at": {"atMobiles": [""], "isAtAll": False},
    }
    # 对请求的数据进行json封装
    message_json = json.dumps(msg)
    try:
        res = requests.post(url=url, data=message_json, headers=header).json()
    except Exception as e:
        print(e)
    if res["errcode"] == 0:
        print("发送钉钉成功！")
    else:
        print("发送钉钉失败!", res)


# env = "prod"
# content = "你好，我是消息推送"
# dd_url = "https://oapi.dingtalk.com/robot/send?access_token=b76ce9429362c7697d34db88d32176ff1b6682286362af82c53fa18394d91ba5"
# lx_url = "https://aicare.longfor.com/gateway/mobile-api/db/external/sendLongChatMsg"    # prod

# # 龙信小秘书消息推送
# longxin(lx_url, content)
# # 钉钉推送消息
# dingding(content, dd_url)
