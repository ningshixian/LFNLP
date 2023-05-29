import pymysql 
import json
import requests
from pyMySQL_test import CaliburData
sql = CaliburData('2019-07-01', '2019-10-12')


# 发送携带请求头的POST请求给接口返回答案
def getAnswersFromDM(query, flow_code='LXHRSDH'):
    '''
        请求字段
        flow	String	是	流程编码
        session	String	否	会话ID，用于标识一个访问者的会话和保持上下文信息。 
        user	String	是	不同业务产品下提问者的唯一标识，不同业务产品下的user确定唯一的用户
        input	String	是	客户的输入
        project	String	是	客户归属项目编码，同一个问题在不同项目下可以对应不同的答案，需要在中台管理后台新建项目，获取项目编码；
    
        ҅ReactionElement→前端处理单元→各个单元可能有不同种类
        # text: 简单文本
        # Option：封装了一个问答对
        # Request: 表示一个回调单元，展示为一个链接
        # rich-text：富文本

        返回数据
        code	Int	状态码。成功为0或大于0的正整数，失败为对应的负数错误码
        msg	String	状态信息
        result	JsonObject	返回的具体内容
    '''

    # DM接口（线上）
    url = 'https://aicarelocal.longfor.com/dmapi/golem/next'
    # url = 'http://10.231.9.138:9588/api/golem/next'  # 138
    # url = 'https://algo01:9588/dmapi/golem/next'

    # DM接口（测试）
    url = 'http://10.231.9.140:9588/api/golem/next'
    url = 'https://aicarelocal.longfor.com/test_dmapi/golem/next'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Host': 'aicarelocal.longfor.com',
        'Connection': 'keep-alive'}

    # 接口参数（post）
    d = {"user": "<Invisible>",
         'project': '',
         'flow': flow_code,
         "channel":"地产",  #  如果不是人事的流程，channel传空字符串
         "session": "",
         "service":"bot",
         "data":{
            "Type": "Text",
            "Text": query
            },
        #  "text": query, 
         "nickname":"TEST",
         "oa_name":"OA",
         "client":"app",
         "content_type":"rich-text"
    } 

    d = json.dumps(d)   # 以json格式传入必要的参数
    r = requests.post(url=url, data=d, headers=headers)
    res = r.json()  # Requests中内置的JSON解码器  直接返回json 格式结果
    # print(res)

    response_questions = []
    if res['result']:
        sess = res['result']['session']    # 获取 session ID
        action = res['result']['action']   # faq(单轮) chat(闲聊) toSeat(坐席) ...
        if action=='noAnswer':
            response_questions.append('无答案')   # 模型回复为空的情况
        else:
            data = res['result']['data']
            for reaction_element in data:
                data_type = reaction_element['data_type']  # 接口返回数据类型
                oc_id = reaction_element['id']

                # 根据返回的id，通过表映射，拿到主问题
                primary_question = sql.get_primary_question(oc_id)  
                response_questions.append(primary_question[0])

                # if data_type=='text':
                #     response_questions.append(reaction_element['content'])
                # else:
                #     k = reaction_element['content']['key']
                #     # v = reaction_element['content']['value']
                #     response_questions.append(k)        

        # 释放session，是delete的
        url = 'https://aicarelocal.longfor.com/dmapi/golem/'+flow_code+'/traveller/'+sess+'?userInvisible'
        r = requests.delete(url=url)    
    return response_questions
