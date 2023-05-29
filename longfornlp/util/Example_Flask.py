import traceback
import os
from flask import send_from_directory
from flask import Flask
from flask import url_for, request, jsonify
from flask import render_template, redirect, flash
from flask import make_response, send_file
from flask.logging import create_logger
from flask import current_app
from werkzeug.utils import secure_filename
from logging.handlers import TimedRotatingFileHandler
from urllib.parse import quote
import logging
import mimetypes

app = Flask(__name__)

"""
参考
https://www.jianshu.com/p/8daa3d011cfd
"""

handler = logging.FileHandler("flask.log", encoding="UTF-8")
# you will need to set log level for app.logger to let your handler get the INFO level message
handler.setLevel(logging.DEBUG)
logging_format = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s"
)
handler.setFormatter(logging_format)
# LOG = create_logger(app)
# LOG.addHandler(handler)
app.logger.addHandler(handler)
app.logger.debug("get error msg:" + str(e) + str(request))
app.logger.debug("error traceback:" + traceback.format_exc())

app.debug = True
app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "download")  # 上传文件要储存的目录
app.secret_key = "super secret key"
app.run(host="0.0.0.0", port=port, threaded=True)  # 局域网可见的服务器，端口设置，在服务器部署


@app.route("/")
def say_hello():
    app.logger.info("An info message")
    app.logger.debug("A debug message")
    app.logger.error("An error message")
    return "hello"


@app.route("/slot_extract", methods=["GET", "POST"])
def slot_extract():
    if request.method == "GET":
        # return app.send_static_file('index.html')  # 默认 /static
        return render_template("index.html")  # 默认 /templates
    elif request.method == "POST":
        try:
            d = request.get_data(as_text=True)  # 避免请求数据中带有'\r' '\n'的情况
            d = clean_request(d)
            json_data = json.loads(d)
            name_set = json_data.get("slot_set")
            sen = json_data.get("text", "")
            ...
        except Exception as e:
            print(str(e))


# url_for()函数对于动态构建特定函数的URL非常有用
url_for("hello_guest", guest=name)


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    """Flask下载文件
    
    Parameters
    ----------
    filename : 保存文件路径
    """
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    directory = os.getcwd()  # 假设在当前目录
    response = make_response(send_from_directory(directory, filename, as_attachment=True))
    response.headers["Content-Disposition"] = "attachment; filename={0}; filename*=utf-8''{0}".format(quote(filename))
    return response


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    """Flask下载文件-接口返回文件数据流
    
    Parameters
    ----------
    filename : 保存文件路径
    
    """
    url = "http://xx.xxx.x.xx:60008/gateway/fanhua/" + downUrl
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36",
        "Accept": "*/*",
        "Content-Type": "application/json",
    }
    r = requests.get(url=url, headers=headers)
    if r.status_code != 200:
        raise Exception("Cannot connect with oss server or file is not existed")
    response = make_response(r.content)
    mime_type = mimetypes.guess_type(downUrl)[0]
    response.headers["Content-Type"] = mime_type
    response.headers["Content-Disposition"] = "attachment; filename={}".format(downUrl.encode().decode("latin-1"))
    return response

