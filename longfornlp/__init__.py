# -*- coding=utf-8 -*-
"""
# library: jionlp
# author: dongrixinyu
# license: Apache License 2.0
# email: dongrixinyu.89@163.com
# github: https://github.com/dongrixinyu/JioNLP
# description: Preprocessing & Parsing tool for Chinese NLP
# website: www.jionlp.com/
"""

__version__ = '1.4.16'


import os

from longfornlp.util.logger import set_logger
from longfornlp.util.zip_file import unzip_file, UNZIP_FILE_LIST


logging = set_logger(level='INFO', log_dir_name='.longfornlp_logs')

# unzip dictionary files
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
for file_name in UNZIP_FILE_LIST:
    if not os.path.exists(os.path.join(DIR_PATH, 'dictionary', file_name)):
        zip_file = '.'.join(file_name.split('.')[:-1]) + '.zip'
        unzip_file(zip_file)


history = """
"""


from longfornlp.util import *
from longfornlp.dictionary import *
from longfornlp.rule import *
from longfornlp.gadget import *
from longfornlp.textaug import *
from longfornlp.algorithm import *

