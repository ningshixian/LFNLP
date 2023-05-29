from .cws_data_converter import tag2word, word2tag
from .measure import F1
from bio2bioes_converter import *
from bio2range_converter import *
from . import jieba
from .jieba import posseg as pseg

print("加载自定义分词词典...")
jieba.load_userdict("jieba/dict.txt")

f1 = F1()
