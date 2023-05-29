# from .time_parser import TimeParser
from .split_sentence import SplitSentence
from .remove_stopwords import RemoveStopwords
from .ts_conversion import TSConversion
from .pinyin import Pinyin
# from .char_radical import CharRadical
from .similarity import *
from .sort_func import *
from .string_func import *
from .vsm_func import *
from .trie_tree import TrieTree
from .Time_NLP.TimeNormalizer import TimeNormalizer
from .Time_NLP.StringPreHandler import StringPreHandler


split_sentence = SplitSentence()
remove_stopwords = RemoveStopwords()
tra_sim_conversion = TSConversion()
tra2sim = tra_sim_conversion.tra2sim
sim2tra = tra_sim_conversion.sim2tra
pinyin = Pinyin()

# del tra_sim_conversion

# tn = TimeNormalizer(isPreferFuture=False)  # 初始化
# sent = self.time_filter(sent)  # 过滤不合法的时间词
# sent = StringPreHandler.numberTranslator(sent)  # 中文数字转化成阿拉伯数字
# tn.save_diff(self.text, sent)    # 不改变原文，还原字符串 tn.map_dict_temp={'23': '二十三'}
# res_json = tn.parse(target=sent, timeBase=arrow.now())
