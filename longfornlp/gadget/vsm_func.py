# -*- coding: UTF-8 -*-
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf(corpus, stpwrdlst=None):
    """获取tfidf向量
    TF(t) = 某个词在文档出现的次数 / 文档中词语的总数 
    IDF(t) = Log_e(文档总数 / 出现该词语的文档总数+1)    # +1的原因是避免分母为0
    平滑版 IDF(t) = Log_e(文档总数+1 / 出现该词语的文档总数+1) +1
    TF-IDF = TF * IDF

    参考：
    TfidfVectorizer 中文处理及一些使用参数 https://blog.csdn.net/blmoistawinde/article/details/80816179
    TfidfVectorizer 参数指南 https://scikit-learn.org.cn/view/728.html
    """
    # 1、实例化一个转换器类
    vectorizer = TfidfVectorizer(
        use_idf=True, 
        smooth_idf=True,    # 通过在文档频率上增加1来平滑idf权重
        # sublinear_tf=True,  # 应用次线性tf缩放，即将tf替换为1 + log(tf)
        # norm=None,      # 每个输出行都有单位范数
        max_df=0.6,     # 过滤出现在超过60%的句子中的词语
        # min_df=0.005,       # 过滤出现在小于0.5%的句子中的低频词（测试acc↓）
        token_pattern=r"(?u)\b\w+\b",    # 保留单字的词语
        max_features=8000,  # 词汇表大小
    )
    # vectorizer = TfidfVectorizer(analyzer='word', stop_words=stpwrdlst, max_df=0.9, min_df=3, sublinear_tf=True, max_features=8000)   # 
    # 2、训练
    vectorizer.fit(corpus)
    # print("A mapping of terms to feature indices: ", vectorizer.vocabulary_)
    # print("词袋: ", vectorizer.get_feature_names())
    # # 3、向量转换
    # print(vectorizer.transform(['你好啊']).toarray())
    return vectorizer


def bow(corpus, stpwrdlst):
    """获取词袋BOW向量"""
    # BOW
    vectorizer = CountVectorizer(analyzer='word', stop_words=stpwrdlst) # 过滤停用词
    vectorizer.fit(corpus)
    # print(vectorizer.vocabulary_)
    # print(vectorizer.transform(['我 爱 北京']).toarray())
    return vectorizer


def doc2vec(corpus, stpwrdlst):
    """
    corpus: 语料 ["我 爱 北京", "今天 天气 不错"]
    stpwrdlst: 停用词列表
    """
    x_train = []
    for i, text in enumerate(corpus):
        word_list = text.split(" ")
        word_list = list(filter(lambda x: x not in stpwrdlst, word_list))
        doc = gensim.models.doc2vec.TaggedDocument(word_list, tags=[i])
        x_train.append(doc)
    # 使用 Doc2Vec 建模
    model = Doc2Vec(
        x_train, min_count=1, window=3, size=300, sample=1e-3, nagative=5, workers=2
    )
    model.train(x_train, total_examples=model.corpus_count, epochs=300)
    # # 经过训练后，Doc2Vec会以字典的形式保存在model对象中，可以使用类似字典的方式直接访问获取
    # sen2docvec = {sen_list[i]: model.docvecs[i] for i in range(len(corpus))}
    # # 预测
    # v2 = model.infer_vector(doc_words='我 爱 上海'.split(' '), alpha=0.025, steps=500)
    return model


class BM25(object):
    """ Implementation of OKapi BM25 with sklearn's TfidfVectorizer
    Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)
    """
    def __init__(self, stpwrdlst=None, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False, stop_words=stpwrdlst)
        self.stpwrdlst = stpwrdlst
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        from scipy import sparse
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)
        
        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1


# def tfw2v(corpora_documents, tfidf_vectorizer, voc):
#     """词向量通过idf加权平均后计算文本向量
#     预测时，无法获得每个词对应的tfidf权值！
#     https://blog.csdn.net/qq_33624866/article/details/106430352
#     """
#     sen2tfw2v = {}
#     tfidf_vocabulary = tfidf_vectorizer.vocabulary_  # {'xx':34}
#     for s1 in corpora_documents:
#         v1 = []
#         tfidf_vector = {}
#         word_tfidfs=[tfidf_vector[0,tfidf_vocabulary.get(word)] if tfidf_vocabulary.get(word) else 0 for word in s1.split()]
#         word_tfidf_map={word:tfidf_val for word,tfidf_val in zip(words,word_tfidfs)}
#         # 1. 词向量idf加权平均
#         for s in s1.split(' '):
#             idf_v = idf.get(s, 1)
#             if s in voc:
#                 v1.append(1.0 * idf_v * voc[s])
#         v1 = np.array(v1).mean(axis=0)
#         sen2tfw2v[''.join(s1.split(' '))] = v1
#     return sen2tfw2v


# if __name__ == "__main__":
    
#     # 训练样本
#     raw_documents = [
#         "我爱北京",
#         "我爱上海",
#         "不到长城非好汉",
#         "0南京江心洲污泥偷排”等污泥偷排或处置不当而造成的污染问题，不断被媒体曝光",
#         "1面对美国金融危机冲击与国内经济增速下滑形势，中国政府在2008年11月初快速推出“4万亿”投资十项措施",
#         "2全国大面积出现的雾霾，使解决我国环境质量恶化问题的紧迫性得到全社会的广泛关注",
#         "3大约是1962年的夏天吧，潘文突然出现在我们居住的安宁巷中，她旁边走着40号王孃孃家的大儿子，一看就知道，他们是一对恋人。那时候，潘文梳着一条长长的独辫",
#         "4坐落在美国科罗拉多州的小镇蒙特苏马有一座4200平方英尺(约合390平方米)的房子，该建筑外表上与普通民居毫无区别，但其内在构造却别有洞天",
#         "5据英国《每日邮报》报道，美国威斯康辛州的非营利组织“占领麦迪逊建筑公司”(OMBuild)在华盛顿和俄勒冈州打造了99平方英尺(约9平方米)的迷你房屋",
#         "6长沙市公安局官方微博@长沙警事发布消息称，3月14日上午10时15分许，长沙市开福区伍家岭沙湖桥菜市场内，两名摊贩因纠纷引发互殴，其中一人被对方砍死",
#         "7乌克兰克里米亚就留在乌克兰还是加入俄罗斯举行全民公投，全部选票的统计结果表明，96.6%的选民赞成克里米亚加入俄罗斯，但未获得乌克兰和国际社会的普遍承认",
#         "8京津冀的大气污染，造成了巨大的综合负面效应，显性的是空气污染、水质变差、交通拥堵、食品不安全等，隐性的是各种恶性疾病的患者增加，生存环境越来越差",
#         "9 1954年2月19日，苏联最高苏维埃主席团，在“兄弟的乌克兰与俄罗斯结盟300周年之际”通过决议，将俄罗斯联邦的克里米亚州，划归乌克兰加盟共和国",
#         "10北京市昌平区一航空训练基地，演练人员身穿训练服，从机舱逃生门滑降到地面",
#         "11腾讯入股京东的公告如期而至，与三周前的传闻吻合。毫无疑问，仅仅是传闻阶段的“联姻”，已经改变了京东赴美上市的舆论氛围",
#         "12国防部网站消息，3月8日凌晨，马来西亚航空公司MH370航班起飞后与地面失去联系，西安卫星测控中心在第一时间启动应急机制，配合地面搜救人员开展对失联航班的搜索救援行动",
#         "13新华社昆明3月2日电，记者从昆明市政府新闻办获悉，昆明“3·01”事件事发现场证据表明，这是一起由新疆分裂势力一手策划组织的严重暴力恐怖事件",
#         "14在即将召开的全国“两会”上，中国政府将提出2014年GDP增长7.5%左右、CPI通胀率控制在3.5%的目标",
#         "15中共中央总书记、国家主席、中央军委主席习近平看望出席全国政协十二届二次会议的委员并参加分组讨论时强调，团结稳定是福，分裂动乱是祸。全国各族人民都要珍惜民族大团结的政治局面，都要坚决反对一切危害各民族大团结的言行",
#     ]
#     print(sbert("我爱旅游", raw_documents))
