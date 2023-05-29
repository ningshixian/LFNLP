from collections import OrderedDict
import jieba
from gensim.models import Word2Vec, Doc2Vec
import numpy as np
import codecs
from tqdm import tqdm

"""
Gensim Tutorials
https://radimrehurek.com/gensim/models/keyedvectors.html
"""


def w2v_train():
    """w2v模型训练"""
    with open("d:/天龙八部.txt", errors="ignore", encoding="utf-8") as fp:
        lines = fp.readlines()
        for line in lines:
            seg_list = jieba.cut(line)
            with open("d:/分词后的天龙八部.txt", "a", encoding="utf-8") as ff:
                ff.write(" ".join(seg_list))  # 词汇用空格分开

    # 加载语料
    sentences = Word2Vec.Text8Corpus("d:/分词后的天龙八部.txt")

    # 训练模型
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    word_vectors = model.wv

    # 选出最相似的10个词
    for e in model.most_similar(positive=["乔峰"], topn=10):
        print(e[0], e[1])

    # 保存模型
    model.save("d:/天龙八部.model")

    # 加载模型
    model = Word2Vec.load("d:/天龙八部.model")


def load_embedding(embedding_path, embedding_dim, format, file_type, with_head=False, word_set=None):
    """
    Args:
        format: 'glove', 'word2vec', 'fasttext'
        file_type: 'text' or 'binary'
    """
    from gensim.models.keyedvectors import KeyedVectors
    import FastText

    embedding_dict = dict()

    if format == "word2vec" or format == "fasttext":
        if file_type == "text":
            vector_total = KeyedVectors.load_word2vec_format(embedding_path, binary=False, unicode_errors="ignore")
        else:
            if format == "word2vec":
                vector_total = KeyedVectors.load_word2vec_format(embedding_path, binary=True, unicode_errors="ignore")
            elif format == "fasttext":
                vector_total = FastText.load_fasttext_format(embedding_path, encoding="utf8")

        assert vector_total.vector_size == embedding_dim
        if word_set is None:
            embedding_dict = vector_total
        else:
            if not (format == "fasttext" and file_type == "binary"):
                word_total = vector_total.index2word  # actually, vector_total.index2word is the word list
            else:
                word_total = vector_total.wv.index2word
            for word in word_total:
                if word in word_set:
                    embedding_dict[word] = vector_total[word]
    elif format == "glove":
        with codecs.open(embedding_path, "r", encoding="utf-8") as fin:
            if with_head == True:
                _ = fin.readline()
            for idx, line in enumerate(fin):
                line = line.rstrip()
                if idx == 0 and len(line.split()) == 2:
                    continue
                if len(line) > 0:
                    word, vec = line.split(" ", 1)
                    if (word_set and word in word_set) or (word_set is None):
                        vector = [float(num) for num in vec.split(" ")]
                        assert len(vector) == embedding_dim
                        embedding_dict[word] = vector
    else:
        raise Exception("The format supported are glove, word2vec or fasttext, dost not support %s now." % format)
    return embedding_dict


def readBinEmbedFile(embFile, word_size):
    """
    读取二进制格式保存的词向量文件
    """
    import word2vec

    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.1, 0.1, word_size)

    model = word2vec.load(embFile)
    print("加载词向量文件完成")
    for i in tqdm(range(len(model.vectors))):
        vector = model.vectors[i]
        word = model.vocab[i].lower()  # convert all characters to lowercase
        embeddings[word] = vector
    return embeddings


def readTxtEmbedFile(embFile, word_size):
    """
    读取预训练的词向量文件 
    """
    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(word_size)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.1, 0.1, word_size)
    embeddings["NUMBER"] = np.random.uniform(-0.1, 0.1, word_size)

    with codecs.open(embFile, "r", "utf-8") as f:
        for line in tqdm(f):
            if len(line.split()) <= 2:
                continue
            values = line.strip().split()
            word = values[0].lower()
            vector = np.asarray(values[1:], dtype=np.float32)
            embeddings[word] = vector
    
    return embeddings
