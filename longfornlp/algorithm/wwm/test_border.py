#! -*- coding: utf-8 -*-
# 预训练语料构建

import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import numpy as np
import pandas as pd
import tensorflow as tf
from bert4keras.snippets import parallel_apply
from bert4keras.backend import K

import re
import jieba

data = pd.read_csv('dc.csv', header=0)
# print(data.iloc[:,1])  #第2列所有数据
longforvocab = []
for index, row in data.iterrows():
    entity = row[1]
    longforvocab.append(entity)  # 1.longfor vocab
    jieba.add_word(entity)  # 2.jieba


def get_border(segment):
    # 输入一个句子，将可以组成龙湖专业词汇的字前打上标记'##'，便于中文整词mask
    cws_dict = {x: 1 for x in longforvocab}
    new_segment = []
    i = 0
    while i < len(segment):
        if len(re.findall('[\u4E00-\u9FA5]', segment[i])) == 0:
            new_segment.append(segment[i])
            i = i + 1
            continue
        has_add = False
        """
        1、为啥不直接词典匹配？效率太低
        2、‘##’改变了原始句子！可以记录一个专有词list变量来解决
        """
        for length in range(10, 0, -1):  # 无法保证最大检索词汇长度为10！
            if i + length > len(segment):
                continue
            tmp = ''.join(segment[i:i + length])
            if tmp in cws_dict:
                for l in range(0, length):
                    new_segment.append('##' + segment[i + l])
                i = i + length
                has_add = True
                break
        if not has_add:
            new_segment.append(segment[i])
            i = i + 1
    return new_segment



segment = jieba.lcut("我住在龙湖的正安飞洋世纪城，唐宁one", HMM=True)
print(segment)
print(get_border(segment))


# ============================================================================= #


from bert4keras.tokenizers import Tokenizer
tokenizer = Tokenizer("/Users/ningshixian/Desktop/longfor-project/corpus/chinese_L-12_H-768_A-12/vocab.txt", do_lower_case=True)
mask_rate = 1   # 0.15
token_mask_id = tokenizer._token_mask_id
vocab_size = tokenizer._vocab_size


def word_segment(text):
    return jieba.lcut(text)


def token_process(token_id):
    """以80%的几率替换为[MASK]，以10%的几率保持不变，
    以10%的几率替换为一个随机token。
    """
    rand = np.random.random()
    if rand <= 0.8:
        return token_mask_id
    elif rand <= 0.9:
        return token_id
    else:
        return np.random.randint(0, vocab_size)


def sentence_process(text):
    """单个文本的处理函数
    流程：分词，然后转id，按照mask_rate构建全词mask的序列
            来指定哪些token是否要被mask
    
    1、BERT的MLM（随机mask语料中15%的token，然后作为分类问题去预测masked token）
        - 80%的时间中：将选中的词用[MASK]token来代替，例如
        - 10%的时间中：将选中的词用任意的词来进行代替，例如
        - 10%的时间中：选中的词不发生变化
    """
    words = word_segment(text)
    rands = np.random.random(len(words))

    token_ids, mask_ids = [], []
    for rand, word in zip(rands, words):
        word_tokens = tokenizer.tokenize(text=word)[1:-1]
        word_token_ids = tokenizer.tokens_to_ids(word_tokens)
        token_ids.extend(word_token_ids)

        if word in longforvocab and rand < mask_rate:
            word_mask_ids = [
                token_process(i) + 1 for i in word_token_ids
            ]
        else:
            word_mask_ids = [0] * len(word_tokens)

        if token_mask_id+1 in word_mask_ids:
            word_mask_ids = [token_mask_id+1] * len(word_mask_ids)
        print(list(zip(word, word_mask_ids)))

        mask_ids.extend(word_mask_ids)

    return [token_ids, mask_ids]


sen_processed = sentence_process("我住在龙湖的正安飞洋世纪城，唐宁one")
# # ['[CLS]', '我', '住', '在', '龙', '湖', '的', '正', '安', '飞', '洋', '世', '纪', '城', '，', '唐', '宁', 'one', '[SEP]']
print(sen_processed)
