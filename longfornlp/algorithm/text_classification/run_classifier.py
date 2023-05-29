# -*- coding:utf-8 -*-
import os
import re
import time
import numpy as np
from utils import readExcel
from sklearn.model_selection import train_test_split  # 将数据分为测试集和训练集
from sklearn.externals import joblib
import codecs
from tqdm import tqdm
import pickle

"""
任务，需要判断一句话是否为闲聊类型

如何来完成这样一个分类的任务？

关于文本分类可以用很多中方式如逻辑回归等

闲聊分类器，训练过程：
1、读取业务/闲聊知识库，拿到训练/验证数据
2、打乱数据顺序
3、获取句子向量
4、SVM训练 && 打分 && 保存模型
5、对测试集进行预测
"""

CHAT_CORPUS = "data/闲聊知识库.xlsx"
MAIN_CORPUS = "data/业务知识主问题.xlsx"
USER_CORPUS = "data/用户日志问题.xlsx"

TEST_MODE = False  # 是否开启测试模式（如果是：不加载训练好的分类模型&&只取20个训练数据）
USE_XIAOHUANGJI = False  # 是否使用小黄鸡闲聊语料
if USE_XIAOHUANGJI:
    CLF_KEY = "gradient_boost"
else:
    CLF_KEY = "linearsvm"

# 分类模型的保存路径
MODEL_SAVAPATH = "model/" + CLF_KEY + "_model.m"
# 训练数据的保存路径
TRAIN_DATA_PKL = "data/train.pkl"
# 测试数据的保存路径
TEST_DATA_PKL = "data/test.pkl"

encoding = "utf-8"

# 开启BERT服务
from bert_serving.client import BertClient

# bc = BertClient(ip='10.240.4.47', port=5555, port_out=5556, timeout=600000, check_version=False)
bc = BertClient(
    ip="10.240.4.47",
    port=5555,
    port_out=5556,
    timeout=-1,
    #     timeout=600000,
    check_version=False,
    check_length=False,
    check_token_info=False,
)

# 仅针对测试集（用户日志问题）
def match(sen):
    if "http" in sen or "<" in sen:
        return True
    # 匹配数字、字母、中文、英文标点特殊符号
    # Python中字符串前面加上 r 表示原生字符串, 不用担心漏写反斜杠
    if re.match("^[0-9a-zA-Z \s+\.\!\/_,$%^*(+\"']+|[+——！，。？、~@#￥%……&*（）]+$", sen):
        return True
    return False


def test():

    print("读取测试集")
    x_test1 = []
    chat_res = readExcel(USER_CORPUS, ["user_input"])
    for item in chat_res:
        if match(str(item[0])):
            continue
        x_test1.append(str(item[0]))
    x_test = list(sorted(set(x_test1), key=x_test1.index))
    with codecs.open(USER_CORPUS+'.txt' ,'w', 'utf-8') as f:
        for line in x_test:
            f.write(line + '\n')

    for char in ["", " ", "\n"]:
        if char in x_test:
            x_test.remove(char)

    print("测试集向量化，并保存")
    if os.path.exists(TEST_DATA_PKL):
        with codecs.open(TEST_DATA_PKL, "rb") as f:
            x_test_vector = pickle.load(f)
    else:
        # 获取句子向量
        t1 = time.time()
        x_test_vector = bc.encode(x_test)
        print("bert向量化时间：", (time.time() - t1))  # 495.2990634441376
        with codecs.open(TEST_DATA_PKL, "wb") as f1:
            pickle.dump(x_test_vector, f1)

    assert len(x_test) == len(x_test_vector)
    print("测试集数量：", len(x_test_vector), " 一条数据的维度：", len(x_test_vector[0]))
    # 测试集数量： 26317  一条数据的维度： 768

    # 加载模型并预测
    clf = joblib.load(MODEL_SAVAPATH)
    predictions = clf.predict(x_test_vector)
    print(predictions)  # 预测

    # 保存预测结果
    with codecs.open("result/predictions.txt", "w+", encoding=encoding) as f:
        for i in range(len(predictions)):
            if predictions[i] == 1:
                f.write(x_test[i])
                f.write("\n")

# 若已存在训练好的模型，直接测试
if os.path.exists(MODEL_SAVAPATH):
    test()
    exit()


# 否则，训练
def get_train_val():
    x_train = []
    y_train = []

    # 闲聊知识库
    chat_res = readExcel(CHAT_CORPUS, ["primary_question", "similar_question"])
    for item in chat_res:
        x_train.append(item[0])
        similar_questions = str(item[1]).split("###")
        x_train.extend(similar_questions)

    x_train = list(set(x_train))
    for char in ["", " ", "\n"]:
        if char in x_train:
            x_train.remove(char)
    y_train.extend([1] * len(x_train))

    # 小黄鸡闲聊语料
    if USE_XIAOHUANGJI:
        temp_data = []
        with codecs.open("data/xiaohuangji50w_nofenci.conv", encoding=encoding) as f:
            for line in tqdm(f):
                if line.startswith("E"):
                    flag = True
                elif line.startswith("M") and flag:
                    flag = False
                    line = " ".join(line.split()[1:])
                    temp_data.append(line)

        temp_data = list(set(temp_data))
        for char in ["", " ", "\n"]:
            if char in temp_data:
                temp_data.remove(char)
        y_train.extend([1] * len(temp_data))
        x_train.extend(temp_data)
        assert len(x_train) == len(y_train)

    # 业务知识库
    temp_data = []
    chat_res = readExcel(MAIN_CORPUS, ["primary_question", "similar_question"])
    for item in chat_res:
        temp_data.append(item[0])
        similar_questions = str(item[1]).split("###")
        temp_data.extend(similar_questions)

    temp_data = list(set(temp_data))
    for char in ["", " ", "\n"]:
        if char in temp_data:
            temp_data.remove(char)
    y_train.extend([0] * len(temp_data))
    x_train.extend(temp_data)
    assert len(x_train) == len(y_train)

    # 统计
    print("闲聊问题: ", y_train.count(1))  # 2731
    print("业务问题: ", y_train.count(0))  # 28223

    # 切分训练/验证集
    x_val = x_train[:200] + x_train[-2000:]
    x_train = x_train[200:-2000]
    y_val = y_train[:200] + y_train[-2000:]
    y_train = y_train[200:-2000]

    return x_train, x_val, y_train, y_val


# 打乱数据顺序
def shuffle_in_unison(x, y):
    assert len(x) == len(y)
    shuffled_x = [0] * len(x)
    shuffled_y = [0] * len(y)
    permutation = np.random.permutation(len(x))
    for old_index, new_index in enumerate(permutation):
        shuffled_x[new_index] = x[old_index]
        shuffled_y[new_index] = y[old_index]
    return shuffled_x, shuffled_y


if os.path.exists(TRAIN_DATA_PKL):
    with codecs.open(TRAIN_DATA_PKL, "rb") as fp:
        (x_train_vector, x_val_vector, y_train, y_val) = pickle.load(fp)
else:
    # 处理数据集，得到训练集和验证集。 
    x_train, x_val, y_train, y_val = get_train_val()
    # 打乱数据
    x_train, y_train = shuffle_in_unison(x_train, y_train)
    x_val, y_val = shuffle_in_unison(x_val, y_val)

    # 获取句子向量
    t1 = time.time()
    x_train_vector = bc.encode(x_train)
    y_train = np.array(y_train)
    x_val_vector = bc.encode(x_val)
    y_val = np.array(y_val)
    print("bert向量化时间：", (time.time() - t1))  # 5534s->1h32m

    # # 对特征数据进行标准化处理 
    # from sklearn.preprocessing import StandardScaler 
    # sc = StandardScaler() 
    # sc.fit(x_train+x_val) 
    # x_train = sc.transform(x_train) 
    # x_val = sc.transform(x_val)  

    # Pickle dictionary using protocol 0.
    with codecs.open(TRAIN_DATA_PKL, "wb") as fp:
        pickle.dump((x_train_vector, x_val_vector, y_train, y_val), fp)

if TEST_MODE:
    x_train_vector, x_val_vector, y_train, y_val = (
        x_train_vector[:20],
        x_val_vector[:20],
        y_train[:20],
        y_val[:20],
    )
print("训练集数量：", len(x_train_vector))  # 28754
print("验证集数量：", len(x_val_vector))  # 2200


from sklearn.linear_model import Perceptron 
from sklearn import tree, svm, naive_bayes, neighbors
from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)

clfs = {
    "svm": svm.SVC(max_iter=-1, verbose=False, gamma="atuo"),
    "linearsvm": svm.LinearSVC(max_iter=40000, verbose=True),
    "perceptron": Perceptron(n_iter=40, eta0=0.1, random_state=0),
    "decision_tree": tree.DecisionTreeClassifier(),
    "naive_gaussian": naive_bayes.GaussianNB(),
    "naive_mul": naive_bayes.MultinomialNB(),
    "K_neighbor": neighbors.KNeighborsClassifier(),
    "bagging_knn": BaggingClassifier(
        neighbors.KNeighborsClassifier(), max_samples=0.5, max_features=0.5
    ),
    "bagging_tree": BaggingClassifier(
        tree.DecisionTreeClassifier(), max_samples=0.5, max_features=0.5
    ),
    "random_forest": RandomForestClassifier(n_estimators=50),
    "adaboost": AdaBoostClassifier(n_estimators=50),
    "gradient_boost": GradientBoostingClassifier(
        n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0
    ),
}

# 训练 && 保存模型
def try_different_method(clf):
    clf.fit(x_train_vector, y_train.ravel(), sample_weight=None)

    score = clf.score(
        x_val_vector, y_val.ravel()
    )  # 返回给定测试数据和标签的平均精度 0.9790051679586563
    print("the score is :", score)  # 0.980909090909091

    # 预测并评估分类器性能
    from sklearn.metrics import accuracy_score 
    y_pred = clf.predict(x_val_vector)  
    acc_score = accuracy_score(y_val, y_pred)
    print(acc_score)
    
    # 保存分类模型
    joblib.dump(clf, MODEL_SAVAPATH)


# for clf_key in clfs.keys():
#     print('the classifier is :',clf_key)
#     clf = clfs[clf_key]
#     try_different_method(clf)

t2 = time.time()
clf = clfs[CLF_KEY]
print("the classifier is :", CLF_KEY)
# 训练分类器 
try_different_method(clf)
print("training time:", time.time() - t2)  # 153s

# 对测试集进行预测
test()
