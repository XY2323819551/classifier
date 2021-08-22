#!/usr/bin/python
# -*- coding:utf-8 -*-

import re
import gensim
import jieba
import time
from typing import List
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.model_selection import KFold
import json

path = "/data/zhangxy/hello_pytorch/classifier/data/"


def excel_one_line_to_list(cols: List[int]):
    df = pd.read_excel(path + "resources/data_source6.xls", usecols=cols, names=None)  # 读取项目名称和行业领域两列，并不要列名
    return df.values.tolist()


def get_train_data():
    """
    #获取拼接后的训练数据x_data，并且保存，读取excel中的B和C列拼接起来
    """
    x_tmp = excel_one_line_to_list([1, 2])
    x_tmp2 = [x_tmp[i][0] + x_tmp[i][1] for i in range(len(x_tmp))]
    x_data = [x_tmp2[i].replace('\n', '').replace('\r', '') for i in range(len(x_tmp))]
    with open(path + "train_data.txt", "w", encoding="utf-8") as f:
        for line in x_data:
            f.write(line + '\n')


def get_label():
    y_tmp1 = excel_one_line_to_list([3])  # 得到所有的维修类别描述
    y_tmp2 = list(set([y_tmp1[i][0] for i in range(len(y_tmp1))]))
    mappings = {y_tmp2[i]: i for i in range(len(y_tmp2))}  # 维修类别描述--->维修类别ID [0,104]
    rev_mappings = {value: key for key, value in mappings.items()}  # 故障类别ID--->故障类别描述
    y_label = [mappings[y_tmp1[i][0]] for i in range(len(y_tmp1))]  # 获取训练数据中的标签值
    return y_label, rev_mappings


def get_acc(pre_label, true_label):
    count_acc = 0
    for k in range(len(pre_label)):
        if pre_label[k] == true_label[k]:
            count_acc += 1
    return count_acc / len(pre_label)


# 用正则表达式清洗数据
def clean_text(text):
    text = text.replace('\n', " ")  # 新行，不需要
    text = re.sub(r"-", " ", text)  # 把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text = re.sub(r"\d+/\d+/\d+", "", text)  # 日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text)  # 时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text)  # 邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)  # 网址，没意义

    # 以防还有其他除了单词以外的特殊字符（数字）等等，把特殊字符过滤掉
    # 只留下字母和空格
    # 再把单个字母去掉，留下单词
    pure_text = ''
    for letter in text:
        if letter.isalpha() or letter == ' ':
            pure_text += letter

    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text


def stopwords_list():
    stopwords = []
    for line1 in open(path + 'resources/stopwords.txt', 'r', encoding="utf-8").readlines():
        line1 = line1.strip()
        stopwords.append(line1)
    return stopwords


def get_vocab(text):
    text_depart = jieba.cut(text.strip())
    stopwords = stopwords_list()
    outstr = ''
    # 去停用词
    for word in text_depart:
        if word not in stopwords:
            outstr += word
            outstr += " "
    return outstr


def clean_text_vocab(text):
    text = clean_text(text)
    stopwords = stopwords_list()
    text = [word for word in text.lower().split() if word not in stopwords]
    vocab = get_vocab(text[0])
    return vocab.split()


def get_text_vector(text, model, vocab_list):
    vocab = clean_text_vocab(text)
    sentence_vecs = []
    for i in range(len(vocab)):
        if vocab[i] in vocab_list:
            sentence_vecs.append(model[vocab[i]])
    vec = np.mean(np.array(sentence_vecs), axis=0)
    return vec


def get_single_vec(text):
    model = gensim.models.KeyedVectors.load_word2vec_format(path + "resources/sgns.wiki.char.txt")
    vocab_list = [word for word in model.index_to_key]
    return get_text_vector(text, model, vocab_list)


def get_split_set(train_index, test_index, data, labels):
    train_data, train_label, test_data, test_label = [], [], [], []
    for i in train_index:
        train_data.append(data[i])
        train_label.append(labels[i])
    for j in test_index:
        test_data.append(data[j])
        test_label.append(labels[j])
    return train_data, train_label, test_data, test_label


def choose_best_model(data, labels):
    kf = KFold(n_splits=10, random_state=233, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        train_data, train_label, test_data, test_label = get_split_set(train_index, test_index, data, labels)
        model = GaussianNB()
        model.fit(np.array(train_data), np.array(train_label))
        pre_label = model.predict(test_data).tolist()
        true_label = test_label
        acc = get_acc(pre_label, true_label)
        print("第{}个模型：测试准确率：{:.2%}".format(i, acc))
        if i == 4:
            with open(path + 'gaussianNB.pickle', 'wb') as f_model:
                pickle.dump(model, f_model)
            print("模型保存完毕！")


def get_best_model():
    data = np.load(path + "data_vectors.npy").tolist()
    labels, rev_mappings = get_label()
    with open(path + 'mapping.json', 'w') as f_m:
        json.dump(rev_mappings, f_m)
    choose_best_model(data, labels)


if __name__ == "__main__":
    # 如果还没有训练数据，则从原始数据中获取训练数据
    get_train_data()
    model = gensim.models.KeyedVectors.load_word2vec_format(path + "resources/sgns.wiki.char.txt")
    vocab_list = [word for word in model.index_to_key]
    print("模型加载完毕！")
    t1 = time.time()
    train_data_matrix = []
    print("开始为训练数据构建词向量！")
    with open(path + "train_data.txt", "r", encoding="utf-8") as f:
        for count, line in enumerate(f.readlines()):
            if count % 100 == 0:
                print("正在构建第{}条数据".format(count))
            vec = get_text_vector(line, model, vocab_list)
            train_data_matrix.append(vec)
    print("词向量矩阵构建完毕，开始保存模型！")
    np.save(path + "data_vectors", np.array(train_data_matrix))
    print("保存成功！")
    t2 = time.time()

    print("用时：{}s".format(t2 - t1))
