import os

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)
# 上两级目录的路径
parent_path = os.path.dirname(os.path.dirname(script_path))
# 根目录的路径
root_path = os.path.abspath(parent_path)

import sys
sys.path.append(root_path)

import joblib
import numpy as np
import pandas as pd

from config import Config
from train_predict_model.word_embedding import TxWordVector
from train_predict_model.structure_embedding import TxStructureVector
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

def load_data(raw_tx, label):
    # Define embedding size
    STRUCTURE_EMBED_SIZE = 16
    SENTENCE_EMBED_SIZE = 36

    print("Load data ...")

    # Get hash label
    label_map = dict()
    df = label[['srcTxhash', 'label']]

    for _, row in df.iterrows():
        label_map[row['srcTxhash']] = row['label']

    # Get sturcture vectors
    structure_vecs = TxStructureVector(
        raw_tx=raw_tx
    ).count_motif()

    # Get sentences vectors
    sentence_vecs = TxWordVector(
        txlist=label,
        train=True
    ).wordembedd()

    # Create feature
    feature, label,txlist = [], [],[]
    for key, val in structure_vecs.items():

        if key[0] not in label_map:
            continue

        structure_feat = [val[i] if i in val else 0.0 for i in range(STRUCTURE_EMBED_SIZE)]
        sentence_feat = [i for arr in sentence_vecs[key[0]] for i in arr]
        sentence_feat = sentence_feat + [0.0 for _ in range(SENTENCE_EMBED_SIZE - len(sentence_feat))]
        feature.append(structure_feat + sentence_feat)
        label.append(label_map[key[0]])
        txlist.append(key[0])
        # print(len(feature), len(label))

    return np.array(feature), np.array(label),np.array(txlist)


def add_result(r, metric, classifier, value):
    if classifier in r[metric]:
        r[metric][classifier] += value
    else:
        r[metric][classifier] = value
    return r

def train_model(feature, label):
    print("train_model 开始训练...")
    # print(len(feature),"xxxxxx",len(label))
    classifiers = [
        RandomForestClassifier(n_jobs=10),
    ]
    POS_LABEL = 1
    r = {'acc': {}, 'pre': {}, 'rec': {}, 'f1': {}, 'auc': {}}

    splits = 1  # 这里你可以增加交叉验证
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.8, random_state=42)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # 记录训练过程中的数据
    train_sizes = []
    acc_scores = []
    f1_scores = []

    # 设置 Matplotlib
    plt.ion()  # 开启交互模式

    for clf in tqdm(classifiers, desc="训练进度"):  # 显示进度条
        name = clf.__class__.__name__
        print(f"\n训练模型: {name} ...")

        # 训练模型
        clf.fit(X_train, y_train)
        joblib.dump(clf, "model.pkl")  # 保存模型

        # 预测
        train_predictions = clf.predict(X_test)

        # 计算指标
        acc = accuracy_score(y_test, train_predictions)
        pre = precision_score(y_test, train_predictions, average='binary', pos_label=POS_LABEL)
        rec = recall_score(y_test, train_predictions, average='binary', pos_label=POS_LABEL)
        f1 = f1_score(y_test, train_predictions, average='binary', pos_label=POS_LABEL)
        auc = roc_auc_score(y_test, train_predictions)

        r = add_result(r, 'acc', name, acc)
        r = add_result(r, 'pre', name, pre)
        r = add_result(r, 'rec', name, rec)
        r = add_result(r, 'f1', name, f1)
        r = add_result(r, 'auc', name, auc)

        # 存储数据量和指标
        train_sizes.append(len(X_train))
        acc_scores.append(acc)
        f1_scores.append(f1)

        # 实时更新训练进度图
        plt.figure(1, figsize=(8, 4))
        plt.clf()  # 清空当前图像

        plt.subplot(1, 2, 1)
        plt.plot(train_sizes, acc_scores, label='Accuracy', marker='o', linestyle='-')
        plt.plot(train_sizes, f1_scores, label='F1 Score', marker='s', linestyle='--')
        plt.xlabel('Training Data Size')
        plt.ylabel('Score')
        plt.title('Training Progress')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance')

        plt.pause(0.1)  # 暂停 0.1 秒用于更新图像

    plt.ioff()  # 关闭交互模式
    plt.show()  # 最终显示图像

    for metric in r.keys():
        for clf in r[metric].keys():
            r[metric][clf] = r[metric][clf] / float(splits)

    print('\n方法 ', end=' ')
    [print("{:6}".format(clf), end=' ') for clf in r['pre'].keys()]
    for metric in r.keys():
        print('\n{:3}'.format(metric), end=' ')
        for clf in r[metric].keys():
            print("{0:.4f}".format(r[metric][clf]), end=' ')




