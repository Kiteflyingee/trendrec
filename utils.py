#coding=utf-8
'''
Created on 2019年3月2日

@author: devkite
'''


import numpy as np
import pickle
import pandas as pd

def cal_sim(vecA, vecB):
    '''
    计算A和B的余弦相似性
    '''
    dot_product = np.dot(vecA, vecB)
    denom = np.linalg.norm(vecA) * np.linalg.norm(vecB)
    
    return dot_product / denom 

def deal_buy(itemset, itemlen):
    '''
    把用户在训练集中item的set集合转换为numpy向量
    '''    
#    这里长度+5000为处理冷启动，只是用于计算相似度，所以增加长度不影响
    vec = np.zeros(itemlen+5000, np.float32)
    for iid in itemset:
        # 其实这里不需要iid-1，iid就行而且泛化性更强，可以包含id从0开始的情况
        vec[iid-1] = 1
    return vec
    
def deal_train(file='./data/delicious/data.pkl'):
    '''
    把训练集改为map格式(字典形式),key为用户id，value为item id
    '''
    train, test = pickle.load(open(file, 'rb+'))
    
    train_set = {}
    item_set = set()
    for idx in train.index:
        uid = train.loc[idx, 0]
        iid = train.loc[idx, 1]
#         date = train.loc[idx, 2]    暂时不考虑时间
        if uid not in train_set:
            train_set[uid] = set()
        train_set[uid].add(iid)
        item_set.add(iid)
    item_len = len(item_set)    
    return train_set, test, item_len