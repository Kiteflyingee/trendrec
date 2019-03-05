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
    
    
def deal_train(file='./data/delicious/data.pkl'):
    train, test = pickle.load(open(file, 'rb+'))
    
    train_set = {}
    item_set = set()
    for idx in train.index:
        uid = train.loc[idx, 0]
        iid = train.loc[idx, 1]
#         date = train.loc[idx, 2]    暂时不考虑时间
        if uid in train_set:
            train_set[uid].add(iid)
        else:
            train_set[uid] = set()
        item_set.add(iid)
    item_len = len(item_set)    
    return train_set, test, item_len