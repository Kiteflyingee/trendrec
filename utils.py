#coding=utf-8
'''
Created on 2019年3月2日

@author: devkite
'''


import numpy as np
import pickle
from _io import open

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
        uid = train[idx, 0]
        iid = train[idx, 1]
#         date = train[idx, 2]    暂时不考虑时间
        if uid in train_set:
            train_set[uid].add(iid)
        else:
            train_set[uid] = set()
        item_set.add(iid)
    item_len = item_set.size()    
    return train_set, test, item_len


if __name__ == "__main__":
    deal_train()