# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:44:25 2018

@author: devkite
"""

import numpy as np 
import pandas as pd
import random
import gc
import time
from sklearn.metrics.pairwise import cosine_similarity
from PreprocessData import readData

def process_data(train, test, ratio=0.9):
    df = train.append(test)
    df.rename(colunms={0:'uid',1:'iid'})
    usernum = df.uid.max()+1 #这里多建一列，以防user,item id从1开始
    itemnum = df.iid.max()+1
    
#    测试集中元素以字典的形式存放，uid itemset
    test = {}
#    行保存user id ，列保存 item id
    train = np.zeros((usernum, itemnum),dtype = np.int32)
    
    user_degree = df.uid.value_counts()
    item_degree = df.iid.value_counts()
    train_udegree = {}
    train_idegree = {}
    
    traincount = 0
    testcount = 0
    for idx in train.index: 
        uid = train.loc[idx,'uid']
        iid = train.loc[idx,'iid']
        #        每次更新，相当于对原数据进行减少连边，对应度减少
        train[uid][iid] = 1
        traincount += 1

    for idx in test.index:
        uid = test.loc[idx, 'uid']
        iid = test.loc[idx, 'iid']
        if uid not in test:
            test[uid] = set()
        test[uid].add(iid)
        testcount += 1

    udegree = pd.Series(train_udegree)
    idegree = pd.Series(train_idegree)
#    释放多次引用的变量的内存
    del train_udegree, train_idegree
    gc.collect()    
    print("训练集%d",traincount)
    print("测试集%d",testcount)
    return train,test,udegree,idegree

    
def massDiffisionForOne(train, user, udegree, idegree, K=1000):
    usernum ,itemnum = np.shape(train)
    item_score = np.zeros((itemnum,1),dtype=np.float32)
    
    user_score = secondMass(train,idegree, user)
    #kNN,选择user端得分最高的K个user进行第三次Mass操作
    user_score = pd.Series(user_score)
    #  这里因为id是从1开始的所以跟index有细微bug，index 0 
    users = user_score.sort_values(ascending=False).index[0:K]
    #进行第三次Mass操作，为每个item进行打分
    for uid in users:
#            获得该user对有关item的打分，用字典表示
        u_itemscore=thirdMass(train, udegree, user_score, uid)
        for item in u_itemscore.keys():
            item_score[item] = u_itemscore[item] + item_score[item]


def secondMass(train, idegree, uid):
#    获得该user的item集合
    items = np.nonzero(train[uid, :])[0]
    user_score = np.zeros(train.shape[0],dtype=np.float32)
    for item in items:
#        找出买过这个item的所有user
        users=np.nonzero(train[:, item])[0]
#        每个user打分为,users/degree
        score = 1.0 / idegree[item]
        
        for user in users:
            user_score[user] = user_score[user] + score
    return user_score

def thirdMass(train, udegree, user_score, uid):    
    items = np.nonzero(train[uid, :])[0]
    u_itemscore = {}
    score = user_score[uid]/udegree[uid]
    for item in items:
        u_itemscore[item] = score
    return u_itemscore
    
def exp():
    '''
    编写实验逻辑
    '''
    filepath = r'./data/netflix5k_result.txt'
    train_data, test_data = readData(filepath, split=',', train_ratio=0.7)    
    process_data(train_data,test_data,0.9)

if __name__ == '__main__':
    exp()