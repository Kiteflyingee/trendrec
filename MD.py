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






def process_data(trainset, testset):
    '''
    把切分的数据集重新整理数据结构
    '''
    
    df = trainset.append(testset)
    df.rename(colunms={0:'uid',1:'iid'})
    usernum = df.uid.max()+1 #这里多建一列，以防user,item id从1开始
    itemnum = df.iid.max()+1
    
#    测试集中元素以字典的形式存放，uid itemset
    test = {}
#    行保存user id ，列保存 item id
    train = np.zeros((usernum, itemnum),dtype = np.int32)
    # 得到是pandas Series 类型
    user_degree = trainset.uid.value_counts()
    item_degree = trainset.iid.value_counts()
    
    traincount = 0
    testcount = 0
    for idx in trainset.index: 
        uid = trainset.loc[idx,'uid']
        iid = trainset.loc[idx,'iid']
        #  填充数据矩阵
        train[uid][iid] = 1
        traincount += 1

    for idx in testset.index:
        uid = testset.loc[idx, 'uid']
        iid = testset.loc[idx, 'iid']
        if uid not in test:
            test[uid] = set()
        test[uid].add(iid)
        testcount += 1

    #    释放多次引用的变量的内存
    print("训练集%d",traincount)
    print("测试集%d",testcount)
    return train,test,user_degree,item_degree








    
def massDiffisionForOne(train, user, udegree, idegree, K=1000):
    '''
    对单用户做MassDiffusion过程
    '''
    itemnum = np.shape(train)[1]
    # 这里的usernum和itemnum都是在原始上加1(在id从1开始的数据集，id从0开始的就没有这种情况)
    item_score = np.zeros((itemnum,1),dtype=np.float32)
    user_score = secondMass(train, idegree, user)
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
    return item_score










def secondMass(train, idegree, uid):
    '''
    MassDiffusion第二次传播过程
    '''
#    获得该user的item集合
    items = np.nonzero(train[uid, :])[0]
    user_score = np.zeros(train.shape[0],dtype=np.float32)
    for item in items:
#        找出买过这个item的所有user，id从1开始也不影响
        users=np.nonzero(train[:, item])[0]
#        每个user打分为,users/degree
        score = 1.0 / idegree[item]
        for user in users:
            user_score[user] = user_score[user] + score
    return user_score









def thirdMass(train, udegree, user_score, uid):   
    '''
    massdiffusion 第三次传播过程
    ''' 
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
    train,test,udegree,idegree = process_data(train_data,test_data,0.9)
    # for user in train.keys()

if __name__ == '__main__':
    '''
    入口
    '''
    exp()