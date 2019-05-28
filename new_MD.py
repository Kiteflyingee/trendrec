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
from tqdm import tqdm

import pickle
import os


def process_data(trainset, testset):
    '''
    把切分的数据集重新整理数据结构
    '''
    df = trainset.append(testset)
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








    
def massDiffisionForOne(train, user, udegree, idegree, _lambda=1, K=1000):
    '''
    对单用户做MassDiffusion过程
    '''
    
    itemnum = np.shape(train)[1]
    # 这里的usernum和itemnum都是在原始上加1(在id从1开始的数据集，id从0开始的就没有这种情况)
    item_score = np.zeros(itemnum,dtype=np.float32)
    # 处理特殊情况
    if udegree.get(user,0.0) == 0.0:
        return item_score
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
    # 这个user对最终item score得分贡献*该user的度的lambda次方
    # item_score = item_score * pow(udegree.get(user), _lambda)
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
#        每个user打分为,users/degree,这个item不可能不在训练集中
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
    if udegree.get(uid, 0.0) != 0.0:
        score = user_score[uid]/udegree[uid]
    else:
        # 如果uid不在训练集中
        score = 0.0
    for item in items:
        u_itemscore[item] = score
    return u_itemscore


# 一个附加的方法，来获得最小N个度的item set
def getNdegree_items(degree_item_map, N=100):
    '''
    获得训练集中N个最低degree的item集合  
    返回 dict
    key为degree ， value为该degree的item set
    '''
    Ndegree_items = {}
    for i in range(1, N+1):
        itemset = degree_item_map[i]
        Ndegree_items[i] = itemset
    return Ndegree_items




def trend_predict(item_score, 
                    Ndegree_items, 
                    test_item_degree, 
                    method='pearson'):
    '''
    这是负责流行度预测，对对应Ndegree的item set生成训练集的流行度推荐列表
    并整理test set里真实的流行度排序

    :param 
        item_score:  numpy数组
        Ndegree_items： dict类型， key=degree value=item set
        test_item_degree: series类型操作与dict基本相同
        method: corr计算指标
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
                callable: callable with input two 1d ndarray
                            and returning a float
    :return 
        平均相关性得分
    '''
    rec_series = {}
    real_series = {}
    corr_score = {}
    print('trend predict.')
    for degree in (Ndegree_items):
        # 遍历每个degree的itemset
        score_map = {}
        test_degree_map = {}
        if Ndegree_items[degree] == {}:
            continue
        for item in Ndegree_items[degree]:
            score_map[item] = item_score[item]
            test_degree_map[item] = test_item_degree.get(item, 0)

        rec_series[degree] = pd.Series(data=score_map)
        real_series[degree] = pd.Series(data=test_degree_map)
        corr_score[degree] = rec_series[degree].corr(real_series[degree], method=method)
    return corr_score





def degree_item_map(item_degrees):
    '''
    建立degree-item倒排表,用于统计训练集中item的degree信息
    '''
    degreedistrev = {}
    for iid, degree in item_degrees.items():
        if degree not in degreedistrev:
            degreedistrev[degree] = []
        degreedistrev[degree].append(iid)
    return degreedistrev




def exp(mylambda):
    '''
    编写实验逻辑
    '''
    score_filepath = 'amazon_score.pkl'
    filepath = r'./data/Amazon/Amazon_2.txt'
    train_data, test_data = readData(filepath, split=',', train_ratio=0.7)    
    train_data = train_data.rename(columns={0:'uid',1:'iid'})
    test_data = test_data.rename(columns={0:'uid',1:'iid'})
    train, _, udegree, idegree = process_data(train_data, test_data)
    # userid从1开始的情况
    total_item_score = np.zeros(train.shape[1], dtype=np.float64)
    if os.path.exists(score_filepath):
        item_scores = pickle.load(open(score_filepath,'rb'))
        for user in tqdm(range(1,train.shape[0]),ascii=True):
            if udegree.get(user, 0.0) == 0.0:
                continue
            one_item_score = item_scores[user]
            total_item_score += one_item_score * pow(udegree.get(user), mylambda)
    else:
        item_scores = {}
        for user in tqdm(range(1,train.shape[0]),ascii=True):
        # userid从0开始的情况
        # for user in range(train.shape[0]):
            if udegree.get(user, 0.0) == 0.0:
                continue
            one_item_score = massDiffisionForOne(train, user, udegree, idegree, mylambda, K=1000)
            total_item_score += one_item_score * pow(udegree.get(user), mylambda)
            item_scores[user] = one_item_score
        pickle.dump(item_scores, open(score_filepath,'wb'))

    # 获得度-itemset 分布信息
    degreedistrev = degree_item_map(idegree)
    Ndegree_items = getNdegree_items(degreedistrev, N=20)
    # 获得testset item度分布
    test_item_degree = test_data.iid.value_counts()
    corr_score = trend_predict(total_item_score, 
                                Ndegree_items,
                                test_item_degree, 
                                method='pearson')
    print(corr_score)
    return corr_score



if __name__ == '__main__':
    '''
    入口
    '''
    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump

    lambdas = list(frange(-1.0,1.01,0.1))
    scores = []
    with open('md_Amazon_result.csv','w',encoding="utf-8") as f:
        for mylambda in lambdas:
            corr_score = exp(mylambda)
            scores.append(corr_score)
            f.write(str(mylambda) + ',')
            result = ''
            for _,score in corr_score.items():
                result += str(score) + ','
            f.write(result[:-1]+'\n')
            f.flush()
    
    
