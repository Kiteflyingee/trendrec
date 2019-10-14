#coding=utf-8


import numpy as np
import pandas as pd


'''
在得到推荐得分矩阵的情况下，进行trend预测并测试
'''

def findLastN(N, degree_item_map):
    '''
    找出训练集中最冷门的N的item
    获得训练集中N个最低degree的item集合  
    返回 dict
    key为degree ， value为该degree的item set
    '''
    Ndegree_items = {}
    for i in range(1, N+1):
        itemset = degree_item_map[i]
        Ndegree_items[i] = itemset
    return Ndegree_items




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




def readData(filepath, split=',', train_ratio=0.7):
    '''
    直接读所有数据集，包装成pandas，返还训练集和测试集
    '''
    data = pd.read_csv(filepath, sep=split, header=None,encoding='utf-8')
    # 有的数据超过三列，只取前3列
    data = data.sort_values(by=2, axis=0, ascending=True).iloc[:,:3]
    data_len = data.shape[0]
    train_len = (int)(data_len * train_ratio)
    train_data = data.iloc[:train_len,:]
    test_data = data.iloc[train_len:,:]
    return train_data, test_data



# def readRecScore():