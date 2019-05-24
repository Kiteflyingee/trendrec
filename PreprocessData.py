#coding=utf-8
'''
Created on 2019年3月1日

@author: 89534

数据集处理类，不通用

'''


import numpy as np
import pandas as pd
import pickle

'''
delicous——sub2 文件格式
uid    iid   timestamp

本脚本处理出训练和测试集合
'''
# 处理数据暂时不做下采样


# def readData(filepath, split=',', train_ratio=0.8, sample_num=100000):
def readData(filepath, split=',', train_ratio=0.8):
    data = pd.read_csv(filepath, sep=split, header=None,encoding='utf-8')
    # data = data.sort_values(by=2, axis=0, ascending=True)[0: sample_num]
    # 有的数据超过三列，只取前3列
    data = data.sort_values(by=2, axis=0, ascending=True).iloc[:,:3]
    data_len = data.shape[0]
    train_len = (int)(data_len * train_ratio)
    train_data = data.iloc[:train_len,:]
    test_data = data.iloc[train_len:,:]
    # 去除那些仅在训练集或者测试集的数据
    temp_traindf = train_data.copy()
    temp_testdf = test_data.copy()
    temp_traindf.columns=['uid','iid','time']
    temp_testdf.columns=['uid','iid','time']
    uid_set = set(temp_traindf.uid.unique()) & set(temp_testdf.uid.unique())
    iid_set = set(temp_traindf.iid.unique()) & set(temp_testdf.iid.unique())
    removeidx=set()
    count=0
    for idx in temp_traindf.index:
        uid = temp_traindf.loc[idx, 'uid']
        iid = temp_traindf.loc[idx, 'iid']
        if (uid not in uid_set) or (iid not in iid_set):
            removeidx.add(idx)
            count+=1
    train_data.drop(removeidx)
    print("removenum:" ,count)
    removeidx=set()
    for idx in temp_testdf.index:
        uid = temp_testdf.loc[idx, 'uid']
        iid = temp_testdf.loc[idx, 'iid']
        if (uid not in uid_set) or (iid not in iid_set):
            removeidx.add(idx)
    test_data.drop(removeidx)
    return train_data, test_data
    
    