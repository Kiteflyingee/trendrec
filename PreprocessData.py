#coding=utf-8
'''
Created on 2019年3月1日

@author: 89534

数据集处理类，不通用

'''


import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import threading 
import multiprocessing

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
    train_data = train_data.drop(removeidx)
    removeidx=set()
    for idx in temp_testdf.index:
        uid = temp_testdf.loc[idx, 'uid']
        iid = temp_testdf.loc[idx, 'iid']
        if (uid not in uid_set) or (iid not in iid_set):
            removeidx.add(idx)
            count+=1
    test_data = test_data.drop(removeidx)
    print("removenum:" ,count)
    return train_data, test_data
    
    

def rerank(trainset, testset):
    df = pd.concat([trainset,testset], ignore_index=True)  
    df.columns = ['uid','iid','time']
    u_unique = df.uid.unique()
    i_unique = df.iid.unique()
    u_id = {}
    i_id = {}

    for idx, uid in enumerate(u_unique):
        u_id[uid] = idx

    for idx, iid in enumerate(i_unique):
        i_id[iid] = idx
    
    new_trainset = trainset.copy()
    for rowidx in tqdm(range(len(trainset)),ascii=True):
        uid = trainset.iloc[rowidx, 0]
        iid = trainset.iloc[rowidx, 1]
        new_trainset.iloc[rowidx, 0] = u_id[uid]
        new_trainset.iloc[rowidx, 1] = i_id[iid]

    new_testset = testset.copy()
    for rowidx in range(len(testset)):
        uid = testset.iloc[rowidx, 0]
        iid = testset.iloc[rowidx, 1]
        new_testset.iloc[rowidx, 0] = u_id[uid]
        new_testset.iloc[rowidx, 1] = i_id[iid]
    
    return new_trainset, new_testset


def rerank_multithread(trainset, testset):
    df = pd.concat([trainset,testset], ignore_index=True)  
    df.columns = ['uid','iid','time']
    u_unique = df.uid.unique()
    i_unique = df.iid.unique()
    u_id = {}
    i_id = {}

    for idx, uid in enumerate(u_unique):
        u_id[uid] = idx

    for idx, iid in enumerate(i_unique):
        i_id[iid] = idx
    
    t1 = MyThread(trainset, u_id, i_id, 'train')
    t2 = MyThread(testset, u_id, i_id, 'test')
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    new_trainset = t1.result
    new_testset = t2.result
    return new_trainset, new_testset


def rerank_multiprocess(trainset, testset):
    df = pd.concat([trainset,testset], ignore_index=True)  
    df.columns = ['uid','iid','time']
    u_unique = df.uid.unique()
    i_unique = df.iid.unique()
    u_id = {}
    i_id = {}

    for idx, uid in enumerate(u_unique):
        u_id[uid] = idx

    for idx, iid in enumerate(i_unique):
        i_id[iid] = idx
    
    t1 = MyProcess(trainset, u_id, i_id, 'train')
    t2 = MyProcess(testset, u_id, i_id, 'test')
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    new_trainset = t1.result
    new_testset = t2.result
    return new_trainset, new_testset

class MyProcess(multiprocessing.Process):
    def __init__(self, df, u_id, i_id, name):
        multiprocessing.Process.__init__(self)
        self.df = df
        self.u_id = u_id
        self.i_id = i_id
        self.name = name

    def run(self):
        print ("开启进程：" + self.name + '\n')
        self.result = processnewdf(self.df, self.u_id, self.i_id)
        print ("退出进程：" + self.name + '\n')


class MyThread(threading.Thread):
    def __init__(self, df, u_id, i_id, name):
        threading.Thread.__init__(self)
        self.df = df
        self.u_id = u_id
        self.i_id = i_id
        self.name = name

    def run(self):
        print ("开启线程：" + self.name + '\n')
        self.result = processnewdf(self.df, self.u_id, self.i_id)
        print ("退出线程：" + self.name + '\n')

def processnewdf(df, u_id, i_id):
    for rowidx in tqdm(range(len(df)), ascii=True):
        uid = df.iloc[rowidx, 0]
        iid = df.iloc[rowidx, 1]
        df.iloc[rowidx, 0] = u_id[uid]
        df.iloc[rowidx, 1] = i_id[iid]
    return df