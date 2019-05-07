#coding=utf-8
'''
Created on 2019年3月1日

@author: 89534

数据集处理类，不通用

'''



import pandas as pd
import pickle

'''
delicous——sub2 文件格式
uid    iid   timestamp

本脚本处理出训练和测试集合
'''
# 处理数据暂时不做下采样


def readData(filepath, split=',', train_ratio=0.8, sample_num=100000):
    data = pd.read_csv(filepath, sep=split, header=None,encoding='utf-8')
    data = data.sort_values(by=2, axis=0, ascending=True)[0: sample_num]
    data_len = data.shape[0]
    train_len = (int)(data_len * train_ratio)
    train_data = data.iloc[:train_len,:]
    test_data = data.iloc[train_len:,:]
    return train_data, test_data
    
# if __name__ == '__main__':
#     filepath = r'./data/delicious/delicious_subset2.txt'
#     train_data, test_data = readData(filepath, split='\t', sample_num=50000)    
#     pickle.dump((train_data, test_data), 
#                 open(r'./data/delicious/data.pkl', 'wb+'))
#     print("dump finished.")
    