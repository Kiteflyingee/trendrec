# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 23:44:30 2019

@author: 89534
"""

import pandas as pd
import pickle

'''
delicous——sub2 文件格式
uid    iid   timestamp

本脚本处理出训练和测试集合
'''
# 处理数据暂时不做下采样


def readData(filepath, split=',', train_ratio=0.9):
    data = pd.read_csv(filepath, sep=split, header=None,encoding='utf-8')
    data = data.sort_values(by=2, axis=0, ascending=True)
    data_len = 50000
    train_len = (int)(data_len * train_ratio)
    train_data = data.iloc[:train_len,:]
    test_data = data.iloc[train_len:data_len,:]
    return train_data, test_data
    
if __name__ == '__main__':
    filepath = r'./data/delicious/delicious_subset2.txt';
    train_data, test_data = readData(filepath, split='\t')    
    pickle.dump((train_data, test_data), 
                open(r'./data/delicious/data_test.pkl', 'wb+'))
    
    