#coding=utf-8
'''
Created on 2019年3月2日

@author: devkite
'''


import numpy as np

def cal_sim(vecA, vecB):
    '''
    计算A和B的余弦相似性
    '''
    dot_product = np.dot(vecA, vecB)
    denom = np.linalg.norm(vecA) * np.linalg.norm(vecB)
    
    return dot_product / denom 
    