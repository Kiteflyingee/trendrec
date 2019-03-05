#coding=utf-8
'''
Created on 2019年3月2日

@author: devkite
'''


from utils import *

class UserCF:
    '''
    基于用户的协同过滤
    '''
    def __init__(self, train, test, item_len):
        self.train = train
        self.test = test
        self.item_len = item_len
    
    def cal_degree(self):    
        '''
            统计训练集中用户的degree信息
        '''
        self.user_degree = {}
        
        for user in train:
            degree = len(train.get(user, set()))
            if user in self.user_degree:
                self.user_degree[user] = self.user_degree[user] + 1
            else:
                self.user_degree[user] = 0  
                                                
                                                
    def cal_sim(self):
        '''
        统计的训练集的用户的相似度信息
        在训练集中的用户两两求相似性
        '''
        self.user_sim = {}
        
        for u in self.train:
            for v in self.train:
#                 如果这两个用户计算过相似性
                if u in self.user_sim and  \
                        v in self.user_sim[u]:
                       continue
                else:
                    sim = cal_sim(u, v)
                    user_sim[u][v] = sim
                    user_sim[v][u] = sim
                    
                       