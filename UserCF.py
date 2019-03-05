#coding=utf-8
'''
Created on 2019年3月2日

@author: devkite
'''


from utils import *
import numpy as np

class UserCF:
    '''
    基于用户的协同过滤
    '''
    def __init__(self, train, test, item_len, topn=10, k=10):
        self.train = train
        self.test = test
        self.item_len = item_len
        self.topn = topn
        self.k = k
    
    def cal_degree(self):    
        '''
            统计训练集中用户的degree信息
        '''
        self.user_degree = {}
        
        for user in self.train:
            degree = len(self.train.get(user, set()))
            if user in self.user_degree:
                self.user_degree[user] = degree
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
                    self.user_sim.setdefault(u, {})
                    self.user_sim.setdefault(v, {})
                    self.user_sim[u][v] = sim
                    self.user_sim[v][u] = sim

        return self.user_sim

    
    def recommend(self, uid):
        watch_item = self.train.get(uid, set())
        vector_rank = np.zeros(self.item_len)

        # 先对用户的相似性列表排序
        sim_list = self.user_sim.get(uid, {})
        sorted_sim = sorted(sim_list.items(), key=lambda x:x[1], reverse=False)
        del sim_list

        for v,sim_val in sorted_sim:
            for item in self.train.get(v, {}):
                if item in watch_item:
                    continue
                vector_rank[item-1] += sim_val

        return np.sort(vector_rank)
