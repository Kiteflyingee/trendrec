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
    def __init__(self, train, test, item_len, topn=100, k=10):
        self.train = train
        self.test = test
        self.item_len = item_len
        self.topn = topn
        self.k = k
    
    def cal_user_degree(self):    
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
        return self.user_degree
    
    def cal_item_degree(self):
        self.item_degree = {}

        for u,items in self.train.items():
            for item in items:
                self.item_degree.setdefault(item, 0)
                self.item_degree[item] = self.item_degree[item] + 1
        
        return self.item_degree
                                                
                                                
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
        '''
        获得用户uid的推荐得分列表
        '''
        watch_item = self.train.get(uid, set())
        vector_rank = np.zeros((self.item_len, 1), dtype=np.float32)

        # 先对用户的相似性列表排序
        sim_list = self.user_sim.get(uid, {})
        sorted_sim = sorted(sim_list.items(), key=lambda x:x[1], reverse=True)[:self.k]
        del sim_list

        for v,sim_val in sorted_sim:
            for item in self.train.get(v, {}):
                if item in watch_item:
                    continue
                vector_rank[item-1] += sim_val

        return np.sort(vector_rank)[:self.topn]


    def get_score(self):
        '''
        获得训练集所有用户的推荐得分
        '''
        recommend_score = {}
        for user in self.train:
            recommend_score[user] = self.recommend(user)
        return recommend_score

    def cf_train(self):
        '''
        cf算法调用逻辑
        '''
        self.cal_sim()
        return self.get_score()

        
def get_item_score(user_degree, recommend_score, item_len):
    '''
    获得训练集的所有item的得分
    '''
    item_socre = np.zeros((item_len, 1), np.float32)

    for user in recommend_score:
        item_socre += recommend_score[user] *user_degree[user]
    return item_socre


def degree_predict(train, test, item_degree, item_socre):
    '''
    预测训练集中degree相同的item，预测它们未来的degree高低
    '''


def auc(train, test, n=50000):
    '''
    计算auc
    
    '''

def accuracy(train, test):
    '''
    '''

if __name__ == "__main__":
    trainset, test, item_len = deal_train() 
    cf = UserCF(trainset, test, item_len)
    recommend_score = cf.cf_train()
    user_degree = cf.cal_user_degree()
    get_item_score(user_degree, recommend_score, item_len)
