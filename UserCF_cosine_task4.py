# coding=utf-8
'''
Created on 2019年3月2日

@author: devkite
'''


import utils
import numpy as np
import pandas as pd
import time
import pickle
from tqdm import tqdm
import math




'''
用lambda=-1得分向量div lambda=1得分向量
'''

class UserCF:
    '''
    基于用户的协同过滤
    '''

    def __init__(self, train, item_len, k=100):
        self.train = train
        self.item_len = item_len
        self.k = k

    def cal_user_degree(self):
        '''
            统计训练集中用户的degree信息
        '''
        self.user_degree = {}

        for user in self.train:
            degree = len(self.train.get(user, set()))
            self.user_degree[user] = degree
        return self.user_degree

    def cal_item_degree(self):
        self.item_degree = {}

        for _, items in self.train.items():
            for item in items:
                self.item_degree.setdefault(item, 0)
                self.item_degree[item] = self.item_degree[item] + 1

        return self.item_degree


    def get_similarity(self, u, v):
        '''
        这里使用余弦相似度
        '''
        u_itemset = self.train[u]
        v_itemset = self.train[v]

        common_itemlen = len(u_itemset & v_itemset)
        sim = common_itemlen / \
            math.sqrt(len(u_itemset) * len(v_itemset) * 1.0)
        return sim



    def similarity(self):
        '''
        统计的训练集的用户的相似度信息
        在训练集中的用户两两求相似性
        使用Jaccard指标
        '''
        self.user_sim = {}
        try:
            with tqdm(self.train, ascii=True, ncols = 70, leave = False, unit = 'b') as T:
                for u in T:
                    for v in self.train:
                        #                 如果这两个用户计算过相似性
                        if u == v:
                            continue
                        if (u in self.user_sim and v in self.user_sim[u]) or \
                                (v in self.user_sim and u in self.user_sim[v]):
                            continue
                        else:
                            sim = self.get_similarity(u, v)

                            self.user_sim.setdefault(u, {})
                            self.user_sim.setdefault(v, {})

                            self.user_sim[u][v] = sim
                            self.user_sim[v][u] = sim
        except KeyboardInterrupt:
            T.close()
            raise
        return self.user_sim

    def recommend(self, uid, Nitemset):
        '''
        获得用户uid的推荐得分列表
        return:
            # vector_rank:向量，顺序存储对所有item的预测打分 np向量(item_len,)
            rank_item:字典，key:itemid value:item score
            sorted_item: (itemid, score) topn
        '''
        watch_item = self.train.get(uid, set())
        rank_item = {}

        # 先对用户的相似性列表排序
        sim_list = self.user_sim.get(uid, {})
        sorted_sim = sorted(
            sim_list.items(), key=lambda x: x[1], reverse=True)[:self.k]

        for v, sim_val in sorted_sim:
            for item in self.train.get(v, {}):
                if item in watch_item:
                    continue
                if item not in Nitemset:
                    continue
                raw_val = rank_item.get(item, 0)
                rank_item[item] = raw_val + sim_val
        return list(rank_item.items())

    def get_score(self, Nitemset):
        '''
        获得训练集所有用户的推荐得分
        '''
        recommend_score = {}
        print('计算推荐得分')
        for user in tqdm(self.train, ascii=True, ncols = 70, leave = False, unit = 'b'):
            recommend_score[user] = self.recommend(user, Nitemset)
        return recommend_score

def get_item_score(user_degree, recommend_score, item_len, our_lambda):
    '''
    获得训练集的所有item的得分
    Return:
        item_score: dict  key itemid
                            value itemscore
                            不是所有item都一定有得分的，只有在推荐列表的时候会加分，所有后面使用get(id, 0)默认值方式
    '''
    # item_socre = np.zeros(item_len, np.float32)
    item_score = {}
    for user in recommend_score:
        # 这里使用矩阵运算, 乘上user的degree作为权重
        if user_degree.get(user, 0) < 1e-100:
            continue
        recommend_score_single = recommend_score[user]
        weight = pow(user_degree[user], our_lambda)
        for element in recommend_score_single:
            itemid, score = element
            pre_score = item_score.get(itemid, 0)
            item_score[itemid] = pre_score + score * weight
    return pd.Series(item_score)


def degree_item_map(cf):
    '''
    建立degree-item倒排表,用于统计训练集中item的degree信息
    '''
    item_degrees = cf.cal_item_degree()
    degreedistrev = {}
    for iid, degree in item_degrees.items():
        if degree not in degreedistrev:
            degreedistrev[degree] = []
        degreedistrev[degree].append(iid)
    return degreedistrev

# 一个附加的方法，来获得最小N个度的item set
def getNdegree_items(degree_item_map, N=10):
    '''
    获得训练集中N个最低degree的item集合  
    返回 dict
    key为degree ， value为该degree的item set
    '''
    Ndegree_items = {}
    Nitemset = set()
    for i in range(1, N+1):
        itemset = degree_item_map[i]
        Ndegree_items[i] = itemset
        Nitemset.update(itemset)
    return Ndegree_items, Nitemset

# 获得测试集中的item-degree map
def get_test_degree(testset):

    test_item_degree = {}
    for idx in range(0, testset.shape[0]):
        itemid = testset.iloc[idx][1]
        degree = test_item_degree.get(itemid,0)
        test_item_degree[itemid] = degree + 1
    return test_item_degree

def trend_predict(item_score, 
                    Ndegree_items, 
                    test_item_degree, 
                    method='pearson'):
    '''
    这是负责流行度预测，对对应Ndegree的item set生成训练集的流行度推荐列表
    并整理test set里真实的流行度排序

    :param 
        item_score:  dict类型，推荐算法给出的item的推荐得分
        Ndegree_items： dict类型， key=degree value=item set
        test_item_degree: dict类型,test set 中item的度map
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
            score_map[item] = item_score.get(item, 0)
            test_degree_map[item] = test_item_degree.get(item, 0)

        rec_series[degree] = pd.Series(data=score_map)
        real_series[degree] = pd.Series(data=test_degree_map)
        corr_score[degree] = rec_series[degree].corr(real_series[degree], method=method)
    return corr_score
    



'''
task4 新增代码块
'''
def our_newrc(user_degree, recommend_score, item_len):
    sub1item_score = get_item_score(user_degree, recommend_score, item_len, our_lambda=-1)
    plus1item_score = get_item_score(user_degree, recommend_score, item_len, our_lambda=1)
    new_item_score = sub1item_score.div(plus1item_score)
    return new_item_score




'''
用减法
'''
def our_newrc_sub(user_degree, recommend_score, item_len):
    sub1item_score = get_item_score(user_degree, recommend_score, item_len, our_lambda=-1)
    plus1item_score = get_item_score(user_degree, recommend_score, item_len, our_lambda=1)
    new_item_score = sub1item_score.sub(plus1item_score)
    return new_item_score



def main(user_sim = None ,data_file=r'./data/movielens_data.pkl'):
    train, test = pickle.load(open(data_file, 'rb+'))
    trainset, test, item_len = utils.deal_train(data_file)
    cf = UserCF(trainset, item_len)
    degreedistrev = degree_item_map(cf)
    Ndegree_items, Nitemset = getNdegree_items(degreedistrev, N=10)
    start = time.time()
    print('calculate similarity.')
    if user_sim is None:
        user_sim = cf.similarity()
    else:
        cf.user_sim = user_sim
    print('calculate similarity finished. cost {:.2f}s'.format(
        time.time()-start))

    start = time.time()
    # 这里获得原始的推荐得分
    recommend_score = cf.get_score(Nitemset)
    print('calculate recommender score. cost {:.2f}s'.format(
        time.time()-start))
    user_degree = cf.cal_user_degree()
    # 获得两个lambda的推荐的分的差值
    # item_score = our_newrc(user_degree, recommend_score, item_len)
    # 后面用减法
    item_score = our_newrc_sub(user_degree, recommend_score, item_len)
    test_item_degree = get_test_degree(test)
    print('start trend predict.')
    corr_score = trend_predict(item_score, Ndegree_items, 
                                test_item_degree,method='pearson')
    print(corr_score)
    return corr_score, recommend_score, user_sim
    

if __name__ == "__main__":

    data_file = "./data/movielens.pkl"
    resultfile = "./result/movielens/ml_k_task4_sub.txt"

    # data_file = "./data/delicious/delicious5k_rerank.pkl"
    # recommend_score_file=r'./temp/delicious_5k.pkl'
    # resultfile = "./result/delicious/delicious_cfk_task4_sub.txt"
    # data_file = "./data/Amazon/amazon_5k.pkl"
    # recommend_score_file = r'./temp/amazon_5k.pkl'
    # resultfile = "./result/Amazon/amazon_cfk_task4.txt"

    # data_file = "./data/netflix.pkl"
    # # recommend_score_file = r'./temp/netflixcfnoknn.pkl'
    # resultfile = "./result/netflix/netflix_k_cf_task4_sub.txt"

    # data_file = './data/Amazon_iterrows.pkl'
    # recommend_score_file = r'./temp/amazon_iterrows_k.pkl'
    # resultfile = "./result/Amazon/amazon_iterrows_k.txt"

    # data_file = './data/Amazon/amazon5k.pkl'
    # # recommend_score_file = r'./temp/amazon5k1.pkl'
    # resultfile = "./result/Amazon/amazon5k_task4_sub.txt"


    corr_score_list = []
    user_sim = None
    corr_score,recommend_score,user_sim= main(user_sim=user_sim,data_file=data_file)
    # corr_score_list.append(corr_score)
    print("dataset:",data_file)
    print(corr_score.values())
    