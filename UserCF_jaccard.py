# coding=utf-8
'''
Created on 2019年3月2日

@author: devkite
'''


import utils
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import pickle
import os
from tqdm import tqdm
import threading
import csv



class UserCF:
    '''
    基于用户的协同过滤
    '''

    def __init__(self, train, test, item_len , k=100):
        self.train = train
        self.test = test
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
        u_itemset = self.train[u]
        v_itemset = self.train[v]

        common_itemlen = len(u_itemset & v_itemset)
        sim = common_itemlen / \
            (len(u_itemset) + len(v_itemset) - common_itemlen)
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

    def recommend(self, uid):
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
                raw_val = rank_item.get(item, 0)
                rank_item[item] = raw_val + sim_val

        return list(rank_item.items())

    def get_score(self):
        '''
        获得训练集所有用户的推荐得分
        '''
        recommend_score = {}
        print('计算推荐得分')
        for user in tqdm(self.train, ascii=True, ncols = 70, leave = False, unit = 'b'):
            recommend_score[user] = self.recommend(user)
        return recommend_score

    def cf_train(self):
        '''
        cf算法调用逻辑

        :Return: score :a dict 
                        key:userid
                        value: a tuple (itemid, recommend score)
        '''
        start = time.time()
        print('calculate similarity.')
        self.similarity()
        print('calculate similarity finished. cost {:.2f}s'.format(
            time.time()-start))

        start = time.time()
        score = self.get_score()
        print('calculate recommender score. cost {:.2f}s'.format(
            time.time()-start))
        return score
    
def get_item_score(user_degree, recommend_score, item_len, our_lambda=1.0):
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
    return item_score


# 舍弃了
def get_item_degree_distribute(cf):
    '''
    获取item的度分布信息，分析数据  
    '''
    item_degrees = cf.cal_item_degree()
    # item_degrees数据结构：key:itemid, value:item degree
    degreedist = {}
    for _, degree in item_degrees.items():
        if degree not in degreedist:
            degreedist[degree] = 0
        degreedist[degree] += 1
    plt.bar(degreedist.keys(), degreedist.values())
    plt.show()
    return degreedist


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


def accuracy(degreedistrev, test, item_score):
    '''
    评估算法性能
    '''

    # test集合数据结构：uid, iid, date(date不需要)
    # 统计测试集的item度信息
    itemdegree_map = {}
    for idx in range(0, test.shape[0]):
        iid = test.iloc[idx][1]
        if iid not in itemdegree_map:
            itemdegree_map[iid] = 0
        # 这里暂时不考虑重复行记录
        itemdegree_map[iid] += 1

    # 对于训练集中度相同的item，获得他们的未来度信息
    hit = 0
    total = 0
    for _, itemset in degreedistrev.items():

        for i in range(0, len(itemset)):
            for j in range(i, len(itemset)):
                # 获得item的编号,从1开始
                itemi, itemj = itemset[i], itemset[j]
                # item_score index 从0开始，item对应得分索引为itemid-1
                if itemi == itemj:
                    continue
                sign_predict = item_score.get(
                    itemi, 0) - item_score.get(itemj, 0)
                sign_label = itemdegree_map.get(
                    itemi, 0) - itemdegree_map.get(itemj, 0)
                # 如果两个符号相同，则说明判断正确， 为0的情况，判断是否都是0（或者跳过）
                if sign_label * sign_predict > 0 or sign_label==sign_predict:
                    hit += 1
                total += 1
    return 1.0 * hit / total

# 一个附加的方法，来获得最小N个度的item set
def getNdegree_items(degree_item_map, N=50):
    '''
    获得训练集中N个最低degree的item集合  
    返回 dict
    key为degree ， value为该degree的item set
    '''
    Ndegree_items = {}
    for i in range(1, N+1):
        itemset = degree_item_map.get(i, {})
        Ndegree_items[i] = itemset
    return Ndegree_items

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
                    train_itemset, 
                    test_itemset, 
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
            # 如果这个item不是同时在训练集和测试集里面出现的
            # 注释掉，因为会出现全为0得分的情况，这样子corr为nan
            # if (item not in train_itemset) or (item not in test_itemset):
            #     continue 

            score_map[item] = item_score.get(item, 0)
            test_degree_map[item] = test_item_degree.get(item, 0)

        rec_series[degree] = pd.Series(data=score_map)
        real_series[degree] = pd.Series(data=test_degree_map)
        corr_score[degree] = rec_series[degree].corr(real_series[degree], method=method)
    return corr_score
    
def stat_train_test_item(dataset):
    '''
    分别统计训练集中出现的item集合和
    '''
    itemset = set()
    for row in dataset.iterrows():
        itemset.add(row[1][1])
        return itemset




def main(our_lambda=1,data_file=r'./data/movielens_data.pkl',recommend_score_file=r'./temp/cf_score.pkl', recommend_score=None):
    train, test = pickle.load(open(data_file, 'rb+'))
    train_itemset = stat_train_test_item(train)
    test_itemset = stat_train_test_item(test)
    trainset, test, item_len = utils.deal_train(data_file)
    cf = UserCF(trainset, test, item_len)
    degreedistrev = degree_item_map(cf)
    # get_item_degree_distribute(cf)
    print('start cf train.')
    if os.path.exists(recommend_score_file):
        # 判断cf是否训练过
        with open(recommend_score_file,'rb') as f:
            recommend_score = pickle.load(f) 
    else:
        recommend_score = cf.cf_train()
        with open(recommend_score_file,'wb') as f:
            pickle.dump(recommend_score, f)

    # if recommend_score is None:
    #     recommend_score = cf.cf_train()

    user_degree = cf.cal_user_degree()
    item_score = get_item_score(user_degree, recommend_score, item_len, our_lambda=our_lambda)
    Ndegree_items = getNdegree_items(degreedistrev, N=10)
    test_item_degree = get_test_degree(test)
    print('start trend predict.')
    corr_score = trend_predict(item_score, Ndegree_items,test_item_degree, 
                                train_itemset, test_itemset, method='pearson')
    print(corr_score)
    return corr_score, recommend_score
    

if __name__ == "__main__":

    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump

    data_file = "./data/movielens.pkl"
    recommend_score_file = r'./temp/movielens_jaccard_1.pkl'
    resultfile = "./result/movielens/movielens_jaccardwithknn.txt"

    # data_file = "./data/delicious/delicious5k.pkl"
    # recommend_score_file=r'./temp/delicious_5k.pkl'
    # resultfile = "./result/delicious/delicious_5k.txt"
    # data_file = "./data/Amazon/amazon_5k.pkl"
    # recommend_score_file = r'./temp/amazon_5k.pkl'
    # resultfile = "./result/Amazon/amazon_5k.txt"
    corr_score_list = []
    recommend_score = None
    for p1 in frange(-1.0, 1.01, 0.1):
        corr_score,recommend_score = main( our_lambda=p1, data_file=data_file,\
            recommend_score_file=recommend_score_file, recommend_score=recommend_score)
        corr_score_list.append(corr_score)
        
    corr_score_arr = np.zeros((21, 10), np.float64)
    with open(resultfile, mode='w',encoding='utf-8') as f:
        i = 0
        for corr_score in corr_score_list:
            # corr_score表示一个lambda的得分字典，用列表示
            for degree,score in corr_score.items():
                corr_score_arr[i, degree-1] = score
            i += 1
            
        for i in range(10):
            for j in range(21):
                f.write(str(corr_score_arr[j,i]) + ',')
            f.write('\n')