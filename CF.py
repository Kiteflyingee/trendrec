# coding=utf-8
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import utils

'''
最原始的CF，没有knn的约束，相似性指标使用jaccard
'''


class UserCF:
    '''
    基于用户的协同过滤
    '''

    def __init__(self, train, item_len):
        self.train = train
        self.item_len = item_len
        # 记录每个item的最终得分
        # self.item_score = {}
        self.user_sim = None

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
        '''
        print('calculate similarity..')
        self.user_sim = {}
        try:
            with tqdm(self.train, ascii=True) as T:
                for u in T:
                    # u_items = pd.Series(np.ones(len(self.train[u])),index=self.train[u])
                    for v in self.train:
                        # 如果这两个用户计算过相似性
                        if u == v:
                            continue
                        if (u in self.user_sim and v in self.user_sim[u]) or \
                                (v in self.user_sim and u in self.user_sim[v]):
                            continue
                        else:
                            # 点乘计算共同的数量
                            # 并集的数量=A的数量+B的数量-相同的数量
                            # v_items = pd.Series(np.ones(len(self.train[v])),
                            #                                     index=self.train[v])
                            # common_item_num = (u_items * v_items).sum()
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
            sorted_item: (itemid, score) topn
        '''
        watch_item = self.train.get(uid, set())
        item_score = {}
        for user in self.train:
            # 带上权重系数，每个用户对item的推荐得分要乘上该用户的degree
            if user == uid:
                continue
            for item in self.train.get(user, {}):
                if item in watch_item:
                    continue
                # assert self.user_sim[uid][user] == self.user_sim[user][uid]
                if item not in item_score:
                    item_score.setdefault(item, 0.0)
                item_score[item] += self.user_sim[uid][user]

        return item_score.items()

    def get_item_score(self):
        '''
        为每个用户做推荐过程
        获得item的推荐得分
        返回用于存储每个用户对所有item推荐得分的字典
        '''
        print('recommend...')

        rec_score = {}
        for user in tqdm(self.train, ascii=True):
            rec_score[user] = self.recommend(user)
        return rec_score

# 一个附加的方法，来获得最小N个度的itemset
def getNdegree_items(degree_item_map, N=10):
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
        degree = test_item_degree.get(itemid, 0)
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
        corr_score[degree] = rec_series[degree].corr(
            real_series[degree], method=method)
    return corr_score


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


if __name__ == "__main__":

    train_set, test, item_len = utils.deal_train(r'./data/movielens_data.pkl')
    cf = UserCF(train_set, item_len)
    cf.cal_user_degree()
    cf.similarity()
    # 定义一个浮点跳步器

    def frange(x, y, jump):
        while x < y:
            yield x
            x += jump
    rec_score = cf.get_item_score()

    degreedistrev = degree_item_map(cf)
    Ndegree_items = getNdegree_items(degreedistrev, N=10)
    test_item_degree = get_test_degree(test)
    for p1 in frange(-1.0, 1.01, 0.1):
        train_itemscore = {}
        for user in rec_score:    
            weight = pow(cf.user_degree[user], p1)
            for itemid, score in rec_score[user]:
                pre_score = train_itemscore.get(itemid, 0)
                train_itemscore[itemid] = pre_score + score * weight

        corr_score = trend_predict(
            train_itemscore, Ndegree_items, test_item_degree)
        print(corr_score)
