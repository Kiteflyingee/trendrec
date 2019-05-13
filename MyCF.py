#coding=utf-8


class UserCF:
    '''
    基于用户的协同过滤
    '''

    def __init__(self, train, test, item_len, topn=100 , k=100):
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
            self.user_degree[user] = degree
        return self.user_degree

    def cal_item_degree(self):
        self.item_degree = {}

        for _, items in self.train.items():
            for item in items:
                self.item_degree.setdefault(item, 0)
                self.item_degree[item] = self.item_degree[item] + 1

        return self.item_degree

    def similarity(self):
        '''
        统计的训练集的用户的相似度信息
        在训练集中的用户两两求相似性
        '''
        self.user_sim = {}
        try:
            with tqdm(self.train, ascii=True ) as T:
                for u in T:
                    for v in self.train:
                        #                 如果这两个用户计算过相似性
                        if u == v:
                            continue
                        if (u in self.user_sim and v in self.user_sim[u]) or \
                                (v in self.user_sim and u in self.user_sim[v]):
                            continue
                        else:
                            u_buy = utils.deal_buy(self.train[u], self.item_len)
                            v_buy = utils.deal_buy(self.train[v], self.item_len)
                            sim = utils.cal_sim(u_buy, v_buy)
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

        return rank_item.items()

    def get_score(self):
        '''
        获得训练集所有用户的推荐得分
        '''
        recommend_score = {}
        print('计算推荐得分')
        for user in tqdm(self.train, ascii=True):
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
        self.similarity()
        print('calculate similarity finished. cost {:.2f}s'.format(
            time.time()-start))

        start = time.time()
        score = self.get_score()
        print('calculate recommender score. cost {:.2f}s'.format(
            time.time()-start))
        return score
    