import pandas as pd 

def readAmazon(sample_num,split='\t',filepath='./data/Amazon/Amazon_data.txt'):
    data = pd.read_csv(filepath, sep=split, header=None,encoding='utf-8')
    data.columns=['iid','uid','time']
    uid = data.uid
    data = data.drop('uid',axis=1)
    data.insert(0, 'uid',uid)
    # 取最近的sample数据
    data = data.sort_values(by='time', axis=0, ascending=False)[0:sample_num]
    # 嘗試只取前2000天(这个代码只适用与amazon)
    # data = data[data.time > 37816]
    data.to_csv('./data/Amazon/Amazon_2.csv',header=False,index=False)


def rerank(filepath='./data/Amazon/Amazon_2.csv', tofile=r'./data/Amazon/Amazon_2.txt', split=','):
    '''
       重新对amazon数据集进行排序
    '''
    data = pd.read_csv(filepath, sep=split, header=None,encoding='utf-8')
    data.columns = ['uid','iid','time']
    u_unique = data.uid.unique()
    i_unique = data.iid.unique()
    u_id = {}
    i_id = {}

    for idx, uid in enumerate(u_unique):
        u_id[uid] = idx

    for idx, iid in enumerate(i_unique):
        i_id[iid] = idx

    with open(tofile,'w') as f:
        for rowidx in range(len(data)):
            uid = data.iloc[rowidx, 0]
            iid = data.iloc[rowidx, 1]
            time = data.iloc[rowidx, 2]
            f.write(str(u_id[uid]) + "," + str(i_id[iid]) + ',' + str(time) + '\n')
            f.flush()
    print('finish')


def changeAmazon(split='\t',filepath='./data/Amazon/Amazon_data.txt', tofile=r'./data/Amazon/amazon_gpu.csv'):
    data = pd.read_csv(filepath, sep=split, header=None,encoding='utf-8')
    data.columns=['iid','uid','time']
    uid = data.uid
    data = data.drop('uid',axis=1)
    data.insert(0, 'uid',uid)
    data.to_csv(tofile,header=False,index=False)


if __name__ == "__main__":
    # readAmazon(sample_num=150000, filepath='./data/Amazon/Amazon_data.txt')
    # rerank(filepath='./data/Amazon/Amazon_2.csv')
    changeAmazon()