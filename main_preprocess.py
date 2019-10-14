from PreprocessData import *
import pickle
import pandas as pd


def smallData():
    # filepath = r'./data/Amazon/Amazon_2.txt'
    # filepath = r'./data/delicious/delicious_subset2.txt'
    filepath = r'./data/Amazon/amazon_uit.csv'
    # filepath = r'./data/movielen5000_7533_link864581_day0_1096.txt'
    # train_data, test_data = readData(filepath, split='\t', train_ratio=0.7, sample_num=150000)    
    train_data, test_data = readData(filepath, split=',', train_ratio=0.7)  
    # train_data.to_csv('delicous_train.csv',header=False,index=False)
    # test_data.to_csv('delicous_test.csv',header=False,index=False)

    # train_data, test_data = rerank(train_data, test_data)
    train_data, test_data = new_rerank(train_data, test_data)


    # train_data, test_data = rerank_multithread(train_data, test_data)
    # train_data, test_data = rerank_multiprocess(train_data, test_data)
    pickle.dump((train_data, test_data), 
                # open(r'./data/delicious_iterrows.pkl','wb'))
                open(r'./data/Amazon_iterrows.pkl','wb'))
                # open(r'./data/Amazon_2.pkl', 'wb+'))
    print("dump finished.")

def selectByUser():
    # filepath = './data/Amazon/amazon_uit.csv'
    # outputpath = './data/Amazon/small_amazon10k.txt'
    filepath = './data/delicious/delicious_subset2.txt'
    # outputpath = './data/delicious/delicious5k.txt'
    outputpath = './data/delicious/delicious5k_test.txt'
    # extractDataByUser(filepath, split=',',usernum=10000,outputfile=outputpath)
    extractDataByUser(filepath, split='\t',usernum=5000,outputfile=outputpath)

def packupDataForUserCF(filepath = r'./data/delicious/small_delicious5k.txt', tofile=r'./data/delicious/delicious5k.pkl'):
    
    train_data, test_data = readData(filepath, split=',', train_ratio=0.7)  
    pickle.dump((train_data, test_data), 
                open(tofile ,'wb'))
    print("dump finished.")

def writeToFile(data, filepath,sep='\t'):
    with(open(filepath, mode='w', encoding='utf-8')) as f:
        for index, row in data.iterrows():
            f.write(str(row[0])+sep+str(row[1])+sep+str(row[2])+'\n')




def rerankData(filepath, to_file):
    train_data, test_data = readData(filepath, split=',', train_ratio=0.7)  
    train_data, test_data = new_rerank(train_data, test_data)
    pickle.dump((train_data, test_data), 
                open(to_file,'wb'))
    print("dump finished.")

if __name__ == "__main__":
    selectByUser()
    # filepath = r'./data/delicious/small_delicious5k.txt'
    # filepath = './data/delicious/delicious5k.txt'
    # tofile = r'./data/delicious/delicious5k_rerank1.pkl'
    # rerankData(filepath, tofile)
    # filepath = r'./data/Amazon/small_amazon5k.txt'
    # tofile = r'./data/Amazon/amazon_5k_norerank.pkl'
    # packupDataForUserCF(filepath, tofile)

    # filepath = r'./data/Amazon/amazon_uit.csv'
    # # filepath = r'./data/delicious/delicious_subset2.txt'
    # trainfile = r'./data/Amazon/train.txt'
    # # # trainfile = r'./data/delicious/train.txt'
    # testfile = r'./data/Amazon/test.txt'
    # # # testfile = r'./data/delicious/test.txt'
    # # smallData()
    # train, test = readData(filepath, split=',', train_ratio=0.7)
    # writeToFile(train, trainfile)
    # writeToFile(test, testfile)
    # print('done.')
    # filepath = r'./data/movielen5000_7533_link864581_day0_1096.txt'
    # tofile = r'./data/movielens.pkl'
    # filepath = r'./data/netflix5k_result.txt'
    # tofile = r'./data/netflix.pkl'
    # packupDataForUserCF(filepath, tofile)
    # selectByUser()
    # filepath = r'./data/delicious/small_delicious5k.txt'
    # tofile=r'./data/delicious/delicious5k.pkl'
    # filepath = r'./data/Amazon/small_amazon5k.txt'
    # tofile=r'./data/Amazon/amazon5k.pkl'
    # rerankData(filepath, tofile)