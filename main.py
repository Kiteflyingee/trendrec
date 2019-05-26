from PreprocessData import *
import pickle

if __name__ == "__main__":

    filepath = r'./data/Amazon/Amazon_2.txt'
    # filepath = r'./data/movielen5000_7533_link864581_day0_1096.txt'
    # train_data, test_data = readData(filepath, split='\t', train_ratio=0.8, sample_num=400000)    
    train_data, test_data = readData(filepath, split=',', train_ratio=0.7)  
    train_data, test_data = rerank(train_data, test_data)
    pickle.dump((train_data, test_data), 
                open(r'./data/Amazon_2.pkl', 'wb+'))
    print("dump finished.")