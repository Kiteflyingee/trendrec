from PreprocessData import readData
import pickle

if __name__ == "__main__":

    filepath = r'./data/netflix5k_result.txt'
    # train_data, test_data = readData(filepath, split=',', train_ratio=0.8, sample_num=10000)    
    train_data, test_data = readData(filepath, split=',', train_ratio=0.7)    
    pickle.dump((train_data, test_data), 
                open(r'./data/netflix.pkl', 'wb+'))
    print("dump finished.")