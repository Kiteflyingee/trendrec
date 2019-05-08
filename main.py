from PreprocessData import readData
import pickle

if __name__ == "__main__":

    filepath = r'./data/movielen5000_7533_link864581_day0_1096.txt'
    train_data, test_data = readData(filepath, split=',', sample_num=20000)    
    pickle.dump((train_data, test_data), 
                open(r'./data/movielens_data.pkl', 'wb+'))
    print("dump finished.")