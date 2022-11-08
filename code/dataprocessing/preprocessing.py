# %%
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
pd.set_option("display.max_columns", None)
# %%

def simple_preprocess(file_path):
    
    data = np.load(file_path)
    
    threshold = 90
    data = np.append(data, np.where(np.isnan(data[:, 0]), 0, 1).reshape(-1, 1), axis=1) ## BIS
    data = np.append(data, np.where(data[:, 1] > threshold, 1, 0).reshape(-1, 1), axis=1) ## SQI over threshold
    data = np.append(data, np.where(data[:, 3:5].sum(axis=1) == 2, 1, 0).reshape(-1, 1), axis=1)
    
    end_points = np.where(data[:, 5] == 1)
    
    return data, end_points

def preprocessing_file(file_list, save_path , output_name):
    
    results = {}
    
    for file in tqdm(file_list, desc='preprocessing_case'):
        
        data, end_points = simple_preprocess(file_path=file)
        
        result = {}
        
        for seq, end_point in enumerate(end_points[0]):
            eeg_data = data[end_point - 100*25 -100*30 : end_point - 100*25, 2]
            
            if end_point <= 100*60*30:
                continue
            
            if (len(eeg_data) == 0):
                continue
            
            if np.isin(eeg_data, np.nan).sum() > 0:
                continue
            
            label = data[end_point, 0]
            
            result[seq] = {'eeg':eeg_data, 'label':label}
            
        results[int(file[52:-4])] = result
    
    eeg = [results[caseid][seq]['eeg'] for caseid, values in results.items() for seq, _ in values.items()]
    label = [results[caseid][seq]['label'] for caseid, values in results.items() for seq, _ in values.items()]
    
    del results
    
    np.save(save_path + output_name + '_EEG.npy', np.vstack(eeg).reshape(-1, 3000))
    np.save(save_path + output_name + '_label.npy', np.vstack(label).reshape(-1, 1))


# %%
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    
    DATAPATH = '/home/ubuntu/Studio/BIS_Prediction/data/vitaldb/raw'
    file_list = glob(pathname=DATAPATH + "/*npy")
    
    train_file, val_test_file = train_test_split(file_list, train_size=700, random_state=1004)
    valid_file, test_file = train_test_split(val_test_file, train_size=200, random_state=1004)
    
    SAVE_PATH = "/home/ubuntu/Studio/BIS_Prediction/data/vitaldb/"
    
    preprocessing_file(train_file, SAVE_PATH, 'train')
    preprocessing_file(valid_file, SAVE_PATH, 'valid')
    preprocessing_file(test_file, SAVE_PATH, 'test')
# %%
