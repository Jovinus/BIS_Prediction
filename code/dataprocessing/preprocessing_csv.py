# %%
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
pd.set_option("display.max_columns", None)
# %%

def simple_preprocess(file_path):
    
    data = np.load(file_path)
    
    threshold = 90
    data = np.append(data, np.where(np.isnan(data[:, 0]), 0, 1).reshape(-1, 1), axis=1)
    data = np.append(data, np.where(data[:, 1] >= threshold, 1, 0).reshape(-1, 1), axis=1)
    data = np.append(data, np.where(data[:, 3:5].sum(axis=1) == 2, 1, 0).reshape(-1, 1), axis=1)
    
    end_points = np.where(data[:, 5] == 1)
    
    return data, end_points
    

# %%

if __name__ == '__main__':
    
    DATAPATH = '../data/vitaldb/raw'
    file_list = glob(pathname=DATAPATH+"/*npy")
    
    results = pd.DataFrame()
        
    for file in tqdm(file_list):
        
        data, end_points = simple_preprocess(file_path=file)
        
        result = pd.DataFrame()
        
        for seq, end_point in enumerate(end_points[0]):
            eeg_data = data[end_point - 100*25 -100*30:end_point - 100*25, 2]
            eeg_data = pd.DataFrame(eeg_data).T
            eeg_data['seq_num'] = seq
            eeg_data['label'] = data[end_point, 0]
            
            result = result.append(eeg_data)
        
        result['caseid'] = file[20:-4]          
            
        results = pd.concat((results, result), axis=0)
    
# %%

