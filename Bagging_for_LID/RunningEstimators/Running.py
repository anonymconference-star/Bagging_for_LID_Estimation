import pandas as pd
from tqdm import tqdm
import multiprocessing
import pickle
import os
###############################################################################################################################RUNNING ESTIMATORS###############################################################################################################################
def save_to_df(d, save_name):
    directory = r'.\csvs'
    df = pd.DataFrame.from_dict(d, orient="index")
    df.to_csv(directory + '\\' + f'{save_name}.csv')

def save_dict(data, directory, filename):
    os.makedirs(directory, exist_ok=True)  # Ensure directory exists
    filepath = os.path.join(directory, filename)  # Create full path
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)