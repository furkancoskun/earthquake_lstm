import torch
from torch.utils.data import Dataset
import os
import pandas as pd

# only N-S column is used
class EarthquakeDataset(Dataset):
    def __init__(self, data_dir, txt_file_list, window_size):
        self.sequences = []
        for txt_file in txt_file_list:
            txt_path = os.path.join(data_dir, txt_file)
            dataframe = pd.read_csv(txt_path, delim_whitespace=True, header=17, encoding ='latin1', on_bad_lines='skip')['N-S']
            for i in range (dataframe.size - window_size - 1):
                sequence = torch.as_tensor(dataframe.iloc[i:i+window_size].values, dtype=torch.float32)
                next_value = torch.as_tensor(dataframe.iloc[i+window_size], dtype=torch.float32)
                self.sequences.append({'sequence':sequence,
                                       'next_value':next_value})

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]['sequence']
        next_value = self.sequences[idx]['next_value']
        return sequence, next_value
    