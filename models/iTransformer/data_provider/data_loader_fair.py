"""
Fair data loader for iTransformer that uses the SAME chronological split
with gaps as DiffusionTSF to prevent data leakage and ensure fair comparison.

Split logic (matches DiffusionTSF exactly):
- Train: indices 0 to 70% of samples
- [GAP]: ceil(window_size / stride) indices  
- Val: indices after gap to 80%
- [GAP]: ceil(window_size / stride) indices
- Test: indices after gap to 100%

This ensures NO window overlap between splits.
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class Dataset_Fair(Dataset):
    """
    Fair dataset for iTransformer with gap-based chronological splits.
    Matches DiffusionTSF's data handling exactly.
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        
        if size is None:
            self.seq_len = 512
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Get target column
        if self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_data = df_raw[cols + [self.target]]
        
        data = df_data.values
        total_rows = len(data)
        
        # =================================================================
        # FAIR SPLIT LOGIC (matches DiffusionTSF exactly)
        # =================================================================
        window_size = self.seq_len + self.pred_len  # 512 + 96 = 608
        stride = 1  # Same as DiffusionTSF training
        
        # Calculate total possible samples
        total_samples = (total_rows - window_size) // stride + 1
        
        # Gap size: ensures no window overlap between splits
        gap_indices = (window_size + stride - 1) // stride  # ceil(608/1) = 608
        
        # Target proportions: 70% train, 10% val, 20% test
        train_end = int(total_samples * 0.7)
        val_start = train_end + gap_indices
        val_end = int(total_samples * 0.8)
        test_start = val_end + gap_indices
        
        # Handle edge cases for small datasets
        if val_start >= val_end:
            val_start = train_end + 1
        if test_start >= total_samples:
            test_start = val_end + 1
        
        # Convert sample indices to row indices
        # Sample i covers rows [i*stride, i*stride + window_size)
        train_row_end = train_end * stride + window_size if train_end > 0 else window_size
        val_row_start = val_start * stride
        val_row_end = val_end * stride + window_size if val_end > val_start else val_row_start + window_size
        test_row_start = test_start * stride
        
        # Clamp to valid range
        train_row_end = min(train_row_end, total_rows)
        val_row_start = min(val_row_start, total_rows)
        val_row_end = min(val_row_end, total_rows)
        test_row_start = min(test_row_start, total_rows)
        
        # Define borders based on flag
        if self.flag == 'train':
            border1 = 0
            border2 = train_row_end
            self.n_samples = train_end
        elif self.flag == 'val':
            border1 = val_row_start
            border2 = val_row_end
            self.n_samples = val_end - val_start
        else:  # test
            border1 = test_row_start
            border2 = total_rows
            self.n_samples = total_samples - test_start
        
        # Ensure we have valid ranges
        if border1 >= border2 or border2 > total_rows:
            print(f"Warning: Invalid borders for {self.flag}: [{border1}, {border2}), total={total_rows}")
            border1 = 0
            border2 = min(window_size + 100, total_rows)
            self.n_samples = max(1, (border2 - border1 - window_size) // stride + 1)
        
        print(f"[Fair Split] {self.flag}: rows [{border1}, {border2}), ~{self.n_samples} samples")
        print(f"  Window: {window_size}, Gap: {gap_indices} indices")
        
        # =================================================================
        # NORMALIZATION (fit on train only, like DiffusionTSF)
        # =================================================================
        if self.scale:
            # Always fit scaler on train data only
            train_border2 = train_end * stride + window_size if train_end > 0 else window_size
            train_border2 = min(train_border2, total_rows)
            train_data = data[0:train_border2]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        
        # Store data for this split
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        # Handle timestamps
        if 'date' in df_raw.columns:
            df_stamp = df_raw[['date']][border1:border2].copy()
            df_stamp['date'] = pd.to_datetime(df_stamp['date'])
            
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], axis=1).values
            else:
                from utils.timefeatures import time_features
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        else:
            # No date column - create dummy timestamps
            data_stamp = np.zeros((len(self.data_x), 4))
        
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # Bounds check
        if r_end > len(self.data_x):
            # Clamp to valid range
            r_end = len(self.data_x)
            r_begin = max(0, r_end - self.label_len - self.pred_len)
            s_end = r_begin + self.label_len
            s_begin = max(0, s_end - self.seq_len)
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

