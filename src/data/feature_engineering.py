import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

class FeatureProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # Identifies features with large magnitudes requiring log transformation[cite: 4, 7]
        self.log_features = ['sys-mem-total', 'sys-mem-free', 'disk-bytes-written', 'disk-bytes-read']

    def fit_transform(self, df: pd.DataFrame):
        df_processed = df.copy()
        
        for col in df_processed.columns:
            if col in self.log_features:
                # X_log = log(1 + X)[cite: 4]
                df_processed[col] = np.log1p(df_processed[col].clip(lower=0))
                
        # Min-Max scaling across all features[cite: 4]
        scaled_data = self.scaler.fit_transform(df_processed)
        return pd.DataFrame(scaled_data, columns=df_processed.columns)

    def transform(self, df: pd.DataFrame):
        df_processed = df.copy()
        
        for col in df_processed.columns:
            if col in self.log_features:
                df_processed[col] = np.log1p(df_processed[col].clip(lower=0))
                
        scaled_data = self.scaler.transform(df_processed)
        return pd.DataFrame(scaled_data, columns=df_processed.columns)

    def save_scaler(self, path="models/feature_processor.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_scaler(cls, path="models/feature_processor.pkl"):
        with open(path, 'rb') as f:
            return pickle.load(f)