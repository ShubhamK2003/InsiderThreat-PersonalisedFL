# data_loader.py
"""
Loads CSV, creates per-client splits and per-client StandardScaler.
Partitioning is based on config.CLIENT_PARTITION_COLUMN.
This version fits a separate StandardScaler for each client on its
own training data, as requested.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import torch
import config
import random

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

def load_raw_sessions(filepath):
    """Loads the raw CSV for session-level analysis (Session Mode)."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
    
    # Ensure all numerical feature columns exist and are numeric
    for col in config.NUMERICAL_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df[config.NUMERICAL_FEATURES] = df[config.NUMERICAL_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Keep only the necessary columns
    columns_to_keep = ['user_id', config.CLIENT_PARTITION_COLUMN, 'date'] + config.NUMERICAL_FEATURES
    
    # Ensure partition column exists
    if config.CLIENT_PARTITION_COLUMN not in df.columns:
         raise KeyError(f"Partition column '{config.CLIENT_PARTITION_COLUMN}' not found in CSV. Check data or config.")
         
    return df[columns_to_keep]


def prepare_clients_for_training(csv_path):
    """
    Orchestrates per-client temporal splitting and per-client scaling.
    Returns a dictionary mapping client_id -> {'train_df', 'test_df', 'scaler'}
    """
    df = load_raw_sessions(csv_path)
    
    clients = {}
    
    # group by client id and do a temporal split
    for client_id, g in df.groupby(config.CLIENT_PARTITION_COLUMN):
        
        # sort by date to ensure temporal split
        g = g.sort_values('date')
        arr = g.reset_index(drop=True)
        
        n = len(arr)
        if n == 0:
            continue
            
        # 1. Get default temporal split
        split_idx = int(np.floor(config.TRAIN_SPLIT_RATIO * n))
        train_df = arr.iloc[:split_idx].copy()
        test_df = arr.iloc[split_idx:].copy()
        
        # Check for empty train_df (can happen if split=0 or client has 1 sample)
        if len(train_df) == 0:
            print(f"⚠️ Client {client_id} has no training data after split. Skipping client.")
            continue
            
        # 2. Fit a PER-CLIENT scaler
        scaler = StandardScaler()
        X_train = train_df[config.NUMERICAL_FEATURES].astype(float).values
        scaler.fit(X_train)
        
        clients[client_id] = {
            'train_df': train_df,
            'test_df': test_df,
            'scaler': scaler
        }
        
    print(f"✅ Data partitioned into {len(clients)} clients: {list(clients.keys())}")
    return clients