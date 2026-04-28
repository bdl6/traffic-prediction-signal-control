import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

def load_data(file_path):
    data = pd.read_csv(file_path).values
    return data

def normalize_data(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def build_spatiotemporal_dataset(data, his_steps=24, pre_steps=12, node_num=12):
    X, Y = [], []
    total_length = data.shape[0]
    for i in range(total_length - his_steps - pre_steps + 1):
        x_seq = data[i:i+his_steps]
        y_seq = data[i+his_steps:i+his_steps+pre_steps]
        X.append(x_seq)
        Y.append(y_seq)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def split_dataset(X, Y, train_ratio=0.7, val_ratio=0.2):
    total = len(X)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def to_tensor(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    Y_train = torch.FloatTensor(Y_train)
    Y_val = torch.FloatTensor(Y_val)
    Y_test = torch.FloatTensor(Y_test)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test