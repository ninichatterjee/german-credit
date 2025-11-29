"""
Data loading and splitting utilities for the German Credit dataset.
"""
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED

def load_raw_data():
    """Load German Credit dataset from CSV."""
    import os
    # Handle both running from root and from src/
    if os.path.exists('data/raw/german_credit.csv'):
        filename = 'data/raw/german_credit.csv'
    else:
        filename = '../data/raw/german_credit.csv'
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = list(reader)
        data = np.array(data)
    
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    return X, y, headers


def split_data_classification(X, y, test_size=0.15, val_size=0.15):
    """Split data for classification task."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=RANDOM_SEED, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data_regression(X, y, test_size=0.15, val_size=0.15):
    """Split data for regression task."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=RANDOM_SEED
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test