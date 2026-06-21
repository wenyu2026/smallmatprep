#!/usr/bin/env python3
"""Fast baseline benchmark - Steel dataset, 7 core methods."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import entropy, dirichlet
from crisp import CRISPImputer, LCRISPImputer, AutoCRISPImputer, project_to_simplex, check_compositional_validity
from matminer.datasets import load_dataset

from matminer.datasets import load_dataset

def load_steel(n=200):
    df = load_dataset("steel_strength").dropna()
    cols = [c for c in ['c','mn','si','cr','ni','mo','v','n','nb','co','w','al','ti'] if c in df.columns]
    X = df[cols].values * 100
    for i in range(len(X)):
        s = X[i].sum()
        if s > 0: X[i] = X[i]/s*100
    return X[:n].copy(), cols

# ... (rest of file truncated for brevity)
