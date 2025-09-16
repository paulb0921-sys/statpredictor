# nfl_stat_predictor_streamlit_safe.py

# Streamlit-safe imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Try LightGBM if installed
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

RANDOM_STATE = 42

# -------------------------
# Data utilities
# -------------------------
@st.cache_data
def load_player_logs(file):
    df = pd.read_csv(file, parse_dates=['date'])
    df = df.sort_values(['player_id', 'date']).reset_index(drop=True)
    return df

@st.cache_data
def load_team_defense(file):
    if file is None:
        return None
    df = pd.read_csv(file)
    return df

# -------------------------
# Feature engineering
# -------------------------
# ... (rest of your functions like add_rolling_features, add_matchup_features, build_feature_matrix, train_model, predict_from_row remain unchanged)
