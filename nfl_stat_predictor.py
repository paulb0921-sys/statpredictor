import streamlit as st

# -------------------------
# Safe imports with error handling
# -------------------------
try:
    import pandas as pd
except ImportError:
    st.error('pandas is not installed. Run `pip install pandas`.')
    st.stop()

try:
    import numpy as np
except ImportError:
    st.error('numpy is not installed. Run `pip install numpy`.')
    st.stop()

try:
    import joblib
except ImportError:
    st.error('joblib is not installed. Run `pip install joblib`.')
    st.stop()

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError:
    st.error('scikit-learn is not installed. Run `pip install scikit-learn`.')
    st.stop()

# Optional LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

RANDOM_STATE = 42

# -------------------------
# The rest of your Streamlit app functions remain unchanged
# (load_player_logs, load_team_defense, add_rolling_features, add_matchup_features,
# build_feature_matrix, train_model, predict_from_row, and the Streamlit UI code)