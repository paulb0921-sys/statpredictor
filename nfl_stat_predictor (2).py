import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

import streamlit as st

RANDOM_STATE = 42

# -------------------------
# Data utilities
# -------------------------
def load_player_logs(file):
    df = pd.read_csv(file, parse_dates=['date'])
    df = df.sort_values(['player_id', 'date']).reset_index(drop=True)
    return df

def load_team_defense(file):
    if file is None:
        return None
    df = pd.read_csv(file)
    return df

# -------------------------
# Feature engineering
# -------------------------
def add_rolling_features(df, stat_cols, window=3):
    df_sorted = df.sort_values(['player_id', 'date']).copy()
    for stat in stat_cols:
        roll_mean = df_sorted.groupby('player_id')[stat].rolling(window, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
        roll_std = df_sorted.groupby('player_id')[stat].rolling(window, min_periods=1).std().shift(1).reset_index(level=0, drop=True)
        roll_count = df_sorted.groupby('player_id')[stat].rolling(window, min_periods=1).count().shift(1).reset_index(level=0, drop=True)
        df_sorted[f'{stat}_roll_mean_{window}'] = roll_mean.fillna(0)
        df_sorted[f'{stat}_roll_std_{window}'] = roll_std.fillna(0)
        df_sorted[f'{stat}_roll_count_{window}'] = roll_count.fillna(0)
    return df_sorted

def add_matchup_features(df, defense_df=None):
    df = df.copy()
    if 'date' in df.columns:
        df['season'] = df['date'].dt.year
    if defense_df is not None:
        if 'team' in df.columns and 'season' in df.columns:
            df = df.merge(defense_df, how='left', left_on=['season', 'opponent'], right_on=['season', 'team'], suffixes=('', '_oppdef'))
            df = df.drop(columns=[c for c in df.columns if c.endswith('_oppdef') and c in ['team_oppdef']], errors='ignore')
    if 'game_total' in df.columns and 'team_spread' in df.columns:
        df['implied_points_for'] = (df['game_total'] + df['team_spread']) / 2
    else:
        df['implied_points_for'] = np.nan
    if 'home' not in df.columns:
        df['home'] = 0
    if 'targets' in df.columns:
        df['target_share'] = df['targets'] / (df['targets'] + 1)
    if 'pass_attempts' in df.columns:
        df['pass_attempt_share'] = df['pass_attempts'] / (df['pass_attempts'] + 1)
    return df

# -------------------------
# Model pipeline
# -------------------------
def build_feature_matrix(df, target_stat='passing_yards', drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    df = df.copy()
    y = df[target_stat].fillna(0)
    feature_cols = []
    roll_cols = [c for c in df.columns if ('roll_mean' in c or 'roll_std' in c or 'roll_count' in c)]
    feature_cols += roll_cols
    for c in ['implied_points_for', 'home', 'team_spread', 'game_total', 'snaps', 'target_share', 'pass_attempt_share']:
        if c in df.columns:
            feature_cols.append(c)
    opp_def_cols = [c for c in df.columns if 'opp_' in c or 'allowed' in c or 'def_' in c]
    feature_cols += opp_def_cols
    for c in ['pass_attempts', 'targets', 'snaps', 'receptions', 'rush_attempts']:
        if c in df.columns:
            feature_cols.append(c)
    feature_cols = [c for c in feature_cols if c not in drop_cols]
    feature_cols = list(dict.fromkeys(feature_cols))
    X = df[feature_cols].fillna(0)
    return X, y, feature_cols

def train_model(X, y, use_lightgbm=True):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    if use_lightgbm and HAS_LGB:
        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)
        params = {'objective':'regression','metric':'l2','verbosity':-1,'seed':RANDOM_STATE,'learning_rate':0.05,'num_leaves':31,'num_threads':4}
        model = lgb.train(params, train_ds, valid_sets=[val_ds], early_stopping_rounds=50, num_boost_round=1000, verbose_eval=False)
        wrapper = ('lgb', model)
    else:
        rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        wrapper = ('sklearn_rf', rf)
    val_preds = wrapper[1].predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    rmse = mean_squared_error(y_val, val_preds, squared=False)
    st.success(f'Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}')
    model_package = {'model_type': wrapper[0], 'model_obj': wrapper[1], 'feature_names': list(X.columns)}
    return model_package

def predict_from_row(model_package, df_row):
    feat_names = model_package['feature_names']
    model_type = model_package['model_type']
    model_obj = model_package['model_obj']
    X_row = df_row[feat_names].fillna(0).values.reshape(1, -1)
    pred = model_obj.predict(X_row)
    return float(pred[0])

# -------------------------
# Streamlit UI
# -------------------------
st.title('NFL Player Stat Predictor')

player_file = st.file_uploader('Upload Player Logs CSV', type=['csv'])
defense_file = st.file_uploader('Optional: Upload Team Defense CSV', type=['csv'])
target_stat = st.selectbox('Stat to Predict', ['passing_yards','rushing_yards','receiving_yards'])
train_button = st.button('Train Model')

if train_button and player_file is not None:
    st.info('Loading data...')
    df_logs = load_player_logs(player_file)
    defense_df = load_team_defense(defense_file) if defense_file else None
    stat_cols = [c for c in ['passing_yards','rushing_yards','receiving_yards','targets','pass_attempts','snaps'] if c in df_logs.columns]
    df_eng = add_rolling_features(df_logs, stat_cols, window=3)
    df_eng = add_matchup_features(df_eng, defense_df)
    X, y, feature_cols = build_feature_matrix(df_eng, target_stat=target_stat)
    st.info('Training model...')
    model_pkg = train_model(X, y)
    st.session_state['df_eng'] = df_eng
    st.session_state['model_pkg'] = model_pkg

if 'df_eng' in st.session_state:
    st.header('Predict Player Stat')
    df_eng = st.session_state['df_eng']
    model_pkg = st.session_state['model_pkg']
    player_list = df_eng['player_name'].unique()
    selected_player = st.selectbox('Select Player', player_list)
    player_rows = df_eng[df_eng['player_name']==selected_player].sort_values('date', ascending=False)
    candidate = player_rows.iloc[0:1].copy()
    if st.button('Predict Stat'):
        missing_cols = [c for c in model_pkg['feature_names'] if c not in candidate.columns]
        for c in missing_cols:
            candidate[c] = 0
        pred_value = predict_from_row(model_pkg, candidate)
        st.success(f'Predicted {target_stat} for {selected_player}: {pred_value:.2f}')