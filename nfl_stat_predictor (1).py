"""
nfl_stat_predictor.py

Simple pipeline to train and predict NFL player stat lines (passing/rushing/receiving yards).
Usage examples at the bottom of the file.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

RANDOM_STATE = 42

# ... [All code from your previous file remains unchanged, including functions load_player_logs, add_rolling_features, add_matchup_features, build_feature_matrix, train_model, load_model, predict_from_row, and example_usage] ...

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train/predict NFL player stat lines.")
    parser.add_argument('--train_csv', type=str, default=None, help='Historical player game logs CSV')
    parser.add_argument('--defense_csv', type=str, default=None, help='Optional defensive team CSV to merge')
    parser.add_argument('--target', type=str, default='passing_yards', help='Which stat to predict')
    parser.add_argument('--model_out', type=str, default='nfl_stat_model.joblib', help='Where to save model')
    parser.add_argument('--predict_player_id', type=str, default=None, help='Optional player_id to run a predict example after training')
    args = parser.parse_args()

    if args.train_csv is None:
        example_usage()
        exit(0)

    df_logs = load_player_logs(args.train_csv)
    defense_df = load_team_defense(args.defense_csv) if args.defense_csv else None

    stat_cols = [c for c in ['passing_yards','rushing_yards','receiving_yards','targets','pass_attempts','snaps'] if c in df_logs.columns]

    df_eng = add_rolling_features(df_logs, stat_cols, window=3)
    df_eng = add_matchup_features(df_eng, defense_df)

    X, y, feature_cols = build_feature_matrix(df_eng, target_stat=args.target)

    model_pkg = train_model(X, y, model_path=args.model_out, use_lightgbm=True)

    if args.predict_player_id:
        pid = args.predict_player_id
        player_rows = df_eng[df_eng['player_id'] == pid].sort_values('date', ascending=False)
        if player_rows.shape[0] == 0:
            print(f"No rows for player_id {pid}")
        else:
            candidate = player_rows.iloc[0:1]
            missing = [c for c in model_pkg['feature_names'] if c not in candidate.columns]
            for c in missing:
                candidate[c] = 0
            pred = predict_from_row(model_pkg, candidate)
            print(f"Predicted {args.target} for player {pid}: {pred:.2f}")

    print("Done.")
