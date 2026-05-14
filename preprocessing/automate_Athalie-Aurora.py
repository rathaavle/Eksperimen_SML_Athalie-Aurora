"""
automate_Athalie-Aurora.py
Automated preprocessing pipeline untuk California Housing Dataset.
Nama: Athalie Aurora
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_data(save_raw: bool = True) -> pd.DataFrame:
    """Load California Housing dataset dan simpan versi raw-nya."""
    california = fetch_california_housing(as_frame=True)
    df = california.frame

    if save_raw:
        os.makedirs('california_housing_raw', exist_ok=True)
        raw_path = 'california_housing_raw/california_housing_raw.csv'
        df.to_csv(raw_path, index=False)
        print(f'[INFO] Raw data saved to {raw_path}')

    print(f'[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns')
    return df


# ─────────────────────────────────────────────
# 2. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Imputasi missing values menggunakan median."""
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    print(f'[INFO] Missing values handled. Remaining: {X_imputed.isnull().sum().sum()}')
    return X_imputed


# ─────────────────────────────────────────────
# 3. HANDLE OUTLIERS
# ─────────────────────────────────────────────
def clip_outliers(df_input: pd.DataFrame) -> pd.DataFrame:
    """Clip outlier menggunakan metode IQR (1.5x)."""
    df_clipped = df_input.copy()
    for col in df_clipped.columns:
        Q1 = df_clipped[col].quantile(0.25)
        Q3 = df_clipped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clipped[col] = df_clipped[col].clip(lower, upper)
    print('[INFO] Outlier clipping selesai.')
    return df_clipped


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Tambah fitur baru hasil rekayasa fitur."""
    X = X.copy()
    X['RoomsPerHousehold']      = X['AveRooms']    / X['HouseAge'].replace(0, 1)
    X['BedroomsPerRoom']        = X['AveBedrms']   / X['AveRooms'].replace(0, 1)
    X['PopulationPerHousehold'] = X['Population']  / X['AveOccup'].replace(0, 1)
    print(f'[INFO] Feature engineering selesai. Shape: {X.shape}')
    return X


# ─────────────────────────────────────────────
# 5. SPLIT & SCALING
# ─────────────────────────────────────────────
def split_and_scale(X: pd.DataFrame, y: pd.Series,
                    test_size: float = 0.2,
                    random_state: int = 42):
    """Train-test split dan StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X.columns)

    print(f'[INFO] Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}')
    return X_train_scaled, X_test_scaled, y_train, y_test


# ─────────────────────────────────────────────
# 6. SAVE PREPROCESSED DATA
# ─────────────────────────────────────────────
def save_preprocessed(X_train, X_test, y_train, y_test,
                       output_dir: str = 'preprocessing/california_housing_preprocessing'):
    """Simpan hasil preprocessing ke CSV."""
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['MedHouseVal'] = y_train.values

    test_df = X_test.copy()
    test_df['MedHouseVal'] = y_test.values

    train_path = os.path.join(output_dir, 'train.csv')
    test_path  = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    print(f'[INFO] Train saved to {train_path}')
    print(f'[INFO] Test  saved to {test_path}')


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_preprocessing():
    print('=' * 50)
    print('  Automated Preprocessing Pipeline')
    print('  California Housing Dataset')
    print('  Nama: Athalie Aurora')
    print('=' * 50)

    # Step 1: Load
    df = load_data(save_raw=True)

    # Step 2: Pisah fitur & target
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    # Step 3: Handle missing values
    X = handle_missing_values(X)

    # Step 4: Handle outliers
    X = clip_outliers(X)

    # Step 5: Feature engineering
    X = feature_engineering(X)

    # Step 6: Split & scale
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    # Step 7: Save
    save_preprocessed(X_train, X_test, y_train, y_test)

    print('=' * 50)
    print('[DONE] Preprocessing selesai!')
    print('=' * 50)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    run_preprocessing()
