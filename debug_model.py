#!/usr/bin/env python3
"""
Debug script to identify and fix the model training issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("DEBUGGING MODEL TRAINING ISSUES")
    print("=" * 50)

    # Load data
    train_df = pd.read_csv(r"C:\Users\Bhekiz\Documents\School\Independent study\hull-tactical-market-prediction\train.csv")

    # 1. Examine the target variable
    print("\nTARGET VARIABLE ANALYSIS")
    print("-" * 30)
    risk_free = train_df['risk_free_rate'].dropna()
    print(f"Shape: {risk_free.shape}")
    print(f"Mean: {risk_free.mean():.10f}")
    print(f"Std: {risk_free.std():.10f}")
    print(f"Min: {risk_free.min():.10f}")
    print(f"Max: {risk_free.max():.10f}")
    print(f"Variance: {risk_free.var():.15f}")
    print(f"Unique values: {len(risk_free.unique())}")

    # Check if target has sufficient variance
    if risk_free.std() < 1e-10:
        print("WARNING: Target variable has extremely low variance!")

    # 2. Examine feature data quality
    print("\nFEATURE DATA QUALITY")
    print("-" * 25)

    # Define features (reduced set for debugging)
    interest_features = ['I1', 'I2', 'I3', 'I4', 'I5']
    economic_features = ['E1', 'E2', 'E3', 'E4', 'E5']
    market_features = ['M1', 'M2', 'M3', 'M4', 'M5']

    debug_features = interest_features + economic_features + market_features
    available_features = [f for f in debug_features if f in train_df.columns]
    print(f"Using {len(available_features)} features for debugging: {available_features}")

    # Check missing data
    missing_counts = train_df[available_features].isnull().sum()
    print(f"Missing data counts:\n{missing_counts}")

    # 3. Create a proper time-series split
    print("\nTIME-SERIES SPLIT")
    print("-" * 20)

    n_total = len(train_df)
    train_size = int(0.7 * n_total)
    val_size = int(0.15 * n_total)

    # Split data maintaining temporal order
    train_data = train_df.iloc[:train_size].copy()
    val_data = train_df.iloc[train_size:train_size+val_size].copy()
    test_data = train_df.iloc[train_size+val_size:].copy()

    print(f"Train: {len(train_data)} samples (dates {train_data.iloc[0]['date_id']} to {train_data.iloc[-1]['date_id']})")
    print(f"Val: {len(val_data)} samples (dates {val_data.iloc[0]['date_id']} to {val_data.iloc[-1]['date_id']})")
    print(f"Test: {len(test_data)} samples (dates {test_data.iloc[0]['date_id']} to {test_data.iloc[-1]['date_id']})")

    # 4. Better missing value handling
    print("\nIMPROVED DATA PREPROCESSING")
    print("-" * 35)

    # Extract features and target
    X_train = train_data[available_features].copy()
    y_train = train_data['risk_free_rate'].copy()

    X_val = val_data[available_features].copy()
    y_val = val_data['risk_free_rate'].copy()

    X_test = test_data[available_features].copy()
    y_test = test_data['risk_free_rate'].copy()

    print(f"Original shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")

    # Advanced missing value handling
    # 1. Forward fill (use previous day's value)
    # 2. Backward fill for any remaining
    # 3. Fill with median as last resort

    def better_fill_missing(df):
        """Better missing value imputation"""
        df_filled = df.copy()

        # Forward fill (carry last observation forward)
        df_filled = df_filled.ffill()

        # Backward fill for any remaining
        df_filled = df_filled.bfill()

        # Fill any remaining with median
        for col in df_filled.columns:
            if df_filled[col].isnull().any():
                median_val = df_filled[col].median()
                if pd.isna(median_val):  # If all values are NaN
                    median_val = 0
                df_filled[col] = df_filled[col].fillna(median_val)

        return df_filled

    X_train_filled = better_fill_missing(X_train)
    X_val_filled = better_fill_missing(X_val)
    X_test_filled = better_fill_missing(X_test)

    print(f"After filling - NaNs: X_train={X_train_filled.isnull().sum().sum()}, X_val={X_val_filled.isnull().sum().sum()}, X_test={X_test_filled.isnull().sum().sum()}")

    # 5. Train models with proper error handling
    print("\nMODEL TRAINING WITH DEBUG")
    print("-" * 30)

    # Model 1: Ridge Regression
    print("\nRidge Regression:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filled)
    X_val_scaled = scaler.transform(X_val_filled)
    X_test_scaled = scaler.transform(X_test_filled)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)

    y_train_pred = ridge.predict(X_train_scaled)
    y_val_pred = ridge.predict(X_val_scaled)
    y_test_pred = ridge.predict(X_test_scaled)

    # Debug predictions
    print(f"  y_train range: [{y_train.min():.8f}, {y_train.max():.8f}]")
    print(f"  y_train_pred range: [{y_train_pred.min():.8f}, {y_train_pred.max():.8f}]")
    print(f"  Prediction std: {y_train_pred.std():.10f}")
    print(f"  Target std: {y_train.std():.10f}")

    # Calculate metrics with debug info
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"  Train R²: {train_r2:.6f}, MSE: {train_mse:.10f}")
    print(f"  Val R²: {val_r2:.6f}, MSE: {val_mse:.10f}")
    print(f"  Test R²: {test_r2:.6f}, MSE: {test_mse:.10f}")

    # Model 2: Random Forest
    print("\nRandom Forest:")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_filled, y_train)

    y_train_pred_rf = rf.predict(X_train_filled)
    y_val_pred_rf = rf.predict(X_val_filled)
    y_test_pred_rf = rf.predict(X_test_filled)

    print(f"  y_train_pred_rf range: [{y_train_pred_rf.min():.8f}, {y_train_pred_rf.max():.8f}]")
    print(f"  Prediction std: {y_train_pred_rf.std():.10f}")

    train_r2_rf = r2_score(y_train, y_train_pred_rf)
    val_r2_rf = r2_score(y_val, y_val_pred_rf)
    test_r2_rf = r2_score(y_test, y_test_pred_rf)

    train_mse_rf = mean_squared_error(y_train, y_train_pred_rf)
    val_mse_rf = mean_squared_error(y_val, y_val_pred_rf)
    test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)

    print(f"  Train R²: {train_r2_rf:.6f}, MSE: {train_mse_rf:.10f}")
    print(f"  Val R²: {val_r2_rf:.6f}, MSE: {val_mse_rf:.10f}")
    print(f"  Test R²: {test_r2_rf:.6f}, MSE: {test_mse_rf:.10f}")

    # 6. Diagnosis
    print("\nDIAGNOSIS")
    print("-" * 15)

    # Check if the issue is with the target variable
    mean_baseline = np.full_like(y_val, y_train.mean())
    baseline_mse = mean_squared_error(y_val, mean_baseline)
    print(f"Baseline (mean) MSE on validation: {baseline_mse:.10f}")

    # Check correlations
    correlation_with_target = X_train_filled.corrwith(y_train).abs().sort_values(ascending=False)
    print(f"Top 5 feature correlations with target:")
    print(correlation_with_target.head())

    # Summary
    print(f"\nSUMMARY")
    print("-" * 10)
    if val_r2 < 0 and val_r2_rf < 0:
        print("ISSUE: Both models have negative R² - models are worse than baseline!")
        print("   Possible causes:")
        print("   - Target variable has insufficient signal")
        print("   - Features are not predictive")
        print("   - Data leakage or temporal issues")
        print("   - Overfitting on training data")
    else:
        print("SUCCESS: At least one model shows positive R²")

    if train_mse < 1e-10 or val_mse < 1e-10:
        print("ISSUE: MSE values are suspiciously small - numerical precision issue")

    return {
        'ridge_val_r2': val_r2,
        'rf_val_r2': val_r2_rf,
        'ridge_val_mse': val_mse,
        'rf_val_mse': val_mse_rf
    }

if __name__ == "__main__":
    results = main()
    print(f"\nFinal Results: {results}")