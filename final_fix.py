#!/usr/bin/env python3
"""
Final targeted fix - addressing the fundamental issues
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("FINAL TARGETED MODEL FIX")
    print("=" * 50)

    # Load data
    train_df = pd.read_csv(r"C:\Users\Bhekiz\Documents\School\Independent study\hull-tactical-market-prediction\train.csv")

    # Key insight: Use only the most recent data where features are most complete
    # and most predictive (less missing data)
    print("STRATEGY: Use only recent, high-quality data")
    print("-" * 45)

    # Use only the last 50% of data where missing data is minimal
    cutoff = len(train_df) // 2
    recent_df = train_df.iloc[cutoff:].copy()
    print(f"Using recent data: {len(recent_df)} samples (from date_id {recent_df.iloc[0]['date_id']} onwards)")

    # Check missing data in recent period
    features = ['E2', 'E3', 'M2', 'M5', 'I1', 'I2', 'I3', 'I4', 'I5']
    available_features = [f for f in features if f in recent_df.columns]

    missing_pct = (recent_df[available_features].isnull().sum() / len(recent_df)) * 100
    print(f"Missing data % in recent period:")
    for feat, pct in missing_pct.items():
        print(f"  {feat}: {pct:.1f}%")

    # Further filter to only use data with complete features
    complete_data = recent_df.dropna(subset=available_features)
    print(f"Complete cases: {len(complete_data)} samples")

    if len(complete_data) < 1000:
        print("Not enough complete cases, using forward-fill approach")
        complete_data = recent_df.copy()
        complete_data[available_features] = complete_data[available_features].ffill().bfill()

    # Simple temporal split on the filtered data
    n = len(complete_data)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_data = complete_data.iloc[:train_end]
    val_data = complete_data.iloc[train_end:val_end]
    test_data = complete_data.iloc[val_end:]

    print(f"Final split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Prepare clean data
    X_train = train_data[available_features].values
    y_train = train_data['risk_free_rate'].values
    X_val = val_data[available_features].values
    y_val = val_data['risk_free_rate'].values
    X_test = test_data[available_features].values
    y_test = test_data['risk_free_rate'].values

    # Simple standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTarget statistics in train set:")
    print(f"  Mean: {y_train.mean():.8f}")
    print(f"  Std: {y_train.std():.8f}")
    print(f"  Range: [{y_train.min():.8f}, {y_train.max():.8f}]")

    # Try extremely simple models
    models = {
        'Very_Simple_Ridge': Ridge(alpha=0.1),
        'Simple_Mean_Model': None  # Just predict the mean
    }

    results = {}

    # Ridge model
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train_scaled, y_train)

    y_val_pred_ridge = ridge.predict(X_val_scaled)
    val_r2_ridge = r2_score(y_val, y_val_pred_ridge)
    val_mse_ridge = mean_squared_error(y_val, y_val_pred_ridge)

    # Mean model (baseline)
    mean_pred = np.full_like(y_val, y_train.mean())
    val_r2_mean = r2_score(y_val, mean_pred)
    val_mse_mean = mean_squared_error(y_val, mean_pred)

    print(f"\nRESULTS:")
    print(f"Ridge Model - Val R²: {val_r2_ridge:.6f}, MSE: {val_mse_ridge:.2e}")
    print(f"Mean Model - Val R²: {val_r2_mean:.6f}, MSE: {val_mse_mean:.2e}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'coefficient': ridge.coef_
    }).sort_values('coefficient', key=abs, ascending=False)

    print(f"\nFeature coefficients (Ridge):")
    print(feature_importance)

    # The key insight: Check if predictions have any signal
    pred_std = y_val_pred_ridge.std()
    target_std = y_val.std()

    print(f"\nSignal analysis:")
    print(f"  Target std: {target_std:.8f}")
    print(f"  Prediction std: {pred_std:.8f}")
    print(f"  Prediction/Target std ratio: {pred_std/target_std:.3f}")

    # Correlation between predictions and actuals
    corr = np.corrcoef(y_val, y_val_pred_ridge)[0,1]
    print(f"  Correlation: {corr:.6f}")

    # Final assessment
    print(f"\nFINAL ASSESSMENT:")
    print("-" * 20)

    if val_r2_ridge > 0.01:
        print("SUCCESS: Achieved positive R² > 0.01")
        status = "SUCCESS"
    elif val_r2_ridge > 0:
        print("PARTIAL SUCCESS: Achieved positive R²")
        status = "PARTIAL"
    elif val_r2_ridge > -0.1:
        print("CLOSE: R² is close to zero")
        status = "CLOSE"
    else:
        print("FAILURE: R² is significantly negative")
        status = "FAILURE"

    # Recommendation
    print(f"\nRECOMMENDATION:")
    if status in ["SUCCESS", "PARTIAL"]:
        print("Model shows some predictive power. Consider:")
        print("- Expanding feature set")
        print("- Ensemble methods")
        print("- More sophisticated time series models")
    else:
        print("Fundamental issues remain. Consider:")
        print("- Risk-free rate may be too smooth/predictable to model with these features")
        print("- Focus on the excess returns model instead")
        print("- Use external risk-free rate data/models")

    return {
        'val_r2': val_r2_ridge,
        'val_mse': val_mse_ridge,
        'correlation': corr,
        'status': status
    }

if __name__ == "__main__":
    results = main()
    print(f"\nFinal result: {results}")