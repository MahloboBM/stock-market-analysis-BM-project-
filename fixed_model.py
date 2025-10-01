#!/usr/bin/env python3
"""
Fixed model training script with proper handling of the identified issues
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

def main():
    print("FIXED MODEL TRAINING PIPELINE")
    print("=" * 50)

    # Load data
    train_df = pd.read_csv(r"C:\Users\Bhekiz\Documents\School\Independent study\hull-tactical-market-prediction\train.csv")

    # Focus on highly correlated features based on our analysis
    # Plus some additional features from each category
    high_correlation_features = ['E3', 'E2', 'M5', 'M2', 'I5']
    additional_features = ['I1', 'I2', 'I3', 'I4', 'E1', 'E4', 'M1', 'M3', 'M4']
    selected_features = high_correlation_features + additional_features

    # Filter to available features
    available_features = [f for f in selected_features if f in train_df.columns]
    print(f"Using {len(available_features)} carefully selected features: {available_features}")

    # 1. Improved train/validation/test split with better proportions
    # Use more recent data for validation and test to reduce distribution shift
    n_total = len(train_df)
    train_size = int(0.8 * n_total)  # Use more training data
    val_size = int(0.1 * n_total)    # Smaller validation set

    train_data = train_df.iloc[:train_size].copy()
    val_data = train_df.iloc[train_size:train_size+val_size].copy()
    test_data = train_df.iloc[train_size+val_size:].copy()

    print(f"Improved split:")
    print(f"Train: {len(train_data)} samples ({len(train_data)/n_total:.1%})")
    print(f"Val: {len(val_data)} samples ({len(val_data)/n_total:.1%})")
    print(f"Test: {len(test_data)} samples ({len(test_data)/n_total:.1%})")

    # 2. Advanced preprocessing
    def advanced_preprocessing(X_train, X_val, X_test):
        """Advanced preprocessing with better missing value handling"""

        # Step 1: Forward fill within each feature
        X_train_filled = X_train.ffill().bfill()
        X_val_filled = X_val.ffill().bfill()
        X_test_filled = X_test.ffill().bfill()

        # Step 2: Fill remaining NaNs with column medians from training data
        for col in X_train_filled.columns:
            train_median = X_train_filled[col].median()
            if pd.isna(train_median):
                train_median = 0
            X_train_filled[col] = X_train_filled[col].fillna(train_median)
            X_val_filled[col] = X_val_filled[col].fillna(train_median)
            X_test_filled[col] = X_test_filled[col].fillna(train_median)

        # Step 3: Robust scaling (less sensitive to outliers)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_filled)
        X_val_scaled = scaler.transform(X_val_filled)
        X_test_scaled = scaler.transform(X_test_filled)

        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    # Prepare data
    X_train = train_data[available_features]
    y_train = train_data['risk_free_rate']
    X_val = val_data[available_features]
    y_val = val_data['risk_free_rate']
    X_test = test_data[available_features]
    y_test = test_data['risk_free_rate']

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = advanced_preprocessing(X_train, X_val, X_test)

    print(f"Data shapes after preprocessing:")
    print(f"X_train: {X_train_scaled.shape}")
    print(f"X_val: {X_val_scaled.shape}")
    print(f"X_test: {X_test_scaled.shape}")

    # 3. Models with strong regularization to prevent overfitting
    models = {
        'Ridge_Strong': Ridge(alpha=10.0, random_state=42),
        'Ridge_Very_Strong': Ridge(alpha=100.0, random_state=42),
        'Lasso_Strong': Lasso(alpha=1e-5, random_state=42, max_iter=2000),
        'RandomForest_Constrained': RandomForestRegressor(
            n_estimators=50,    # Fewer trees
            max_depth=5,        # Shallow trees
            min_samples_split=20,  # Require more samples to split
            min_samples_leaf=10,   # Require more samples in leaf
            max_features=0.5,      # Use only half the features
            random_state=42,
            n_jobs=-1
        )
    }

    print(f"\nTraining {len(models)} models with strong regularization...")

    results = {}
    for name, model in models.items():
        print(f"\n{name}:")

        # Train model
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)

        print(f"  Train R²: {train_r2:.6f}, Val R²: {val_r2:.6f}, Test R²: {test_r2:.6f}")
        print(f"  Train MSE: {train_mse:.2e}, Val MSE: {val_mse:.2e}, Test MSE: {test_mse:.2e}")

        # Check for overfitting
        overfitting = train_r2 - val_r2
        print(f"  Overfitting gap: {overfitting:.3f}")

        results[name] = {
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'overfitting_gap': overfitting,
            'model': model
        }

    # 4. Cross-validation for more robust evaluation
    print(f"\nCROSS-VALIDATION ANALYSIS")
    print("-" * 30)

    # Use TimeSeriesSplit for proper temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Test best performing model from above
    best_model_name = max(results.keys(), key=lambda x: results[x]['val_r2'])
    best_model = results[best_model_name]['model']

    cv_scores = []
    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        temp_model = type(best_model)(**best_model.get_params())
        temp_model.fit(X_cv_train, y_cv_train)
        cv_pred = temp_model.predict(X_cv_val)
        cv_score = r2_score(y_cv_val, cv_pred)
        cv_scores.append(cv_score)

    print(f"Time Series CV R² scores: {cv_scores}")
    print(f"Mean CV R²: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")

    # 5. Final results and model selection
    print(f"\nFINAL RESULTS COMPARISON")
    print("=" * 40)

    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train_R2': [r['train_r2'] for r in results.values()],
        'Val_R2': [r['val_r2'] for r in results.values()],
        'Test_R2': [r['test_r2'] for r in results.values()],
        'Overfitting_Gap': [r['overfitting_gap'] for r in results.values()],
        'Val_MSE': [r['val_mse'] for r in results.values()]
    })

    print(results_df.round(6))

    # Select best model (highest validation R² with reasonable overfitting)
    # Filter out models with excessive overfitting (gap > 0.5)
    reasonable_models = results_df[results_df['Overfitting_Gap'] < 0.5]
    if len(reasonable_models) > 0:
        best_idx = reasonable_models['Val_R2'].idxmax()
        best_model_final = reasonable_models.loc[best_idx, 'Model']
    else:
        # If all models overfit, choose the one with smallest overfitting gap
        best_idx = results_df['Overfitting_Gap'].idxmin()
        best_model_final = results_df.loc[best_idx, 'Model']

    best_val_r2 = results_df.loc[best_idx, 'Val_R2']
    best_test_r2 = results_df.loc[best_idx, 'Test_R2']

    print(f"\nBEST MODEL: {best_model_final}")
    print(f"Validation R²: {best_val_r2:.6f}")
    print(f"Test R²: {best_test_r2:.6f}")
    print(f"Cross-validation R²: {np.mean(cv_scores):.6f}")

    # Success criteria
    print(f"\nSUCCESS EVALUATION")
    print("-" * 20)
    if best_val_r2 > 0:
        print("SUCCESS: Achieved positive validation R²")
    else:
        print("ISSUE: Still negative validation R² - further work needed")

    if best_val_r2 > 0.1:
        print("EXCELLENT: Validation R² > 0.1")
    elif best_val_r2 > 0.05:
        print("GOOD: Validation R² > 0.05")
    elif best_val_r2 > 0.01:
        print("FAIR: Validation R² > 0.01")

    return {
        'best_model': best_model_final,
        'val_r2': best_val_r2,
        'test_r2': best_test_r2,
        'cv_r2_mean': np.mean(cv_scores),
        'cv_r2_std': np.std(cv_scores),
        'all_results': results_df
    }

if __name__ == "__main__":
    results = main()
    print(f"\nFinal summary: {results['best_model']} achieved validation R² of {results['val_r2']:.6f}")