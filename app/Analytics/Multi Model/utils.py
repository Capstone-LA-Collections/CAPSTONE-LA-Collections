import numpy as np
import pandas as pd
from datetime import date, timedelta
# R2_SCORE ADDED HERE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred, model_name):
    """Calculates MAE, RMSE, NRMSE, R2, SMAPE, and WMAPE."""
    
    # Define SMAPE function
    def smape(a, f):
        # We ensure a and f are non-negative for robustness, though historical data should be sales
        a = np.maximum(a, 0)
        f = np.maximum(f, 0)
        return 100/len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f) + 1e-10))

    # Align indices if they don't match
    common_idx = y_true.index.intersection(y_pred.index)
    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]

    # Drop rows where EITHER y_true or y_pred is NaN to ensure finite inputs
    combined = pd.DataFrame({'y_true': y_true_aligned, 'y_pred': y_pred_aligned}).dropna()
    
    # Check if any samples remain
    if combined.empty:
        return {
            "Model": model_name,
            "MAE": np.nan,
            "RMSE": np.nan,
            "NRMSE": np.nan,
            "R2": np.nan,  # R2 ADDED
            "SMAPE": np.nan,
            "WMAPE": np.nan,
        }
        
    y_true_final = combined['y_true']
    y_pred_final = combined['y_pred']

    # --- 1. MAE and RMSE ---
    mae = mean_absolute_error(y_true_final, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))

    # --- 2. NRMSE (Normalized RMSE) ---
    mean_y = y_true_final.mean()
    nrmse = (rmse / mean_y) if mean_y != 0 else np.nan
    
    # --- 3. R2 (R-squared) ---
    r2 = r2_score(y_true_final, y_pred_final)

    # --- 4. SMAPE ---
    smape_val = smape(y_true_final.values, y_pred_final.values)

    # --- 5. WMAPE (Weighted MAE) ---
    sum_abs_error = np.sum(np.abs(y_pred_final - y_true_final))
    sum_actuals = np.sum(y_true_final)
    
    wmape = (sum_abs_error / sum_actuals) * 100 if sum_actuals != 0 else np.nan
    
    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "NRMSE": round(nrmse, 4), 
        "R2": round(r2, 4),      # R2 ADDED
        "SMAPE": round(smape_val, 2),
        "WMAPE": round(wmape, 2)
    }

def generate_future_features(df_history, periods):
    """
    Generates future dates AND robust regressors (Mega Sale, Payday, Weekend).
    """
    last_date = df_history['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    future_df = pd.DataFrame({'ds': future_dates})
    
    # --- 1. Calendar Regressors (The 'Manual Test' Logic) ---
    MEGA_SALE_DAYS = [
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), 
        (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
        (2, 14), (5, 1), (6, 12), (8, 21), (11, 1), (12, 24), (12, 25), (12, 30)
    ]

    def is_mega_sale(d):
        if (d.month, d.day) in MEGA_SALE_DAYS: return 1
        return 0

    def is_payday(d):
        # Simple Payday Logic (15th and 30th/last day)
        if d.day == 15: return 1
        if d.day >= 28:
            next_day = d + timedelta(days=1)
            if next_day.month != d.month: return 1
        return 0

    future_df['is_mega_sale_day'] = future_df['ds'].apply(is_mega_sale)
    future_df['is_payday'] = future_df['ds'].apply(is_payday)
    future_df['is_weekend'] = future_df['ds'].dt.dayofweek.isin([5, 6]).astype(int)

    # --- 2. Continuous Regressors (Imputation) ---
    recent_aov = df_history['aov'].tail(30).mean()
    recent_promo = df_history['promo_intensity'].tail(30).mean()
    
    future_df['aov'] = recent_aov
    future_df['promo_intensity'] = recent_promo
    
    return future_df