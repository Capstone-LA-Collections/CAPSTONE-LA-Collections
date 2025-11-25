import pandas as pd
import numpy as np
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import logging
import os

# --- ASSUMED IMPORTS ---
try:
    from load_data import load_base_sales_data
    from stat_models import TimeSeriesTrainer
    import hybrid_xgboost
    import utils 
except ImportError:
    print("Warning: Core project modules (data_loader, utils, stat_models, hybrid_xgboost) could not be imported.")

# --- CONFIGURATION ---
START_DATE = '2020-09-19'
PLATFORMS = ['lazada', 'shopee']
OUTPUT_DIR = 'app/Analytics/Multi Model/tuning_results'

# --- HYPERPARAMETER GRID ---
PARAM_GRID = {  
    'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
    'seasonality_mode': ['additive', 'multiplicative'],
    # NEW: Seasonality Toggles
    'daily_seasonality': [True, False],
    'weekly_seasonality': [True, False],
    'yearly_seasonality': [True, False]
}

def tune_prophet_for_platform(df_plat, platform_name):
    # --- Data Prep for Prophet CV (Same as previous step) ---
    df_prophet = df_plat[['ds', 'y']].copy()
    regressor_cols = ['aov', 'promo_intensity', 'is_mega_sale_day', 'is_payday', 'is_weekend', 'units_per_order']
    for col in regressor_cols:
        if col in df_plat.columns:
            df_prophet[col] = df_plat[col]
            
    # --- Generate All Combinations (Same as previous step) ---
    keys, values = zip(*PARAM_GRID.items())
    all_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    rmses = []
    
    print(f"\n{'='*40}")
    print(f"🚀 Starting Tuning for Platform: {platform_name.upper()}")
    print(f"Testing {len(all_params)} parameter combinations...")

    # --- Grid Search Loop (Same as previous step) ---
    for i, params in enumerate(all_params):
        try:
            m = Prophet(**params)
            for reg in regressor_cols:
                if reg in df_prophet.columns:
                    m.add_regressor(reg)

            m.fit(df_prophet)
            df_cv = cross_validation(m, initial='730 days', period='90 days', horizon='30 days', parallel=None)
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmse_val = df_p['rmse'].values[0]
            rmses.append(rmse_val)
            
            if i % 20 == 0:
                 print(f"   Checked {i}/{len(all_params)} combinations... (Current Best RMSE: {min(rmses) if rmses and min(rmses) != float('inf') else 'N/A'})")

        except Exception as e:
            print(f"!!! FAILURE !!! Combination {i}: {params}")
            print(f"Error: {e}") # <--- ADDED
            rmses.append(float('inf'))

    # --- Find Best Parameters & Save Results (Same as previous step) ---
    results_df = pd.DataFrame(all_params)
    results_df['rmse'] = rmses
    results_df = results_df.sort_values('rmse').reset_index(drop=True)
    
    best_params = results_df.iloc[0].to_dict()
    best_params.pop('rmse') 
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f'{platform_name}_tuning_log.csv'), index=False)
    
    return best_params

def print_best_params_summary(best_params_storage):
    """Prints the final summary of the best parameters found."""
    print("\n" + "="*50)
    print("📋 BEST PROPHET HYPERPARAMETER SUMMARY")
    print("="*50)
    for plat, params in best_params_storage.items():
        if params is not None:
            # Print parameters alphabetically for consistent viewing
            sorted_params = dict(sorted(params.items()))
            print(f"\nPlatform: {plat.upper()}")
            for k, v in sorted_params.items():
                print(f"  {k:<25}: {v}")
        else:
             print(f"\nPlatform: {plat.upper()}")
             print("  Tuning failed or no data found.")
    print("-" * 50)
    
def evaluate_hybrid_performance(df, best_params, platform_name):
    # --- Hybrid Evaluation Logic (Same as previous step) ---
    print(f"\n📊 EVALUATING HYBRID MODEL WITH OPTIMIZED PARAMETERS ({platform_name.upper()})")
    
    forecast_days = 30
    
    # Feature Engineering for Future
    future_exog = utils.generate_future_features(df, forecast_days)
    if 'units_per_order' in df.columns:
        recent_units = df['units_per_order'].tail(30).mean()
        future_exog['units_per_order'] = recent_units

    # Train Prophet Base with BEST PARAMS
    trainer = TimeSeriesTrainer(df)
    regressor_cols = ['aov', 'promo_intensity', 'is_mega_sale_day', 'is_payday', 'is_weekend', 'units_per_order']
    
    preds_base, fitted_values_base = trainer.train_predict(
        'Prophet', 
        forecast_horizon=forecast_days,
        future_exog=future_exog[[c for c in regressor_cols if c in future_exog.columns]],
        model_params=best_params
    )
    
    # Calculate Residuals
    y_true_aligned = trainer.df['y']
    residuals = y_true_aligned - fitted_values_base
    residuals = residuals.fillna(0).replace([np.inf, -np.inf], 0)
    resid_df = pd.DataFrame({'ds': y_true_aligned.index, 'residuals': residuals})
    
    # Train Hybrid Corrector (XGBoost)
    future_hybrid, fitted_residuals_xgboost = hybrid_xgboost.run_hybrid_forecast(
        base_forecast=preds_base,
        historical_residuals_df=resid_df,
        future_df=future_exog,
        platform_name=platform_name
    )
    
    # Calculate Final Hybrid Metrics
    fitted_values_hybrid = fitted_values_base + fitted_residuals_xgboost
    
    metrics = utils.calculate_metrics(y_true_aligned, fitted_values_hybrid, f"Hybrid_{platform_name}")
    
    return metrics

def run_tuning_pipeline():
    # Suppress logging
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    logging.getLogger('prophet').setLevel(logging.WARNING)

    platform_data_storage = {} # Store data for later evaluation
    best_params_storage = {}    # Store best params found
    
    # --- PHASE 1: TUNING AND PARAMETER STORAGE ---
    print("\n" + "="*50)
    print("PHASE 1: PROPHET HYPERPARAMETER TUNING (CV)")
    print("="*50)

    for platform in PLATFORMS:
        # 1. Load Data
        if platform == 'lazada':
            START_DATE = '2022-10-05'
        else:
            START_DATE
        try:
            df = load_base_sales_data(start_date=START_DATE, platform=platform)
        except Exception as e:
            print(f"Skipping {platform}: {e}")
            continue

        if df.empty: continue

        # 2. Preprocessing (Ensure Regressors exist)
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Add Regressors if missing
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        
        def is_mega_sale(d):
            MEGA_SALE_DAYS = [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (2,14), (5,1), (12,25)]
            if (d.month, d.day) in MEGA_SALE_DAYS: return 1
            return 0
        if 'is_mega_sale_day' not in df.columns:
            df['is_mega_sale_day'] = df['ds'].apply(is_mega_sale)
        if 'is_payday' not in df.columns:
            df['is_payday'] = df['ds'].apply(lambda x: 1 if x.day in [15, 30, 31] else 0)
        
        for col in ['aov', 'promo_intensity', 'units_per_order']:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(0)
        
        platform_data_storage[platform] = df

        # 3. Tune Prophet & Store Best Params
        best_params = tune_prophet_for_platform(df, platform)
        best_params_storage[platform] = best_params

    # --- PHASE 2: DISPLAY BEST PARAMETERS (NEW LOCATION) ---
    


    # --- PHASE 3: HYBRID MODEL EVALUATION ---
    print("\n" + "="*50)
    print("PHASE 3: HYBRID MODEL METRICS EVALUATION")
    print("="*50)
    final_results = []
    
    for platform in PLATFORMS:
        if platform not in platform_data_storage or best_params_storage.get(platform) is None:
            continue
            
        df = platform_data_storage[platform]
        best_params = best_params_storage[platform]
        
        # 4. Evaluate Hybrid Performance
        metrics = evaluate_hybrid_performance(df, best_params, platform)
        metrics['Platform'] = platform
        final_results.append(metrics)

    # Summary
    if final_results:
        print("\n" + "="*50)
        print("🏁 FINAL HYBRID METRICS SUMMARY")
        print("="*50)
        summary_df = pd.DataFrame(final_results)
        print(summary_df[['Platform', 'MAE', 'RMSE', 'SMAPE', 'R2']])
        summary_df.to_csv(os.path.join(OUTPUT_DIR, 'final_hybrid_metrics.csv'), index=False)
    
    print_best_params_summary(best_params_storage)

if __name__ == "__main__":
    run_tuning_pipeline()