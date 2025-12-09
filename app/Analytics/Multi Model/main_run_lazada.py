import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import our custom modules
# (Assumes data_loader.py is the file from the previous conversation)
from load_data import load_base_sales_data 
import utils
from stat_models import TimeSeriesTrainer
import hybrid_xgboost

# Configuration
START_DATE = '2022-10-05'
END_DATE = '2025-10-31'
PLATFORM = 'lazada'
FORECAST_YEARS = 2
FORECAST_DAYS = 365 * FORECAST_YEARS
OUTPUT_DIR = 'app/Analytics/Multi Model/forecast_outputs'

MODELS_TO_RUN = [
    'SNaive', 
    'ARIMA', 
    'ARIMAX', 
    'SARIMA', 
    'SARIMAX', 
    'Holt-Winters', 
    'Prophet'
]

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("--- Step 1: Loading Data ---")
    df = load_base_sales_data(start_date=START_DATE, platform=PLATFORM)
    if df.empty:
        print("No data found. Exiting.")
        return

    # --- Step 1.5: Enriching Historical Data ---
    print("--- Step 1.5: Enriching Historical Data ---")
    
    # Apply feature logic directly to historical df
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Feature Engineering (as defined in previous steps)
    df['is_weekend'] = df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    
    def is_mega_sale(d):
        MEGA_SALE_DAYS = [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (2,14), (5,1), (12,25)]
        if (d.month, d.day) in MEGA_SALE_DAYS: return 1
        return 0
        
    df['is_mega_sale_day'] = df['ds'].apply(is_mega_sale)
    df['is_payday'] = df['ds'].apply(lambda x: 1 if x.day in [15, 30, 31] else 0)

    # --- Step 2: Future Features ---
    print(f"--- Step 2: Preparing Future Features ({FORECAST_DAYS} days) ---")
    future_df = utils.generate_future_features(df, FORECAST_DAYS)
    
    # Initialize Trainer (creates an internal date-indexed dataframe)
    trainer = TimeSeriesTrainer(df)
    
    # Use the date-indexed version of y for metric calculation
    y_true_aligned = trainer.df['y']
    
    metrics_list = []
    
    # 3. Loop through Models
    print("--- Step 3: Training Models & Hybridization ---")
    
    for model_name in MODELS_TO_RUN:
        print(f"\nProcessing Model: {model_name}")
        
        try:
            # A. Train & Predict
            future_exog_cols = ['aov', 'promo_intensity', 'is_mega_sale_day', 'is_payday', 'is_weekend']
            # future_exog_cols = ['aov', 'is_mega_sale_day', 'is_payday', 'is_weekend']
            # future_exog_cols = ['promo_intensity', 'is_mega_sale_day', 'is_payday', 'is_weekend']
            preds, fitted_values = trainer.train_predict(
                model_name, 
                forecast_horizon=FORECAST_DAYS,
                future_exog=future_df[future_exog_cols],
                model_params=None,
                platform=PLATFORM
            )
            
            # B. Base Metrics
            metric_res = utils.calculate_metrics(y_true_aligned, fitted_values, model_name)
            
            # C. Residuals (Cleaned)
            residuals = y_true_aligned - fitted_values
            residuals = residuals.fillna(0).replace([np.inf, -np.inf], 0)
            
            resid_df = pd.DataFrame({
                'ds': y_true_aligned.index, 
                'residuals': residuals
            })
            
            # D. Hybrid (The New Iterative Loop)
            print("   Training Hybrid XGBoost (Iterative)...")
            # RECEIVE NEW RETURN VALUE: fitted_residual_preds_series
            hybrid_forecast_values, fitted_residual_preds_series = hybrid_xgboost.run_hybrid_forecast(
                base_forecast=preds, 
                historical_residuals_df=resid_df, 
                future_df=future_df,
                platform_name=PLATFORM
            )
            
            # **NEW: Calculate Hybrid In-Sample Fitted Values for Metrics**
            
            # 1. Calculate Hybrid Fitted Values
            # The addition is index-aligned: NaNs from the base model are preserved, 
            # and the first day's XGBoost prediction (missing due to resid_lag_1) is also NaN/ignored.
            fitted_hybrid_values = fitted_values + fitted_residual_preds_series

            # 2. Calculate Hybrid Metrics 
            hybrid_metric_res = utils.calculate_metrics(y_true_aligned, fitted_hybrid_values, model_name)
            
            # E. Save Metrics (Combined)
            base_metric_res_dict = {f'Base {k}': v for k, v in metric_res.items() if k != 'Model'}
            hybrid_metric_res_dict = {f'Hybrid {k}': v for k, v in hybrid_metric_res.items() if k != 'Model'}
            
            final_metrics_dict = {'Model': model_name, **base_metric_res_dict, **hybrid_metric_res_dict}
            metrics_list.append(final_metrics_dict)
            
            print(f"   Base Metrics: {metric_res}")
            print(f"   Hybrid Metrics (Base+XGBoost): {hybrid_metric_res}")

            # F. Save
            final_future_df = future_df.copy()
            final_future_df['base_forecast'] = preds.values if hasattr(preds, 'values') else preds
            final_future_df['hybrid_forecast'] = hybrid_forecast_values
            final_future_df['model'] = model_name
            
            csv_name = f"{OUTPUT_DIR}/Lazada_{model_name}_3yr_forecast.csv"
            final_future_df.to_csv(csv_name, index=False)
            
            # G. Plot
            plt.figure(figsize=(12, 6))
            #recent_history = df.tail(548) 
            #plt.plot(recent_history['ds'], recent_history['y'], label='History (Last 6mo)', color='gray', alpha=0.5)
            plt.plot(df['ds'], df['y'], label='History', color='gray', alpha=0.5)
            plt.plot(final_future_df['ds'], final_future_df['hybrid_forecast'], label='Hybrid Forecast', color='red')
            plt.plot(final_future_df['ds'], final_future_df['base_forecast'], label='Base Forecast', linestyle='--')
            plt.title(f"{model_name} + Iterative XGBoost")
            plt.legend()
            plt.savefig(f"{OUTPUT_DIR}/Lazada_{model_name}_plot.png")
            plt.close()
            
        except Exception as e:
            print(f"   ERROR processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    # 4. Save Summary
    print("\n--- Step 4: Saving Final Metrics ---")
    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df)
    metrics_df.to_csv(f"{OUTPUT_DIR}/Lazada_model_performance_metrics.csv", index=False)
    print(f"Done. Outputs saved to /{OUTPUT_DIR}")

if __name__ == "__main__":
    main()