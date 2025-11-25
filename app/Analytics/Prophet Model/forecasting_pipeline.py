import pandas as pd
import numpy as np 
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
# We need to import Matplotlib for custom plotting
import matplotlib.pyplot as plt
from prophet.plot import plot_cross_validation_metric
import logging
import warnings
import os
import sys

# --- 1. CLEANUP & SETUP ---

# Suppress cmdstanpy (Prophet's backend) INFO messages, which removes the progress bar
# Set level to ERROR to entirely suppress progress bars and INFO/WARNING messages.
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# Suppress the specific FutureWarning related to Series.view in Prophet's plotting
# This addresses the warnings seen in the original output.
warnings.filterwarnings('ignore', category=FutureWarning)

OUTPUT_DIR = 'app/Analytics/Prophet Model/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. CORE SEGMENT MODELING FUNCTION ---

# CRITICAL CHANGE: Accept the definitive list of regressors
def _run_single_platform_pipeline(model: Prophet, df_input: pd.DataFrame, platform_name: str, regressor_cols: list):
    print(f"\n--- Processing {platform_name} ---")
    if df_input.empty:
        print(f"[ERROR] Dataframe for {platform_name} is empty. Cannot proceed.")
        return False

    try:
        # We are using the definitive list of regressors
        if not regressor_cols and not model.extra_regressors:
            print(f"[NOTE] Model for {platform_name} has no explicit regressors.")
        else:
            # We use the list provided by the modeler (which is consistent with model.add_regressor calls)
            print(f"[INFO] Using {len(regressor_cols)} Regressors for CV/Forecast.")


        # --- Cross-Validation (CV) ---
        print(f"Running cross-validation for {platform_name}...")
        
        horizon_days = 365
        horizon = f'{horizon_days} days' 
        required_periods = 7 # To achieve 8 total cuts

        total_days = len(df_input)
            
        # Initial: Use 50% of the data, min 365 days
        initial_days = max(365, total_days // 2)
        initial = f'{initial_days} days'
            
        # Calculate remaining days for CV (after initial and horizon)
        remaining_cv_days = total_days - initial_days - horizon_days
            
        if remaining_cv_days <= 0:
            # If not enough data remains for even one cut, use a minimal period
            period_days = 7
            print(f"  [WARNING] {platform_name} has insufficient data for CV. Total days: {total_days}")
        else:
            if platform_name == 'Shopee':
                period_days = 81
            else:
                period_days = 27   

        period = f'{period_days} days'
                 
        print(f"  - {platform_name} CV settings: initial={initial}, period={period}, horizon={horizon}")

        # Drop non-Prophet columns (only keep 'ds', 'y', and the actual regressors)
        cols_to_keep_cv = ['ds', 'y'] + regressor_cols
            
        # Ensure all required columns exist before subsetting
        missing_cols = [col for col in cols_to_keep_cv if col not in df_input.columns]
        if missing_cols:
             print(f"[FATAL] Cross-validation failed: Missing required columns in data: {missing_cols}")
             raise ValueError(f"Missing required columns in data: {missing_cols}")

            # ==================================================================
            # CRITICAL FIX (Change 1):
            # Removed the 'df_segment_cv' argument. This version of Prophet
            # uses the data already attached to the 'model' object for CV.
            # The function now receives (model, initial, period, horizon).
            # ==================================================================
        cv_results = cross_validation(model, initial=initial, period=period, horizon=horizon, parallel=None)
        print(f"Cross-validation completed for {platform_name}. Found {len(cv_results)} CV results.")


        # --- Performance Metrics & Plotting (if CV ran) ---
        if cv_results is not None and not cv_results.empty:
            df_p = performance_metrics(cv_results, metrics=['rmse', 'mae', 'mape'])

            mean_mae = df_p['mae'].mean()
            print(f"  - {platform_name} CV Mean MAE: {mean_mae:,.2f} ({len(df_p)} cuts)")
            
            metrics_path = os.path.join(OUTPUT_DIR, f"{platform_name}_cv_metrics.csv")
            df_p.to_csv(metrics_path, index=False)
            print(f"Performance metrics saved to: {metrics_path}")

            fig_cv = plot_cross_validation_metric(cv_results, metric='rmse')
            fig_cv.suptitle(f'Cross-Validation RMSE for {platform_name}', y=1.02)
            fig_cv.tight_layout()
            cv_plot_path = os.path.join(OUTPUT_DIR, f"{platform_name}_cv_rmse_plot.png")
            fig_cv.savefig(cv_plot_path)
            plt.close(fig_cv)
            print(f"Cross-validation plot saved to: {cv_plot_path}")


        # --- Forecasting ---
        print(f"Generating future forecast for {platform_name}...")
        
        # ==================================================================
        # CRITICAL FIX (Change 2):
        # Removed the 'df=df_input[['ds']].copy()' argument.
        # This function signature uses the model's fitted history.
        # ==================================================================
        future = model.make_future_dataframe(
            periods=365, 
            include_history=True,
            freq='D'
        )

        # 2. Merge future dates with ONLY the required regressor columns from df_input
        # We need the regressors for both history and the 90-day future forecast period.
        cols_to_merge = ['ds'] + regressor_cols
        
        # Merge the future dates with the regressor values from the original full data
        # Note: df_input already contains the synthetic future regressor values (ds up to 2024-12-31)
        future = future[['ds']].merge(
            df_input[cols_to_merge], 
            on='ds', 
            how='left'
        )
        
        # ==================================================================
        # CRITICAL FIX: Fill NaN values in future regressors
        # The merge above creates NaNs for all future dates (e.g., 90 days).
        # We must impute these before prediction. Using 0.0 is consistent
        # with the imputation strategy in feature_engineer.py's safety net.
        future[regressor_cols] = future[regressor_cols].fillna(0.0)
        # ==================================================================
        
        # Generate the forecast
        forecast = model.predict(future)

        # Save forecast to CSV
        forecast_path = os.path.join(OUTPUT_DIR, f"{platform_name}_forecast_data.csv")
        forecast.to_csv(forecast_path, index=False)
        print(f"Forecast data saved to: {forecast_path}")

        # Plot the forecast
        fig = model.plot(forecast, xlabel="Date", ylabel="Daily Net Sales")
        
        # Generate and close component plot
        a = model.plot_components(forecast) 
        plt.close(a) 

        ax = fig.gca()
        ax.set_title(f"Sales Forecast for {platform_name}", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Net Sales")
        ax.legend()
        plt.grid(True, alpha=0.3)
        fig.savefig(os.path.join(OUTPUT_DIR, f"{platform_name}_forecast_plot.png"))
        plt.close(fig)
        print(f"Forecast plot saved to: {OUTPUT_DIR}/{platform_name}_forecast_plot.png")

    except Exception as e:
        # Provide a clearer error message for potential column mismatches
        print(f"[CRITICAL ERROR] Forecast failed for {platform_name}. The issue is likely with missing or incorrect regressor columns/data structure. Details: {e}")
        return False

    return True

# --- 3. EXPORTABLE WRAPPER FUNCTION ---

# CRITICAL CHANGE: Accept the definitive list of regressors
def run_cross_validation_and_forecast(segmented_models: dict, df_prophet_ready: pd.DataFrame, regressor_cols: list) -> bool:
    """
    Runs the full cross-validation and forecasting process using pre-fitted models.
    """
    print("STEP 4: Running Cross-Validation, Forecasting, and Saving Outputs...")
    
    overall_success = True
    
    for platform_name, model in segmented_models.items():
        # Get the specific segment data and rename target column to 'y'
        df_segment = df_prophet_ready[df_prophet_ready['platform_name'] == platform_name].copy()
        # Ensure 'daily_items_sold' is renamed to 'y' for consistency with internal function
        df_segment = df_segment.rename(columns={'daily_net_sales': 'y'})
        
        # Pass the regressor list down
        success = _run_single_platform_pipeline(model, df_segment, platform_name, regressor_cols)
        
        if not success:
            overall_success = False
            
    return overall_success