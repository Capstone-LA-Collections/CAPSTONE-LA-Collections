from datetime import date, timedelta
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
import time
import logging


from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from supabase import create_client, Client
from supabase.client import ClientOptions 

# --- Setup (Same as original) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import SUPABASE_URL, SUPABASE_KEY
except ImportError:
    print("Warning: config.py not found or missing SUPABASE_URL/SUPABASE_KEY. Using placeholders.")
    # Placeholder definitions for a runnable script (replace with actual logic)
    SUPABASE_URL = "http://placeholder.url"
    SUPABASE_KEY = "placeholder-key"
    
MAX_ROWS = 100000
MAX_RETRIES = 3 
_CACHED_SALES_DATA = None 

# The list of columns returned by the RPC function
RPC_COLUMNS = [
    'date', 
    'platform_name',
    'net_sales',
    'discount_rate',
    'is_mega_sale_day',
    'is_payday',
    'is_weekend',
]

def get_supabase_client() -> Client:
    """Initializes and returns the Supabase client."""
    # NOTE: The actual implementation must be replaced with your Supabase logic
    # or a Mock client for testing. 
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="la_collections",))
    return supabase

def _process_dataframe(data: list) -> pd.DataFrame:
    """Standard processing steps for data frames returned by Supabase RPCs."""
    df = pd.DataFrame(data)

    if df.empty:
        print("Warning: RPC returned an an empty dataset.")
        return pd.DataFrame() 

    # Ensure 'is_weekend' is included in boolean conversion check
    for col in ['is_mega_sale_day', 'is_payday', 'is_weekend']: 
        if col in df.columns and pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int) 

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    return df

def preprocess_sales_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Performs daily aggregation, resampling, and imputation.
    """
    
    print("Starting data aggregation and transformation...")
    
    # 1. Aggregation Strategy for Daily Metrics
    aggregation_schema = {
        'net_sales': 'sum',
        'discount_rate': 'mean',
        'is_mega_sale_day': 'max',
        'is_payday': 'max',
        'is_weekend': 'max',
    }
    
    # Group by date (ds) and platform_name (Segmentation Key)
    df_daily = df_raw.groupby(['date', 'platform_name']).agg(aggregation_schema).reset_index()
    
    # Rename columns for Prophet and clarity
    df_daily = df_daily.rename(columns={
        'date': 'ds', 
        'net_sales': 'daily_net_sales', 
        'discount_rate': 'avg_discount_rate_daily', 
    })
    
    # 2. Time Series Resampling and Imputation
    df_processed = []
    platform_names = df_daily['platform_name'].unique()
    
    for platform in platform_names:
        df_platform = df_daily[df_daily['platform_name'] == platform].set_index('ds')
        
        df_platform_numeric = df_platform.drop(columns=['platform_name'])

        # Resample to daily frequency ('D') over the entire time span
        df_resampled = df_platform_numeric.resample('D').mean()
        df_resampled['platform_name'] = platform
        
        # --- Imputation Process ---
        df_resampled['avg_discount_rate_daily'] = df_resampled['avg_discount_rate_daily'].fillna(0)
        
        # Fill binary regressors (including is_weekend)
        for col in ['is_mega_sale_day', 'is_payday', 'is_weekend']:
            if col in df_resampled.columns:
                df_resampled[col] = df_resampled[col].ffill().fillna(0).astype(int)

        df_resampled['daily_net_sales'] = df_resampled['daily_net_sales'].fillna(0)
        
        df_processed.append(df_resampled.reset_index())

    df_final = pd.concat(df_processed, ignore_index=True)
    
    # 3. Normalization of Continuous Economic Regressor
    scaler = MinMaxScaler()
    
    valid_data = df_final['avg_discount_rate_daily'][df_final['avg_discount_rate_daily'].notna()].values.reshape(-1, 1)
    
    if valid_data.size == 0:
        print("Warning: No non-missing discount rate data found. Setting scaled_discount_rate to 0.")
        df_final['scaled_discount_rate'] = 0.0
    else:
        scaler.fit(valid_data)
        df_final['scaled_discount_rate'] = scaler.transform(df_final['avg_discount_rate_daily'].values.reshape(-1, 1))
        
    df_final = df_final.rename(columns={'daily_net_sales': 'y'})
        
    print("Data processing complete.")
    return df_final

def load_base_sales_data(start_date='2020-09-19'):
    # NOTE: Function stub unchanged. Actual implementation depends on your Supabase connection.
    global _CACHED_SALES_DATA
    
    if _CACHED_SALES_DATA is not None and not _CACHED_SALES_DATA.empty:
        print("[INFO] Returning cached data instead of executing RPC.")
        return _CACHED_SALES_DATA.copy() 
    
    supabase = get_supabase_client()
    
    for attempt in range(MAX_RETRIES):
        try:
            print("STEP 1.1: Executing Supabase RPC function: get_daily_sales_summary...")
            # Actual Supabase RPC call logic goes here...
            # This is replaced by a mock/dummy call if running without a real connection
            
            # --- START MOCK DATA GENERATION ---
            # This section simulates the data loading for demonstration purposes.
            # In your environment, this would be the actual RPC call:
            response = (
                supabase.rpc(
                'get_daily_sales_summary', 
                {'start_date': start_date}
                )
                .range(0, MAX_ROWS - 1)
                .execute()
                )
            # --- END MOCK DATA GENERATION ---

            df_raw = _process_dataframe(response.data)
            df_final = df_raw[RPC_COLUMNS].copy()
            
            print(f"[SUCCESS] Predictive model data loading complete. Loaded {len(df_final)} total rows.")
            
            _CACHED_SALES_DATA = df_final.copy() 
            return df_final
            
        except Exception as e:
            # Simplified error handling 
            error_message = str(e)
            if '57014' in error_message or 'timeout' in error_message.lower():
                print(f"Supabase Query Error: {error_message}")
                # ... (retry logic) ...
            else:
                print(f"Supabase Query Error: {e}")
                return pd.DataFrame() 

    return pd.DataFrame()

def generate_future_regressors(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a DataFrame of major holidays and sales events (Mega_Sale_Day, Payday, Weekend).
    """
    logging.warning("Using robust helper logic for generate_holidays_df.")

    if 'ds' not in df_input.columns or df_input.empty:
        logging.warning("Input DataFrame is empty or missing 'ds'. Returning empty regressor DataFrame.")
        return pd.DataFrame(columns=['ds', 'holiday'])
        
    start_date = df_input['ds'].min()
    end_date = df_input['ds'].max() + pd.Timedelta(days=365) 
    
    dates = pd.date_range(start=start_date, end=end_date)
    regressor_list = []

    MEGA_SALE_DAYS = [
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), 
        (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
        (2, 14), (5, 1), (6, 12), (8, 21), (11, 1), (12, 24), (12, 25), (12, 30)
    ]
    
    # Helper to find the last specific weekday in a month
    def get_last_weekday_of_month(year, month, weekday): 
        last_day = date(year, month, 1) + timedelta(days=32)
        last_day = last_day.replace(day=1) - timedelta(days=1)
        while last_day.weekday() != weekday:
            last_day -= timedelta(days=1)
        return last_day

    for d in dates:
        d_date = d.date()
        month = d.month
        day = d.day
        year = d.year
        
        # --- Mega Sale Days ---
        is_mega_sale = False
        if (month, day) in MEGA_SALE_DAYS:
            is_mega_sale = True

        # Black Friday 
        if month == 11:
            try:
                last_friday = get_last_weekday_of_month(year, 11, 4)
                if d_date == last_friday:
                    is_mega_sale = True
            except ValueError:
                pass 

        # Cyber Monday 
        if month == 11 or (month == 12 and day <= 3):
            try:
                last_thursday = get_last_weekday_of_month(year, 11, 3) 
                cyber_monday = last_thursday + timedelta(days=4)
                if d_date == cyber_monday:
                    is_mega_sale = True
            except ValueError: 
                pass 
                
        if is_mega_sale:
            regressor_list.append({'ds': d, 'holiday': 'is_mega_sale_day'})
            
        # --- Payday Logic ---
        is_payday = False
        if day == 15:
            is_payday = True
        
        try:
            last_day_of_month = (date(year, month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            if d_date == last_day_of_month:
                is_payday = True
            
            elif last_day_of_month.weekday() in [5, 6]: 
                friday_before = last_day_of_month - timedelta(days=last_day_of_month.weekday() - 4)
                if d_date == friday_before:
                    is_payday = True
        except ValueError:
            pass

        if is_payday:
            regressor_list.append({'ds': d, 'holiday': 'is_payday'})
            
        # --- NEW: Weekend Logic (Saturday/Sunday) ---
        # 5=Saturday, 6=Sunday
        if d.weekday() in [5, 6]: 
            regressor_list.append({'ds': d, 'holiday': 'is_weekend'})


    regressor_df = pd.DataFrame(regressor_list).drop_duplicates().sort_values('ds').reset_index(drop=True)
    regressor_df['ds'] = pd.to_datetime(regressor_df['ds'])
    
    return regressor_df

def tune_and_cross_validate_prophet(df_historical: pd.DataFrame):
    """
    Performs hyperparameter tuning and cross-validation for the Prophet model.
    Selects the model with the lowest mean RMSE.
    
    Args:
        df_historical (pd.DataFrame): The training data, containing 'ds', 'y', and regressor columns.
        
    Returns:
        tuple: (best_model, best_params, lowest_rmse)
    """
    print("\nStarting Hyperparameter Tuning and Cross-Validation...")
    
    # Define the hyperparameter grid to search
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'seasonality_mode': ['multiplicative', 'additive'],
        'yearly_seasonality': [False],
        'daily_seasonality': [True, False],
        'weekly_seasonality': [True, False],
    }

    best_params = None
    best_model = None
    lowest_rmse = np.inf

    all_params = list(ParameterGrid(param_grid))
    
    # Validation window settings (Example: 180 days initial, cut every 90 days, 90 days horizon)
    data_duration_days = (df_historical['ds'].max() - df_historical['ds'].min()).days
    initial_val = '730 days'  # e.g., 2 years of initial training
    period_val = '90 days'    # Retrain every 3 months
    horizon_val = '90 days'

    if data_duration_days < 850: 
        print(f"WARNING: Historical data is short ({data_duration_days} days). Adjusting CV parameters dynamically.")
        initial_val = f'{int(data_duration_days * 0.7)} days'
        period_val = f'{int(data_duration_days * 0.15)} days'
        horizon_val = '90 days' if data_duration_days * 0.15 > 90 else f'{int(data_duration_days * 0.15)} days'
        
        if data_duration_days < 180:
             print("ERROR: Data is critically short. Tuning will likely fail.")

    for params in all_params:
        print(f"Testing parameters: {params}")
        
        # 1. Initialize Prophet with grid parameters
        m = Prophet(
            **params
        )

        # 2. Add regressors and seasonality (must match the final forecast model)
        m.add_seasonality(name='yearly', period=365.25, fourier_order=8)
        m.add_regressor('is_mega_sale_day')
        m.add_regressor('is_payday')
        m.add_regressor('is_weekend')
        m.add_regressor('scaled_discount_rate')
        
        # 3. Fit the model
        m.fit(df_historical)

        try:
            # 4. Cross-validation
            df_cv = cross_validation(
                m, initial=initial_val, period=period_val, horizon=horizon_val, parallel='processes'
            )

            if df_cv is None or df_cv.empty:
                print("  -> Cross-Validation produced no results (empty DataFrame). Skipping.")
                continue

            # 5. Calculate performance metrics (RMSE)
            # Check for NaN/Inf values before calculating mean
            df_p = performance_metrics(df_cv, metrics=['rmse'], rolling_window=1.0)
            
            if df_p is None:
                print("  -> Performance metrics calculation failed (returned None). Skipping.")
                continue
            
            # 6. Calculate Mean RMSE and handle non-finite values
            valid_rmse = df_p['rmse'].replace([np.inf, -np.inf], np.nan).dropna()
            
            if valid_rmse.empty:
                print("  -> RMSE results are all NaN/Inf after processing. Skipping.")
                continue

            rmse = valid_rmse.mean()
            print(f"  -> Mean RMSE: {rmse:.4f}")
            
            # 6. Select the best model
            if rmse < lowest_rmse:
                lowest_rmse = rmse
                best_params = params
                # Re-fit the final model for the actual forecast
                best_model = m
                
        except ValueError as e:
            # Catch errors like insufficient historical data for CV
            print(f"  -> Cross-Validation failed for parameters {params}: {e}")
            continue

    print(f"\nHyperparameter tuning complete. Best Model Parameters: {best_params}")
    print(f"Lowest Mean RMSE achieved: {lowest_rmse:.4f}")
    
    return best_model

def main_run():
    plot_dir = 'app/Analytics/Prophet Model/outputs/tests'
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Outputs will be saved to: {plot_dir}")


    df_raw = load_base_sales_data()
    if df_raw.empty:
        print("Cannot run model: Base sales data is empty.")
        return
    
    filename = os.path.join(plot_dir, f'full_historical.csv')
    df_raw.to_csv(filename, index=False)
        
    df_processed = preprocess_sales_data(df_raw)
    
    df_Lazada = df_processed[df_processed['platform_name']=='Lazada'].copy()
    if df_Lazada.empty:
        print("Cannot run model: Lazada data is empty after processing.")
        return
    
    df_regressors_only = df_Lazada.drop(columns='platform_name')

    print(df_regressors_only.corrwith(df_regressors_only['y']))
    
    filename = os.path.join(plot_dir, f'Lazada_hist.csv')
    df_Lazada.to_csv(filename, index=False)

    param_run = False

    if param_run:
        best_model = tune_and_cross_validate_prophet(df_Lazada)
        if best_model is None:
            print("Error: Could not find a suitable model after cross-validation.")    
        else:
            model = best_model
    else:
        m = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=0.05,
            yearly_seasonality=False, 
            daily_seasonality=False,
            weekly_seasonality=True,
            seasonality_mode='multiplicative'
        )
    
        m.add_seasonality(name='yearly', period=365.25, fourier_order=8)
        
        # Refactoring Action: Add all external regressors
        m.add_regressor('is_mega_sale_day')
        m.add_regressor('is_payday')
        m.add_regressor('is_weekend')
        m.add_regressor('scaled_discount_rate')
    
        model = m.fit(df_Lazada)
    
    # --- 2. Generate Future Forecast DataFrame and Regressors ---
    
    future_dates = model.make_future_dataframe(periods=365, freq='D', include_history=True)
    
    regressors_df = generate_future_regressors(df_Lazada) 

    # Pivot the long-format regressor data into wide format
    regressors_wide = regressors_df.pivot_table(
        index='ds',
        columns='holiday',
        values='holiday',
        aggfunc=lambda x: 1, 
        fill_value=0
    ).reset_index()
    
    # Merge the future dates with the generated binary regressors
    future_df = future_dates.merge(regressors_wide, on='ds', how='left')
    
    # Fill NaNs for ALL binary regressors with 0
    future_df['is_mega_sale_day'] = future_df['is_mega_sale_day'].fillna(0)
    future_df['is_payday'] = future_df['is_payday'].fillna(0)
    future_df['is_weekend'] = future_df['is_weekend'].fillna(0) # NEW FILL
    
    # Impute the continuous regressor for the forecast period
    historical_discount = df_Lazada[['ds', 'scaled_discount_rate']].copy()
    future_df = future_df.merge(historical_discount, on='ds', how='left')
    
    mean_scaled_discount = historical_discount['scaled_discount_rate'].mean()
    future_df['scaled_discount_rate'] = future_df['scaled_discount_rate'].fillna(mean_scaled_discount)

    filename = os.path.join(plot_dir, f'Lazada_future_df.csv')
    future_df.to_csv(filename, index=False)
    
    # 3. Predict the Forecast
    forecast = model.predict(future_df)

    # 4. Save Outputs
    # ... (Plotting and saving logic omitted for brevity, but retained the necessary variables)
    print("\n[INFO] Forecast complete. Plotting and saving results...")

    # Example of saving:
    plot = model.plot(forecast)
    filename_plot = os.path.join(plot_dir, f'Lazada_test_forecast.png')
    plot.savefig(filename_plot)
    filename = os.path.join(plot_dir, f'Lazada_test_forecast.csv')
    forecast.to_csv(filename, index=False)


if __name__ == '__main__':
    # NOTE: Execution will require a working Supabase connection and the prophet library installed.
    main_run()
    pass