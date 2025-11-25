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

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
# from prophet.utilities import set_log_level

# NOTE: Assuming these imports and configuration setup are correct in your environment
# If you are running this code standalone, ensure SUPABASE_URL and SUPABASE_KEY are defined.
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
    'ds', 
    'y',
    'total_orders',
    'total_units',
    'aov',
    'promo_intensity',
    'units_per_order',
]

start_date_str = date(2022, 10, 1).isoformat()  # '2022-10-01'
end_date_str = date(2025, 11, 1).isoformat()    # '2025-11-01'
platform_str = 'Lazada'

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

    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
    
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

def load_base_sales_data():
    # NOTE: Function stub unchanged. Actual implementation depends on your Supabase connection.
    global _CACHED_SALES_DATA
    
    if _CACHED_SALES_DATA is not None and not _CACHED_SALES_DATA.empty:
        print("[INFO] Returning cached data instead of executing RPC.")
        return _CACHED_SALES_DATA.copy() 
    
    supabase = get_supabase_client()
    
    for attempt in range(MAX_RETRIES):
        try:
            print("STEP 1.1: Executing Supabase RPC function: get_monthly_sales_metrics...")
            # Actual Supabase RPC call logic goes here...
            # This is replaced by a mock/dummy call if running without a real connection
            
            # --- START MOCK DATA GENERATION ---
            # This section simulates the data loading for demonstration purposes.
            # In your environment, this would be the actual RPC call:
            response = (
                supabase.rpc(
                'get_monthly_sales_metrics',
                {
                'start_date': start_date_str,
                'end_date': end_date_str,
                'p_platform_name': platform_str
                }
                )
                .execute()
                )
            # --- END MOCK DATA GENERATION ---

            df_raw = _process_dataframe(response.data)
            print("Columns available in df_raw:", df_raw.columns)
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

def create_xgb_features(data, residuals=None):
    """
    Creates time-series features and lagged residuals for XGBoost training.
    Assumes 'ds' is datetime and other regressors (is_mega_sale_day, etc.) already exist.
    """
    data['dayofweek'] = data['ds'].dt.dayofweek
    data['dayofyear'] = data['ds'].dt.dayofyear
    data['month'] = data['ds'].dt.month
    data['year'] = data['ds'].dt.year
    
    # Add Lagged Residuals if provided
    if residuals is not None:
        data['Residuals'] = residuals
        data['resid_lag_1'] = data['Residuals'].shift(1)
        # Note: We will dropna() after calling this function
    return data

def main_run():
    plot_dir = 'app/Analytics/Prophet Model/outputs/tests'
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Outputs will be saved to: {plot_dir}")

    # =========================================================================
    # 0. CONFIGURATION (from test.py)
    # =========================================================================
    FORECAST_DAYS = 365 # Match original manual_test.py
    # Define the features XGBoost will be trained on
    FEATURES = [
        'is_mega_sale_day', 
        'is_payday', 
        'is_weekend', 
        'dayofweek', # 'dayofweek' from create_xgb_features
        'dayofyear',
        'month',
        'year',
        'scaled_discount_rate', 
        'resid_lag_1'
    ]

    # =========================================================================
    # 1. DATA LOADING AND PREPARATION
    # =========================================================================
    print("1. Loading and Preparing Data...")
    
    df_Lazada = load_base_sales_data()
    if df_Lazada.empty:
        print("Cannot run model: Base sales data is empty.")
        return

    # Save the historical data (which test.py used as an input)
    filename = os.path.join(plot_dir, f'Lazada_hist.csv')
    df_Lazada.to_csv(filename, index=False)
    print(f"Saved historical data to {filename}")

    # Create the base df for Prophet (y and ds only)
    df_hist_features = df_Lazada.copy() # Keep all features for later
    prophet_df = df_hist_features[['ds', 'y']].sort_values(by='ds').reset_index(drop=True)

    # Define the historical training cutoff (using all but the last 60 days)
    cutoff_date = prophet_df['ds'].max() - pd.Timedelta(days=60)
    train_prophet = prophet_df[prophet_df['ds'] <= cutoff_date].copy()

    # =========================================================================
    # 2. PROPHET BASE MODEL TRAINING AND RESIDUAL CALCULATION
    # =========================================================================
    print("2. Training Prophet Model and Calculating Residuals...")
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
            yearly_seasonality=True, 
            daily_seasonality=False,
            weekly_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        # Refactoring Action: Add all external regressors
        m.add_regressor('is_mega_sale_day')
        m.add_regressor('is_payday')
        m.add_regressor('is_weekend')
        m.add_regressor('scaled_discount_rate')
    
        model = m.fit(df_Lazada)

    forecast_with_regressor = pd.DataFrame()
    forecast_with_regressor['ds'] = prophet_df['ds']

    regressors = generate_future_regressors(forecast_with_regressor)

    regressors_wide =regressors.pivot_table(
        index='ds',
        columns='holiday',
        values='holiday',
        aggfunc=lambda x: 1, 
        fill_value=0
    ).reset_index()
    
    # Merge the future dates with the generated binary regressors
    hist_df = forecast_with_regressor.merge(regressors_wide, on='ds', how='left')
    
    # Fill NaNs for ALL binary regressors with 0
    hist_df['is_mega_sale_day'] = hist_df['is_mega_sale_day'].fillna(0)
    hist_df['is_payday'] = hist_df['is_payday'].fillna(0)
    hist_df['is_weekend'] = hist_df['is_weekend'].fillna(0) # NEW FILL
    
    # Impute the continuous regressor for the forecast period
    historical_discount = df_Lazada[['ds', 'scaled_discount_rate']].copy()
    hist_df = hist_df.merge(historical_discount, on='ds', how='left')
    
    mean_scaled_discount = historical_discount['scaled_discount_rate'].mean()
    hist_df['scaled_discount_rate'] = hist_df['scaled_discount_rate'].fillna(mean_scaled_discount)

    # Predict on the entire historical dataset to get residuals
    full_forecast = model.predict(hist_df)
    prophet_df['Prophet_yhat'] = full_forecast['yhat']
    prophet_df['Residuals'] = prophet_df['y'] - prophet_df['Prophet_yhat']

    # =========================================================================
    # 3. XGBOOST TRAINING ON RESIDUALS
    # =========================================================================
    print("3. Training XGBoost Model on Historical Residuals...")

    # Merge residuals back into the feature-rich dataframe and create lag
    feature_df = df_hist_features.merge(prophet_df[['ds', 'Residuals', 'Prophet_yhat']], on='ds', how='left')
    
    # Use the helper function to create time features and lag
    feature_df = create_xgb_features(feature_df.copy(), residuals=feature_df['Residuals'])
    feature_df = feature_df.dropna() # Drop the first row(s) due to the lag feature
    
    filename_csv = os.path.join(plot_dir, f'Lazada_residuals.csv')
    feature_df.to_csv(filename_csv, index=False)

    # Define Train Data for XGBoost (aligned with Prophet training cutoff)
    X_train = feature_df[feature_df['ds'] <= cutoff_date][FEATURES]
    y_train = feature_df[feature_df['ds'] <= cutoff_date]['Residuals']

    # Train XGBoost (using params from test.py)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.3, 
                       random_state=42, tree_method='hist',
                       objective='reg:squarederror',max_depth=2, 
                       reg_alpha=0.1, reg_lambda=0.5,
                       early_stopping_rounds=25, eval_metric='rmse')

    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)])

    # =========================================================================
    # 3.5 HISTORICAL TEST SET EVALUATION (from test.py)
    # =========================================================================
    print("3.5 Evaluating Model Performance on Historical Test Set...")

    X_test = feature_df[feature_df['ds'] > cutoff_date][FEATURES]
    actual_test_values_aligned = feature_df[feature_df['ds'] > cutoff_date]['y'].values
    prophet_test_yhat_aligned = feature_df[feature_df['ds'] > cutoff_date]['Prophet_yhat'].values

    # Predict residuals on the test set
    xgb_test_pred_residuals = xgb.predict(X_test.astype(float))
    
    # Calculate Final Hybrid Forecast on Test Set
    final_hybrid_forecast_test = prophet_test_yhat_aligned + xgb_test_pred_residuals

    # Calculate RMSE
    prophet_rmse = mean_squared_error(actual_test_values_aligned, prophet_test_yhat_aligned, squared=False)
    final_rmse = mean_squared_error(actual_test_values_aligned, final_hybrid_forecast_test, squared=False)

    print("\n--- Model Performance on Historical Test Set ---")
    print(f"Prophet Base RMSE: {prophet_rmse:,.2f}")
    print(f"Prophet + XGBoost Hybrid RMSE: {final_rmse:,.2f}")
    if prophet_rmse > 0:
        print(f"RMSE Improvement: {((prophet_rmse - final_rmse) / prophet_rmse) * 100:.2f}%")
    print("---------------------------------------------")

    # =========================================================================
    # 4. FUTURE FORECAST: Prophet and ITERATIVE XGBoost
    # =========================================================================
    print(f"4. Generating {FORECAST_DAYS}-Day Future Forecast (Iterative XGBoost)...")

    # --- 4.1 Create Future Dataframe ---
    last_date = prophet_df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='D')
    future = pd.DataFrame({'ds': future_dates})

    # --- 4.2 Populate Future External Features (The "manual_test.py" way) ---
    print("4.1 Generating REAL Future Regressors using 'generate_future_regressors'...")
    # Use the robust function from manual_test.py
    regressors_df = generate_future_regressors(df_Lazada) 

    # Pivot the long-format regressor data into wide format
    regressors_wide = regressors_df.pivot_table(
        index='ds',
        columns='holiday',
        values='holiday',
        aggfunc=lambda x: 1, 
        fill_value=0
    ).reset_index()
    
    # Merge these REAL regressors into our future dataframe
    future = future.merge(regressors_wide, on='ds', how='left')
    
    # Fill NaNs for ALL binary regressors with 0
    future['is_mega_sale_day'] = future['is_mega_sale_day'].fillna(0)
    future['is_payday'] = future['is_payday'].fillna(0)
    future['is_weekend'] = future['is_weekend'].fillna(0)
    
    # Impute the continuous regressor (scaled_discount_rate)
    # We'll use the same logic as the original manual_test.py: fill with the historical mean.
    mean_scaled_discount = df_Lazada['scaled_discount_rate'].mean()
    future['scaled_discount_rate'] = mean_scaled_discount
    
    # --- 4.3 Prophet Prediction ---
    print("4.2 Prophet Base Future Prediction...")
    prophet_future_forecast = model.predict(future)
    future_xgb = future.merge(prophet_future_forecast[['ds', 'yhat']], on='ds')
    future_xgb.rename(columns={'yhat': 'Prophet_yhat'}, inplace=True)
    future_xgb['dayofweek'] = future_xgb['ds'].dt.dayofweek
    future_xgb['dayofyear'] = future_xgb['ds'].dt.dayofyear
    future_xgb['month'] = future_xgb['ds'].dt.month
    future_xgb['year'] = future_xgb['ds'].dt.year

    # --- 4.4 Initialize Lag and Run Iterative XGBoost Prediction ---
    print("4.3 Corrected Iterative XGBoost Prediction Loop...")
    
    last_historical_residual = prophet_df['Residuals'].iloc[-1]
    future_xgb['resid_lag_1'] = np.nan
    future_xgb.loc[future_xgb.index[0], 'resid_lag_1'] = last_historical_residual

    xgb_future_residuals = []
    X_train_cols = FEATURES # Reusing the defined feature list

    for i in range(len(future_xgb)):
        # 1. Prepare features for the current day i
        X_pred = future_xgb.iloc[i][X_train_cols].to_frame().T
        
        # *** FIX: Explicitly cast all feature columns to float ***
        X_pred = X_pred.astype(float)
        
        # 2. Predict the residual
        current_residual_pred = xgb.predict(X_pred)[0]
        xgb_future_residuals.append(current_residual_pred)
        
        # 3. Set the predicted residual as the lag input for the next day (i+1)
        if i < len(future_xgb) - 1:
            future_xgb.loc[future_xgb.index[i + 1], 'resid_lag_1'] = float(current_residual_pred)

    future_xgb['XGBoost_Residuals'] = xgb_future_residuals

    # --- 4.5 Final Hybrid Forecast ---
    future_xgb['Final_Hybrid_Forecast'] = future_xgb['Prophet_yhat'] + future_xgb['XGBoost_Residuals']
    print("Forecast generation complete.")

    # =========================================================================
    # 5. VISUALIZATION AND SAVING
    # =========================================================================
    print("5. Plotting and Saving Forecast...")
    
    # Save the final forecast CSV
    filename_csv = os.path.join(plot_dir, f'Lazada_hybrid_forecast.csv')
    future_xgb.to_csv(filename_csv, index=False)
    print(f"Saved hybrid forecast to {filename_csv}")

    # --- Plotting ---
    historical_data = prophet_df.copy()
    future_data = future_xgb[['ds', 'Prophet_yhat', 'Final_Hybrid_Forecast']].copy()
    future_data.rename(columns={'Prophet_yhat': 'Prophet_Only', 'Final_Hybrid_Forecast': 'Hybrid_Forecast'}, inplace=True)

    plt.figure(figsize=(16, 8))

    # Plot historical actuals
    plt.plot(historical_data['ds'], historical_data['y'], label='Historical Actual Sales', color='blue', linewidth=2, alpha=0.7)
    # Plot historical Prophet base
    plt.plot(historical_data['ds'], historical_data['Prophet_yhat'], label='Historical Prophet Base', color='orange', linestyle=':', alpha=0.5)

    # Plot forecast lines
    plt.plot(future_data['ds'], future_data['Prophet_Only'], label='Prophet Only Forecast', linestyle='--', color='orange', alpha=0.7)
    plt.plot(future_data['ds'], future_data['Hybrid_Forecast'], label='Hybrid Forecast (Prophet + XGBoost)', color='red', linewidth=3)

    # Mark the forecast start
    plt.axvline(last_date, color='gray', linestyle='--', label='Forecast Start Date')

    plt.title(f'Prophet vs. Hybrid Forecast ({FORECAST_DAYS} Days)')
    plt.xlabel('Date')
    plt.ylabel('Sales (y)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    filename_plot = os.path.join(plot_dir, f'Lazada_hybrid_forecast.png')
    plt.savefig(filename_plot)
    print(f"Saved forecast plot to {filename_plot}")
    
    print("\n--- DONE ---")
    print("The final hybrid forecast data is in the 'future_xgb' DataFrame.")

if __name__ == '__main__':
    # NOTE: Execution will require a working Supabase connection and the prophet library installed.
    main_run()
    pass