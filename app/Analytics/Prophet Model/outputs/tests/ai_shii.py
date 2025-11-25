import time
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from supabase import create_client, Client
from supabase.client import ClientOptions

# --- Setup (Same as original) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..','..','..')) 

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config import SUPABASE_URL, SUPABASE_KEY
except ImportError:
    print("Error: config.py not found or missing SUPABASE_URL/SUPABASE_KEY.")
    sys.exit(1)

MAX_ROWS = 100000
MAX_RETRIES = 3 
_CACHED_SALES_DATA = None 

RPC_COLUMNS = [
    'net_sales',
    'total_items_sold',
    'platform_key',
    'date', 
    'platform_name',
    'is_mega_sale_day',
    'is_payday',
    'avg_paid_price',
    'avg_original_price',
    'avg_discount_rate',
    'prev_day_sales',
    'daily_sales_growth',
    'rolling_sales_7d',
    'rolling_sales_growth_7d',
    'rolling_discount_rate_7d',
    'discount_change_rate_1d',
    'price_change_rate_1d',
]

def get_supabase_client() -> Client:
    """Initializes and returns the Supabase client."""
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=ClientOptions(schema="la_collections",))
    return supabase

def _process_dataframe(data: list) -> pd.DataFrame:
    """Standard processing steps for data frames returned by Supabase RPCs."""
    df = pd.DataFrame(data)

    if df.empty:
        print("Warning: RPC returned an an empty dataset.")
        return pd.DataFrame() 

    for col in ['is_mega_sale_day', 'is_payday']:
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
        'total_items_sold': 'sum',
        'net_sales': 'sum',
        'avg_discount_rate': 'mean',
        'is_mega_sale_day': 'max',
        'is_payday': 'max',
    }
    
    # Group by date (ds) and platform_name (Segmentation Key)
    df_daily = df_raw.groupby(['date', 'platform_name']).agg(aggregation_schema).reset_index()
    
    # Rename columns for Prophet and clarity
    df_daily = df_daily.rename(columns={
        'date': 'ds',
        'total_items_sold': 'daily_items_sold', 
        'net_sales': 'daily_net_sales',
        'avg_discount_rate': 'avg_discount_rate_daily', 
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
        
        # --- Two-step Imputation Process ---
        df_resampled['avg_discount_rate_daily'] = df_resampled['avg_discount_rate_daily'].ffill()
        
        for col in ['is_mega_sale_day', 'is_payday']:
            df_resampled[col] = df_resampled[col].ffill().fillna(0).astype(int)

        df_resampled['daily_items_sold'] = df_resampled['daily_items_sold'].fillna(0)
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
        
    print("Data processing complete.")
    return df_final

def load_base_sales_data(start_date='2020-09-19'):
    """
    Executes the PostgreSQL RPC function, then applies the REFACTORED
    data-filling logic to *all* Shopee data from Sep 2020 to Dec 2024.
    """
    global _CACHED_SALES_DATA
    
    if _CACHED_SALES_DATA is not None and not _CACHED_SALES_DATA.empty:
        print("[INFO] Returning cached data instead of executing RPC.")
        return _CACHED_SALES_DATA.copy() 
    
    supabase = get_supabase_client()
    
    for attempt in range(MAX_RETRIES):
        try:
            print("STEP 1.1: Executing Supabase RPC function: get_factor_analysis_data_v2 (for model)...")
            response = (
                supabase.rpc(
                'get_factor_analysis_data_v2', 
                {'start_date_param': start_date}
                )
                .range(0, MAX_ROWS - 1)
                .execute()
                )

            df_raw = _process_dataframe(response.data)
            
            
            
            # NOTE: When the data is fully loaded, this section will be replaced by:
            df_final = df_raw[RPC_COLUMNS].copy()
            
            print(f"[SUCCESS] Predictive model data loading complete. Loaded {len(df_final)} total rows (Real + Synthetic).")
            
            # Cache the *final, filled* data
            _CACHED_SALES_DATA = df_final.copy() 
            return df_final
            
        except Exception as e:
            error_message = str(e)
            if '57014' in error_message or 'timeout' in error_message.lower():
                print(f"Supabase Query Error: {error_message}")
                if attempt < MAX_RETRIES - 1:
                    sleep_time = 2 ** (attempt + 1)
                    print(f"Attempt {attempt + 1} of {MAX_RETRIES} failed due to timeout. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Final attempt failed after {MAX_RETRIES} retries. Returning empty DataFrame.")
                    return pd.DataFrame()
            else:
                print(f"Supabase Query Error: {e}")
                return pd.DataFrame() 

    return pd.DataFrame()


# --- Configuration ---
PLATFORM_NAME = 'Shopee'
FORECAST_PERIODS = 365 # 1 year forecast
OUTPUT_DIR = 'prophet_outputs_rpc'

# --- 1. Data Loading and Preprocessing (Refactored) ---
def load_and_preprocess_data():
    """
    Loads data via RPC, filters for Shopee, cleans zeros, and engineers features.
    """
    # Call the user's RPC function to get the full historical dataset
    df_full = load_base_sales_data()

    # Filter for the target platform
    df_platform = df_full[df_full['platform_name'] == PLATFORM_NAME].copy()

    # Standardize column names for Prophet
    df_platform['ds'] = pd.to_datetime(df_platform['ds'])
    df_platform = df_platform.rename(columns={'daily_net_sales': 'y'})

    # Refactoring Action 1: Remove the initial block of zero sales
    # Assuming start of meaningful data is 2020-10-04, check your data for verification!
    first_valid_date = pd.to_datetime('2020-10-04')
    df_clean = df_platform[df_platform['ds'] >= first_valid_date].copy()

    # Refactoring Action 2: Feature Engineering - Create Lag Regressor
    # This highly predictive feature requires a sorted DataFrame
    df_clean = df_clean.sort_values('ds').reset_index(drop=True)
    df_clean['prev_day_sales'] = df_clean['y'].shift(1)
    
    # Fill the first NaN created by shifting with 0.0 (the value from the day before the clean data)
    df_clean['prev_day_sales'] = df_clean['prev_day_sales'].fillna(0.0)

    # Select final columns for Prophet
    df_prophet = df_clean[[
        'ds', 'y',
        'is_mega_sale_day',
        'is_payday',
        'avg_discount_rate_daily',
        'prev_day_sales'
    ]].copy()
    
    print(f"Data prepared for {PLATFORM_NAME}. Start date: {df_prophet['ds'].min().date()}")
    
    return df_prophet

# --- 2. Regressor Generation for Future Dates ---
def generate_future_regressors(df_prophet, periods):
    """Generates the future dates and imputes values for extra regressors."""
    
    # Create the future date frame
    m = Prophet()
    future = m.make_future_dataframe(periods=periods, freq='D', include_history=False)

    # 2.1 Mega Sale Day Logic (Double-date sales: 1.1, 2.2, ..., 12.12)
    mega_sale_dates = [
        pd.to_datetime(f'{date.year}-{m:02d}-{d:02d}')
        for date in future['ds']
        for m, d in zip(range(1, 13), range(1, 13))
        if date.month == m and date.day == d
    ]

    # 2.2 Payday Logic (15th and 30th/End of Month)
    payday_dates = [
        date for date in future['ds'] 
        if date.day == 15 or date.day == 30 or date.day == date.days_in_month
    ]
    
    # Initialize future regressor columns to 0
    future['is_mega_sale_day'] = 0
    future['is_payday'] = 0

    # Apply the calculated dates
    future.loc[future['ds'].isin(mega_sale_dates), 'is_mega_sale_day'] = 1
    future.loc[future['ds'].isin(payday_dates), 'is_payday'] = 1
    
    # 2.3 Discount Rate (Imputed Future Value)
    # Use the historical average discount rate for future predictions.
    avg_discount = df_prophet['avg_discount_rate_daily'].mean()
    future['avg_discount_rate_daily'] = avg_discount

    # 2.4 Prev Day Sales (Imputed Future Value)
    # Use the historical average sales value as a placeholder for the lag feature.
    avg_sales = df_prophet['y'].mean()
    future['prev_day_sales'] = avg_sales 

    # Combine history (for the fit step) and future regressors (for the predict step)
    future_combined = pd.concat([
        df_prophet[['ds', 'is_mega_sale_day', 'is_payday', 'avg_discount_rate_daily', 'prev_day_sales']],
        future
    ], ignore_index=True)

    return future_combined.sort_values(by='ds').reset_index(drop=True)

# --- 3. Model Definition and Execution ---
def run_refactored_prophet_model(df_prophet, future_df):
    
    m = Prophet(
        yearly_seasonality=False, # Disabled to use custom fourier order
        daily_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    
    # Custom yearly seasonality (fourier_order=8 is often a good fit)
    m.add_seasonality(name='yearly', period=365.25, fourier_order=8)
    
    # Refactoring Action: Add external regressors
    # 'Mega Sale' and 'Payday' are now separate, highly flexible regressors
    m.add_regressor('is_mega_sale_day', mode='multiplicative')
    m.add_regressor('is_payday', mode='multiplicative')
    m.add_regressor('avg_discount_rate_daily', mode='additive')
    m.add_regressor('prev_day_sales', mode='additive')

    # Fit the model
    model = m.fit(df_prophet)

    # Make the prediction
    forecast = model.predict(future_df)

    # --- 4. Plotting and Saving ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f'Refactored Prophet Sales Forecast ({PLATFORM_NAME})')
    plt.savefig(os.path.join(OUTPUT_DIR, 'refactored_forecast.png'))
    plt.close(fig)

    # Plot the components
    fig_comp = model.plot_components(forecast)
    plt.savefig(os.path.join(OUTPUT_DIR, 'refactored_components.png'))
    plt.close(fig_comp)
    
    # Save the forecast data to CSV
    forecast.to_csv(os.path.join(OUTPUT_DIR, 'refactored_forecast_data.csv'), index=False)
    
    print(f"\n✅ Success! Prophet model run complete.")
    print(f"Output files saved to the '{OUTPUT_DIR}' directory.")
    
    return forecast

if __name__ == '__main__':
    try:
        df_prophet_history = load_and_preprocess_data()
        df_prophet_future = generate_future_regressors(df_prophet_history, FORECAST_PERIODS)
        run_refactored_prophet_model(df_prophet_history, df_prophet_future)
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to run model. Check your Supabase connection and function.")
        print(f"Details: {e}")