import pandas as pd
import numpy as np
import math
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import time # Import for introducing delays in retries

import sys
import os
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

# - to delete once db is filled -
# --- NEW GLOBAL CONSTANTS FOR FILLER LOGIC ---
SHOPEE_HISTORICAL_YEARLY_AGGREGATES = {
    # These are the correct, desired sales targets.
    2020: 10949,
    2021: 3396123,
    2022: 7799930,
    2023: 6195280,
}
# Shop was established on Sep 19, 2020
SHOPEE_START_DATE = date(2020, 9, 19)

# Existing Monthly data for 2024 (used for seasonal index/AOV proxy)
shopee_2024_monthly_data = [
    {'net_sales': 710711, 'avg_order_value': 528.41},  # Jan 2024
    {'net_sales': 472977, 'avg_order_value': 563.07},   # Feb 2024
    {'net_sales': 515507, 'avg_order_value': 593.90},   # Mar 2024
    {'net_sales': 656759, 'avg_order_value': 567.64},   # Apr 2024
    {'net_sales': 867408, 'avg_order_value': 610.06},   # May 2024
    {'net_sales': 725489, 'avg_order_value': 633.61},   # Jun 2024
    {'net_sales': 847212, 'avg_order_value': 639.89},   # Jul 2024
    {'net_sales': 779787, 'avg_order_value': 586.75},   # Aug 2024
    {'net_sales': 636100, 'avg_order_value': 619.98},   # Sep 2024
    {'net_sales': 583596, 'avg_order_value': 662.42},   # Oct 2024
    {'net_sales': 605118, 'avg_order_value': 609.38},   # Nov 2024
    {'net_sales': 691400, 'avg_order_value': 598.61},   # Dec 2024
]
# - delete until here -


# - to delete once db is filled -
# --- Helper Function to Generate Monthly Targets from Yearly Aggregates ---
def _generate_shopee_monthly_targets():
    """
    Generates a comprehensive list of monthly targets (2020-2024) for Shopee,
    using 2024's monthly distribution as a seasonal index for historical years.
    """
    # 1. Calculate Monthly Seasonal Weights from 2024 data
    total_2024_sales = sum(d['net_sales'] for d in shopee_2024_monthly_data)
    monthly_seasonal_weights = [d['net_sales'] / total_2024_sales for d in shopee_2024_monthly_data]
    
    # Use the average monthly AOV/Pricing as a proxy for the entire historical year
    monthly_avg_aov = [d['avg_order_value'] for d in shopee_2024_monthly_data]
    
    all_monthly_targets = []
    
    # Handle historical years (2020-2023)
    for year, yearly_target_sales in SHOPEE_HISTORICAL_YEARLY_AGGREGATES.items():
        
        # Determine the sum of weights for the active months in this year
        start_month_index = 0
        if year == SHOPEE_START_DATE.year:
            start_month_index = SHOPEE_START_DATE.month - 1
            
        active_weights_sum = sum(monthly_seasonal_weights[start_month_index:])
        
        for month_index in range(12):
            target_date = date(year, month_index + 1, 1)
            
            # Skip months before the shop was established
            if target_date.year < SHOPEE_START_DATE.year or (target_date.year == SHOPEE_START_DATE.year and target_date.month < SHOPEE_START_DATE.month):
                continue

            month_weight = monthly_seasonal_weights[month_index]
            
            # Distribute the yearly target based on active weights
            if active_weights_sum > 0:
                 monthly_sales = yearly_target_sales * (month_weight / active_weights_sum)
            else:
                 monthly_sales = yearly_target_sales * month_weight
                
            # Use the 2024 average order value as a proxy for all years
            monthly_aov = monthly_avg_aov[month_index]

            all_monthly_targets.append({
                'year': year,
                'month': month_index + 1,
                'net_sales': monthly_sales,
                'avg_order_value': monthly_aov
            })
            
    # Add the current 2024 targets
    for month_index in range(12):
        all_monthly_targets.append({
            'year': 2024,
            'month': month_index + 1,
            'net_sales': shopee_2024_monthly_data[month_index]['net_sales'],
            'avg_order_value': shopee_2024_monthly_data[month_index]['avg_order_value']
        })
        
    return all_monthly_targets
# - delete until here -


# --- Original Helper Functions (Unchanged) ---

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

def _calculate_trend_features(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """
    Helper function to calculate the time-series/trend features 
    (mimicking the SQL Window Functions) for a given platform's data block.
    """
    if df.empty:
        return df
        
    df = df.sort_values('date').reset_index(drop=True)

    df['prev_day_sales'] = df['net_sales'].shift(1).fillna(0)
    df['daily_sales_growth'] = (df['net_sales'] - df['prev_day_sales']) / df['prev_day_sales'].replace(0, np.nan)
    df['daily_sales_growth'] = df['daily_sales_growth'].fillna(0)
    df['rolling_sales_7d'] = df['net_sales'].rolling(window=7, min_periods=1).mean()
    df['rolling_sales_growth_7d'] = (df['net_sales'] - df['rolling_sales_7d']) / df['rolling_sales_7d'].replace(0, np.nan)
    df['rolling_sales_growth_7d'] = df['rolling_sales_growth_7d'].fillna(0)
    df['rolling_discount_rate_7d'] = df['avg_discount_rate'].rolling(window=7, min_periods=1).mean()
    df['discount_change_rate_1d'] = df['avg_discount_rate'].diff().fillna(0)
    df['price_change_rate_1d'] = df['avg_paid_price'].diff().fillna(0)
    
    return df


# - to delete once db is filled -
# --- REFACTORED DATA FILLER FUNCTION (Handles 2020-2024) ---
def _fill_shopee_missing_data(df_shopee_existing: pd.DataFrame, all_monthly_targets: list) -> pd.DataFrame:
    """
    REFACTORED: Fills missing Shopee data (2020-2024).
    2020 is treated as a special case: The total yearly deficit is spread across 
    all missing days to ensure a continuous time series.
    2021-2024 are filled based on a positive monthly deficit.
    """
    print("\n[NOTE] Applying REFACTORED Shopee data filler across 2020-2024...")

    if df_shopee_existing.empty:
        df_shopee_existing = pd.DataFrame({'date': pd.to_datetime([]), 'net_sales': []})

    start_date = pd.to_datetime(SHOPEE_START_DATE)
    end_date = pd.to_datetime('2024-12-31')
    
    all_dates_in_scope = pd.date_range(start=start_date, end=end_date, freq='D')
    existing_dates = set(pd.to_datetime(df_shopee_existing['date']))
    missing_dates = [d for d in all_dates_in_scope if d not in existing_dates]
    
    synthetic_records = []
    df_all_targets = pd.DataFrame(all_monthly_targets)
    
    
    # ---------------------------------------------
    # 1. SPECIAL 2020 HANDLING (Yearly Deficit Fill)
    # ---------------------------------------------
    year_2020 = 2020
    
    # Calculate overall 2020 deficit
    existing_sales_2020 = df_shopee_existing[df_shopee_existing['date'].dt.year == year_2020]['net_sales'].sum()
    target_sales_2020 = SHOPEE_HISTORICAL_YEARLY_AGGREGATES.get(year_2020, 0)
    total_deficit_2020 = target_sales_2020 - existing_sales_2020
    
    # Find all missing dates in 2020
    missing_dates_2020 = [d for d in missing_dates if d.year == year_2020 and d.date() >= SHOPEE_START_DATE]

    if total_deficit_2020 > 1.0 and missing_dates_2020:
        
        # Use a placeholder monthly data for pricing info (e.g., Dec 2020 target)
        monthly_data_2020_proxy = df_all_targets[(df_all_targets['year'] == year_2020) & (df_all_targets['month'] == 12)].iloc[0]
        
        print(f"  - YEAR {year_2020}: Total Target ₱{target_sales_2020:,.0f}, Existing ₱{existing_sales_2020:,.0f}. Filling NET DEFICIT of ₱{total_deficit_2020:,.0f} across {len(missing_dates_2020)} missing days. (Skipping monthly weighing/redistribution)")

        # --- Run the weighting/simulation logic for 2020 missing days ---
        day_props = []
        for day in missing_dates_2020:
            is_mega_sale_day = (day.day == day.month)
            is_payday = (day.day in [1, 15, 30])
            
            base_weight = np.random.uniform(0.7, 1.3)
            event_multiplier = 1.0
            
            if is_mega_sale_day:
                event_multiplier = np.random.uniform(3.0, 5.0)
            elif is_payday:
                event_multiplier = np.random.uniform(1.5, 2.5)
                
            day_props.append({
                'date': day,
                'weight': base_weight * event_multiplier,
                'is_mega_sale_day': int(is_mega_sale_day),
                'is_payday': int(is_payday)
            })
        
        # Apply zero and low sales days
        num_days = len(day_props)
        num_zero_days = max(1, int(num_days * 0.1)) 
        num_low_days = max(1, int(num_days * 0.2))
        
        day_props.sort(key=lambda x: x['weight'])
        
        for i in range(num_zero_days):
            day_props[i]['weight'] = 0.0
        for i in range(num_zero_days, num_zero_days + num_low_days):
            day_props[i]['weight'] *= 0.2
            
        # NORMALIZE & CREATE RECORDS:
        total_weight = sum(d['weight'] for d in day_props)
        if total_weight > 0:
             unit_sales = total_deficit_2020 / total_weight
        
             avg_paid_price = monthly_data_2020_proxy['avg_order_value']
             avg_original_price = avg_paid_price / 0.95 
             avg_discount_rate = 1 - (avg_paid_price / avg_original_price)
             ASSUMED_ITEMS_PER_ORDER = 1.5
             avg_item_price_per_sale = avg_paid_price / ASSUMED_ITEMS_PER_ORDER

             for day in day_props:
                 daily_sales = day['weight'] * unit_sales
                 daily_items = 0
                 if daily_sales > 0 and avg_item_price_per_sale > 0:
                     daily_items = math.ceil(daily_sales / avg_item_price_per_sale)
                 elif daily_sales < 0:
                     daily_sales = 0

                 daily_record = {
                     'date': day['date'],
                     'platform_key': 1,
                     'platform_name': 'Shopee',
                     'is_mega_sale_day': day['is_mega_sale_day'],
                     'is_payday': day['is_payday'],
                     'net_sales': daily_sales,
                     'total_items_sold': daily_items,
                     'avg_paid_price': avg_paid_price,
                     'avg_original_price': avg_original_price,
                     'avg_discount_rate': avg_discount_rate,
                 }
                 synthetic_records.append(daily_record)


    # ---------------------------------------------
    # 2. NORMAL HANDLING (Monthly Deficit Fill for 2021-2024)
    # ---------------------------------------------
    df_targets_2021_onwards = df_all_targets[df_all_targets['year'] >= 2021]

    for _, monthly_data in df_targets_2021_onwards.iterrows():
        year = int(monthly_data['year'])
        month_num = int(monthly_data['month'])
        
        start_of_month = date(year, month_num, 1)
        end_of_month = start_of_month + relativedelta(months=1) - relativedelta(days=1)
        
        monthly_target_sales = monthly_data['net_sales']
        
        # 1. CROSS-CHECK: Calculate sales from *existing* DB records for this month
        df_month_existing = df_shopee_existing[
            (df_shopee_existing['date'] >= pd.to_datetime(start_of_month)) &
            (df_shopee_existing['date'] <= pd.to_datetime(end_of_month))
        ]
        
        sales_existing = df_month_existing['net_sales'].sum()

        # 2. CALCULATE DEFICIT: Find sales we need to generate
        sales_deficit = monthly_target_sales - sales_existing
        
        # Find the specific missing days for *this* month
        month_missing_dates = [d for d in missing_dates if d.year == year and d.month == month_num and d.date() >= start_of_month]
        
        # --- NORMAL HANDLING (2021-2024): ONLY FILL IF POSITIVE DEFICIT ---
        if sales_deficit <= 0 or not month_missing_dates:
            continue
            
        deficit_to_fill = sales_deficit
        
        # This is the print line for 2021+ months being filled
        print(f"  - Month {year}-{month_num:02d}: Target ₱{monthly_target_sales:,.0f}, Existing ₱{sales_existing:,.0f}. Filling deficit of ₱{deficit_to_fill:,.0f} across {len(month_missing_dates)} days.")


        # 3. SIMULATE: Distribute deficit across missing days with weighted logic
        day_props = []
        for day in month_missing_dates:
            is_mega_sale_day = (day.day == day.month)
            is_payday = (day.day in [1, 15, 30])
            
            base_weight = np.random.uniform(0.7, 1.3)
            event_multiplier = 1.0
            
            if is_mega_sale_day:
                event_multiplier = np.random.uniform(3.0, 5.0)
            elif is_payday:
                event_multiplier = np.random.uniform(1.5, 2.5)
                
            day_props.append({
                'date': day,
                'weight': base_weight * event_multiplier,
                'is_mega_sale_day': int(is_mega_sale_day),
                'is_payday': int(is_payday)
            })
        
        # Apply zero and low sales days
        num_days = len(day_props)
        num_zero_days = max(1, int(num_days * 0.1)) 
        num_low_days = max(1, int(num_days * 0.2)) 
        
        day_props.sort(key=lambda x: x['weight'])
        
        for i in range(num_zero_days):
            day_props[i]['weight'] = 0.0 
        for i in range(num_zero_days, num_zero_days + num_low_days):
            day_props[i]['weight'] *= 0.2 
            
        # 4. NORMALIZE & CREATE RECORDS:
        total_weight = sum(d['weight'] for d in day_props)
        if total_weight == 0:
             continue
             
        unit_sales = deficit_to_fill / total_weight
        
        # Monthly average pricing from the generated targets
        avg_paid_price = monthly_data['avg_order_value']
        avg_original_price = avg_paid_price / 0.95
        avg_discount_rate = 1 - (avg_paid_price / avg_original_price)
        ASSUMED_ITEMS_PER_ORDER = 1.5
        avg_item_price_per_sale = avg_paid_price / ASSUMED_ITEMS_PER_ORDER

        for day in day_props:
            daily_sales = day['weight'] * unit_sales
            daily_items = 0
            if daily_sales > 0 and avg_item_price_per_sale > 0:
                daily_items = math.ceil(daily_sales / avg_item_price_per_sale)
            elif daily_sales < 0:
                daily_sales = 0

            daily_record = {
                'date': day['date'],
                'platform_key': 1,
                'platform_name': 'Shopee',
                'is_mega_sale_day': day['is_mega_sale_day'],
                'is_payday': day['is_payday'],
                'net_sales': daily_sales,
                'total_items_sold': daily_items,
                'avg_paid_price': avg_paid_price,
                'avg_original_price': avg_original_price,
                'avg_discount_rate': avg_discount_rate,
            }
            synthetic_records.append(daily_record)
            
    # --- 5. COMBINE AND CALCULATE TRENDS ---
    df_synthetic_new = pd.DataFrame(synthetic_records)
    
    # Combine the *existing* data with the *newly generated* synthetic data
    df_shopee_complete = pd.concat([df_shopee_existing, df_synthetic_new], ignore_index=True)
    df_shopee_complete = df_shopee_complete.sort_values('date').reset_index(drop=True)
    
    print(f"\nSuccessfully generated {len(df_synthetic_new)} new records for missing days across 2020-2024.")
    
    # CRITICAL: Calculate trend features on the *full Shopee block*
    df_shopee_filled = _calculate_trend_features(df_shopee_complete, 'Shopee')
    
    return df_shopee_filled
# - delete until here -

# --- REFACTORED DATA LOADER FUNCTION (UNCHANGED) ---
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

# --- PREPROCESSING FUNCTION (Unchanged) ---
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