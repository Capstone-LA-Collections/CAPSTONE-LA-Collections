import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
import logging
from datetime import date, timedelta
import calendar # Required for the new robust holiday logic

from sklearn.model_selection import ParameterGrid
import warnings

# --- IMPROVED HOLIDAY GENERATION FUNCTION ---
def generate_holidays_df(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a DataFrame of major holidays and sales events (Mega_Sale_Day, Payday).
    This function is based on the detailed logic provided by the user.
    """
    logging.warning("Using robust helper logic for generate_holidays_df.")

    if df_input.empty:
        return pd.DataFrame(columns=['ds', 'holiday'])
        
    start_date = df_input['ds'].min()
    end_date = df_input['ds'].max() + pd.Timedelta(days=365) # Generate holidays into the future
    
    dates = pd.date_range(start=start_date, end=end_date)
    holidays_list = []

    MEGA_SALE_DAYS = [
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), 
        (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
        (2, 14), (5, 1), (6, 12), (8, 21), (11, 1), (12, 24), (12, 25), (12, 30)
    ]
    
    # Helper to find the last specific weekday in a month
    def get_last_weekday_of_month(year, month, weekday): # 0=Mon, 4=Fri
        last_day = date(year, month, 1) + timedelta(days=32)
        last_day = last_day.replace(day=1) - timedelta(days=1)
        while last_day.weekday() != weekday:
            last_day -= timedelta(days=1)
        return last_day

    for d in dates:
        month = d.month
        day = d.day
        year = d.year
        
        # --- Mega Sale Days (Double Digits + National Holidays) ---
        is_mega_sale = False
        if (month, day) in MEGA_SALE_DAYS:
            is_mega_sale = True

        # Black Friday (Last Friday of November)
        if month == 11:
            try:
                last_friday = get_last_weekday_of_month(year, 11, 4)
                if d.date() == last_friday:
                    is_mega_sale = True
            except ValueError:
                pass # Skip if date calculation fails

        # Cyber Monday (Monday after last Thursday of November)
        if month == 11 or (month == 12 and day <= 3):
            try:
                # Find last Thursday of Nov
                last_thursday = get_last_weekday_of_month(year, 11, 3) 
                cyber_monday = last_thursday + timedelta(days=4)
                if d.date() == cyber_monday:
                    is_mega_sale = True
            except ValueError: 
                pass 
                
        if is_mega_sale:
            # We use 'Mega_Sale_Day' as the holiday name for Prophet's holidays parameter
            holidays_list.append({'ds': d, 'holiday': 'Mega_Sale_Day'})
            
        # --- Payday Logic ---
        is_payday = False
        
        # Payday 1: 15th
        if day == 15:
            is_payday = True
        
        # Payday 2: Last day of the month (or prior Friday if on weekend)
        try:
            last_day_of_month = (date(year, month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            # Check for last day
            if d.date() == last_day_of_month:
                is_payday = True
            
            # Check for early payment (Friday before weekend)
            elif last_day_of_month.weekday() in [5, 6]: # Sat or Sun
                friday_before = last_day_of_month - timedelta(days=last_day_of_month.weekday() - 4)
                if d.date() == friday_before:
                    is_payday = True
        except ValueError:
            pass # Skip if date calculation fails

        if is_payday:
            holidays_list.append({'ds': d, 'holiday': 'Payday'})
            
    holidays_df = pd.DataFrame(holidays_list).drop_duplicates().sort_values('ds').reset_index(drop=True)
    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
    
    # We use lower_window=0, upper_window=0 because the lag regressors already capture pre/post effects
    holidays_df['lower_window'] = 0
    holidays_df['upper_window'] = 0
    
    return holidays_df

def tune_model_hyperparameters(df_input: pd.DataFrame, holidays_df: pd.DataFrame, regressor_cols: list) -> dict:
    """
    Performs grid search hyperparameter tuning for a single segment (platform).

    NOTE: For simplicity and speed, this only tunes the first platform found.
    For production, this should be run for each platform and the best params saved.

    Returns:
        dict: The best hyperparameters found.
    """
    
    warnings.filterwarnings('ignore') 
    
    # 1. Prepare Data Segments
    df_segment_map = {}
    df_prep = df_input.rename(
        columns={'daily_net_sales': 'y', 'date': 'ds'}
    ).copy()
    
    platform_names = df_prep['platform_name'].unique()
    
    for name in platform_names:
        df_segment_map[name] = df_prep[df_prep['platform_name'] == name][
            #['ds', 'y', 'cap'] + regressor_cols
            ['ds', 'y'] + regressor_cols
        ].copy()

    print(f"\n--- Starting Unified Hyperparameter Tuning for {len(platform_names)} Platforms ---")
    print("Goal: Ensure at least 8 CV cuts (7 periods) per platform.")
    print(f"Platforms: {', '.join(platform_names)}")

    # Define the full hyperparameter grid (to test every parameter possible)
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'holidays_prior_scale': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'seasonality_mode': ['multiplicative', 'additive'],
        'yearly_seasonality': [True,],
        'weekly_seasonality': [True],
        'daily_seasonality': [True]
    }

    all_params = list(ParameterGrid(param_grid))
    overall_maes = [] 
    horizon_days = 365
    horizon = f'{horizon_days} days' 

    # 2. Iterate through all parameter sets
    for i, params in enumerate(all_params):
        print(f"\n- Testing param set {i+1}/{len(all_params)}")
        
        param_maes = [] 

        # 3. Perform Cross-Validation for EACH Platform
        for platform_name, df_segment in df_segment_map.items():
            
            # --- DYNAMIC CV PARAMETER CALCULATION ---
            total_days = len(df_segment)
            
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
                # Calculate the largest period that still gives us 7 periods (8 cuts)
                if platform_name == 'Shopee':
                    period_days = 81
                else:
                    period_days = 27    

            period = f'{period_days} days'
                 
            print(f"  - {platform_name} CV settings: initial={initial}, period={period}, horizon={horizon}")
            
            # Re-initialize the model for each platform's fit
            m = Prophet(
                holidays=holidays_df,
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                holidays_prior_scale=params['holidays_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                daily_seasonality=params['daily_seasonality']
            )
            
            # Add regressors
            for col in regressor_cols:
                m.add_regressor(col, standardize='auto', prior_scale=10.0)

            # Fit and perform cross-validation with dynamic parameters
            try:
                m.fit(df_segment)
                df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, parallel=None)
                df_p = performance_metrics(df_cv, metrics=['mae'])
                
                mean_mae = df_p['mae'].mean()
                param_maes.append(mean_mae)
                print(f"  - {platform_name} CV Mean MAE: {mean_mae:,.2f} ({len(df_p)} cuts)")
                
            except Exception as e:
                print(f"  [ERROR] CV failed for {platform_name} with set {i+1}. Details: {e}")
                param_maes.append(np.inf) 

        # 4. Aggregate Metrics to get the Unified Score (Mean of all platforms' mae)
        unified_mae = np.mean(param_maes)
        overall_maes.append(unified_mae)
        print(f"  --> Unified Mean mae: {unified_mae:,.2f}")


    # 5. Find the Best Parameters
    best_params_index = np.argmin(overall_maes)
    best_params = all_params[best_params_index]
    best_mae = overall_maes[best_params_index]
    
    print(f"\n--- Unified Tuning Complete ---")
    print(f"Best Unified Mean MAE: {best_mae:,.2f}")
    print(f"Best Parameters: {best_params}")
    warnings.filterwarnings('default') 
    
    return best_params

def create_and_fit_segmented_models(df_input: pd.DataFrame, holidays_df: pd.DataFrame, perform_tuning: bool = False):
    """
    Initializes, configures, and fits a separate Prophet model for each platform.
    
    Args:
        df_input: DataFrame containing 'ds', 'daily_items_sold', 'platform_name', and regressors.
        holidays_df: DataFrame containing the generated holiday/event calendar.

    Returns:
        tuple: (dict of trained Prophet models, list of regressor column names)
    """
    print("Configuring segmented Prophet model instances with aggressive tuning...")
    
    segmented_models = {}
    platform_names = df_input['platform_name'].unique()

    # --- REGRESSOR SELECTION ---
    # Use a predefined list of features identified from feature engineering
    EXCLUDED_COLS = ['ds', 'y', 'platform_name', 'daily_net_sales', 'daily_items_sold',
                     'daily_sales_growth_smoothed', 'rolling_sales_growth_7d','is_mega_sale_day', 'is_payday']
    
    base_regressors = [c for c in df_input.columns if c not in EXCLUDED_COLS]
    
    regressor_cols = [
        c for c in base_regressors
        if not c.startswith('lag_') and
           not c.startswith('rolling_') and
           not c.startswith('daily_growth_rate')
    ]
    
    print(f"Using {len(regressor_cols)} Regressors, excluding {EXCLUDED_COLS[4:]}: {regressor_cols}")

    # --- AGGRESSIVE TUNING PARAMETERS ---\
    TUNING_PARAMS = {
        'changepoint_prior_scale': 0.05, 
        'seasonality_prior_scale': 0.01,  
        'holidays_prior_scale': 0.05,   
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality':True,
        'daily_seasonality': True
    }
    
    # --- HYPERPARAMETER TUNING LOGIC ---
    if perform_tuning and not df_input.empty:
        best_params = tune_model_hyperparameters(df_input, holidays_df, regressor_cols)
        
        # Overwrite default aggressive tuning with the best found parameters
        TUNING_PARAMS.update(best_params)
        print("\n[NOTE] Using tuned parameters for final model training.")

    print(f"Applying aggressive tuning: changepoint_prior_scale={TUNING_PARAMS['changepoint_prior_scale']}, holidays_prior_scale={TUNING_PARAMS['holidays_prior_scale']}")
    
    for platform_name in platform_names:
        
        # 1. Initialize the Prophet model with tuned parameters
        m = Prophet(
            holidays=holidays_df,
            changepoint_prior_scale=TUNING_PARAMS['changepoint_prior_scale'],
            seasonality_prior_scale=TUNING_PARAMS['seasonality_prior_scale'],
            holidays_prior_scale=TUNING_PARAMS['holidays_prior_scale'],
            seasonality_mode=TUNING_PARAMS['seasonality_mode'],
            daily_seasonality=TUNING_PARAMS['daily_seasonality']
        )
        
        # 2. Add Regressors
        for col in regressor_cols:
            # CRITICAL: This is where the model is informed about external regressors
            m.add_regressor(col, standardize='auto', prior_scale=10.0) # Using a large prior_scale for aggressive fitting
            
        # 3. Filter data for segment
        df_segment = df_input[df_input['platform_name'] == platform_name].copy()
        
        # 4. Prepare for fit
        df_segment = df_segment.rename(columns={'daily_net_sales': 'y', 'date': 'ds'})
        
        # 5. Fit the model
        try:
            # Drop unnecessary columns before fit (e.g., platform_name and the other excluded columns)
            cols_to_fit = ['ds', 'y'] + regressor_cols
            df_segment_fit = df_segment[cols_to_fit].copy()

            nigger = len(df_segment_fit)

            print(f"Receing a total of {nigger} records")
            
            m.fit(df_segment_fit)
            segmented_models[platform_name] = m
            print(f"Successfully trained model for {platform_name}.")
        except Exception as e:
            print(f"[ERROR] Failed to fit model for {platform_name}. Details: {e}")
            
    # CRITICAL FIX: Return both the models and the definitive list of regressors
    return (segmented_models, regressor_cols)