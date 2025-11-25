import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# Assuming these files are in the same directory or properly added to the path
from sales_preprocessor import load_base_sales_data, preprocess_sales_data
from prophet_modeler import generate_holidays_df, create_and_fit_segmented_models
from forecasting_pipeline import run_cross_validation_and_forecast
from feature_engineer import build_time_series_features

# CRITICAL FIX: Define a global variable to store the regressor list from the modeler
# This ensures it can be accessed if the pipeline logic requires it explicitly.
REGRESSOR_COLS = []

def plot_historical_data(df: pd.DataFrame):
    """Generates and saves a line plot of the full historical data."""
    print("\nSAVING OUTPUT: Plotting historical daily items sold...")
    
    # Ensure output directory exists
    plot_dir = 'app/Analytics/Prophet Model/outputs/historical'
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plotting loop for each platform
    for platform in df['platform_name'].unique():
        df_platform = df[df['platform_name'] == platform]
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_platform['ds'], df_platform['daily_net_sales'], label=platform, linewidth=1.5)
        
        plt.title(f'Historical Daily Net Sales - {platform}')
        plt.xlabel('Date')
        plt.ylabel('Daily Net Sales')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        filename = os.path.join(plot_dir, f'historical_sales_{platform}.png')
        plt.savefig(filename)
        plt.close() # Close figure to free up memory
        print(f"  - Saved: {filename}")
    print("Historical data plots complete.")

def save_platform_data_to_csv(df: pd.DataFrame):
    """Saves the processed DataFrame for each unique platform to a separate CSV file."""
    output_dir = 'app/Analytics/Prophet Model/outputs/historical'
    os.makedirs(output_dir, exist_ok=True)
    
    for platform in df['platform_name'].unique():
        # Filter data for the current platform
        df_platform = df[df['platform_name'] == platform].copy()
        
        # Create a unique filename for each platform
        filename = f'processed_prophet_input_{platform}.csv' 
        full_path = os.path.join(output_dir, filename)
        
        # Save the platform data
        df_platform.to_csv(full_path, index=False)
        print(f"  - Saved platform data for **{platform}** to: {full_path}")
        
    print("Saving processed data to CSVs complete.")

def main_pipeline():
    """
    Main function to execute the end-to-end predictive sales forecasting pipeline.
    """
    global REGRESSOR_COLS

    # --- CONFIGURATION FLAG ---
    # Set this to True to run the computationally intensive hyperparameter tuning
    PERFORM_HYPERPARAMETER_TUNING = False
    
    # --- STEP 1: Load Data ---\
    print("STEP 1: Retrieving Raw Data...")
    df_raw = load_base_sales_data()
    
    if df_raw.empty:
        print("FAILURE: No data loaded. Exiting pipeline.")
        return

    print(f"SUCCESS: Loaded {len(df_raw)} transaction records.")

    # --- STEP 2: Preprocessing and Aggregation ---\
    print("\nSTEP 2: Preprocessing and Aggregation...")
    # df_prophet_ready contains 'ds', 'platform_name', 'daily_items_sold', and regressors
    df_prophet_ready = preprocess_sales_data(df_raw)

    print("\nSTEP 2.5: Building Time-Series Features...")
    df_prophet_ready = build_time_series_features(df_prophet_ready, target_col='daily_net_sales')

    plot_historical_data(df_prophet_ready)
    print("\nSAVING OUTPUT: Saving final processed data to separate CSVs...")
    save_platform_data_to_csv(df_prophet_ready)
    
    if df_prophet_ready.empty:
        print("FAILURE: Preprocessing resulted in an empty DataFrame. Exiting pipeline.")
        return

    # --- STEP 3: Model Training ---\
    print("\nSTEP 3: Generating Holidays and Training Segmented Models...")
    # We are using generate_mega_sale_holidays alias which points to generate_holidays_df
    holidays_df = generate_holidays_df(df_prophet_ready) 
    
    # Train the models for each platform
    # CRITICAL FIX: create_and_fit_segmented_models now returns a tuple (models_dict, regressor_cols)
    models_raw_output = create_and_fit_segmented_models(
        df_prophet_ready, 
        holidays_df, 
        perform_tuning=PERFORM_HYPERPARAMETER_TUNING # <-- NEW PARAMETER
    )

    # Handle the raw output defensively (tuple vs. dict)
    if isinstance(models_raw_output, tuple) and len(models_raw_output) >= 2:
        segmented_models, REGRESSOR_COLS = models_raw_output
        print(f"[NOTE] Training function returned a tuple, successfully extracted model dictionary (Type: {type(segmented_models)} for {segmented_models}).")
    elif isinstance(models_raw_output, dict):
        # Fallback if the modeler didn't return the regressors list
        segmented_models = models_raw_output
        REGRESSOR_COLS = [] # Empty list as fallback
        print(f"[NOTE] Training function returned a dictionary.")
    else:
        print(f"FAILURE: Model training returned an unexpected type ({type(models_raw_output)}). Expected a dictionary or a tuple containing one.")
        return
    
    if not segmented_models:
        print("FAILURE: Model training resulted in an empty model set. Exiting pipeline.")
        return

    # --- STEP 4: Cross-Validation, Forecasting, and Saving Outputs ---\
    print("\nSTEP 4: Running Cross-Validation, Forecasting, and Saving Outputs...")
    # CRITICAL FIX: Pass the definitive list of regressors to the forecasting pipeline
    success = run_cross_validation_and_forecast(segmented_models, df_prophet_ready, REGRESSOR_COLS)

    if success:
        print("\nPipeline completed successfully. Check output files for results.")
    else:
        print("\nPipeline failed during cross-validation or forecasting.")

# CRITICAL FIX: The entire pipeline is now guarded. This prevents child processes
# spawned by Prophet from re-running the entire script when multiprocessing is used
# (e.g., in cross_validation).
if __name__ == '__main__':
    # Add project root to sys path if needed (assuming structure is flat here)
    main_pipeline()