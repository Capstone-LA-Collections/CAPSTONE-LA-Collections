from matplotlib import ticker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta

# --- ASSUMED IMPORTS FROM PROJECT FILES ---
# NOTE: Ensure these files (data_loader.py, utils.py, stat_models.py, hybrid_xgboost.py) 
# are accessible in the execution environment or directory.
from load_data import load_base_sales_data 
import utils
from stat_models import TimeSeriesTrainer
import hybrid_xgboost

# --- CONFIGURATION ---
START_DATE = '2022-10-05'
PLATFORM = 'lazada'
FORECAST_YEARS = 2 
FORECAST_DAYS = 365 * FORECAST_YEARS
OUTPUT_DIR = 'app/Analytics/Multi Model/what_if_outputs'
BASE_MODEL = 'Prophet' # Using Prophet as the base model for the hybrid

def create_optimization_schedule_daily(current, target, ramp_days=180, total_days=730):
    """
    Creates S-curve adoption schedule (Sigmoid) adapted for DAILY data.
    """
    # S-curve (sigmoid) for the ramp-up period
    x = np.linspace(-3, 3, ramp_days)
    sigmoid = 1 / (1 + np.exp(-x))
    ramp = current + (target - current) * sigmoid

    # Maintain at target for remaining days
    remaining_days = total_days - ramp_days
    if remaining_days > 0:
        steady = np.repeat(target, remaining_days)
        return np.concatenate([ramp, steady])
    else:
        return ramp[:total_days]

def plot_exact_style(df_hist, df_base, df_opt, df_aov_traj, strategy_start_date, current_aov, target_aov):
    """
    Replicates the exact 2-panel plot structure.
    """
    # Aggregate to Monthly for the Visuals (to match the dots/lines style)
    hist_m = df_hist.set_index('ds').resample('MS').agg({'y': 'sum', 'aov': 'mean'}).reset_index()
    base_m = df_base.set_index('ds').resample('MS').agg({'hybrid_forecast': 'sum'}).reset_index()
    opt_m = df_opt.set_index('ds').resample('MS').agg({'hybrid_forecast': 'sum'}).reset_index()
    
    # Setup Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    # --- PANEL 1: REVENUE ---
    # 1. Historical Data (Black Dots)
    ax1.plot(hist_m['ds'], hist_m['y'], 'ko', markersize=6, label='Historical Revenue', alpha=0.7, zorder=4)
    
    # 2. Baseline Forecast (Red Dashed)
    ax1.plot(base_m['ds'], base_m['hybrid_forecast'], 'r--', linewidth=2.5, label='Baseline (Status Quo)', alpha=0.8, zorder=3)
    
    # 3. Optimized Forecast (Blue Solid)
    ax1.plot(opt_m['ds'], opt_m['hybrid_forecast'], 'b-', linewidth=3, label='Optimized (Conditional Discounts)', alpha=0.9, zorder=3)
    
    # 4. Strategy Start Line
    ax1.axvline(x=strategy_start_date, color='green', linestyle=':', alpha=0.7, linewidth=2.5, zorder=2)
    ax1.text(strategy_start_date, ax1.get_ylim()[1]*0.95, 'Conditional\nDiscount Strategy\nStarts', 
             ha='center', fontsize=11, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Formatting Panel 1
    ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Monthly Revenue (₱)', fontsize=13, fontweight='bold')
    ax1.set_title('Revenue Forecast Comparison: Baseline vs Conditional Discount Strategy', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'₱{x/1000:.0f}K'))

    # --- PANEL 2: AOV TRAJECTORY ---
    # 1. Historical AOV (Black Dots - Connected)
    ax2.plot(hist_m['ds'], hist_m['aov'], 'ko-', markersize=5, linewidth=1.5, label='Historical AOV', alpha=0.7, zorder=3)
    
    # 2. Target AOV Trajectory (Blue Solid)
    ax2.plot(df_aov_traj['ds'], df_aov_traj['aov'], 'b-', linewidth=3, label=f'Target AOV (₱{target_aov:.0f})', alpha=0.9, zorder=3)
    
    # 3. Reference Lines
    ax2.axhline(y=current_aov, color='red', linestyle='--', linewidth=2, label=f'Current AOV (₱{current_aov:.0f})', alpha=0.7, zorder=2)
    ax2.axhline(y=target_aov, color='green', linestyle='--', linewidth=2, label=f'Target AOV (₱{target_aov:.0f})', alpha=0.7, zorder=2)
    ax2.axvline(x=strategy_start_date, color='green', linestyle=':', alpha=0.7, linewidth=2.5, zorder=1)
    
    # Formatting Panel 2
    ax2.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Order Value (₱)', fontsize=13, fontweight='bold')
    ax2.set_title('AOV Optimization Trajectory (S-Curve Adoption)', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'₱{x:.0f}'))

    plt.tight_layout()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plot_path = os.path.join(OUTPUT_DIR, 'fig_4_3_2_optimization_comparison_replicated.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] Plot saved to: {plot_path}")


def run_what_if():
    print("--- 1. Data Loading and Preparation ---")
    
    try:
        df = load_base_sales_data(start_date=START_DATE, platform=PLATFORM)
    except (NameError, ImportError):
        print("ERROR: Data loading failed.")
        return
        
    if df.empty:
        print("No data found.")
        return

    # Ensure Date type
    df['ds'] = pd.to_datetime(df['ds'])
    
    # --- 2. CALCULATE STRATEGY LOGIC ---
    print("\n--- 2. Calculating Conditional Discount Strategy ---")
    
    # Use last 6 months for current state
    current_aov = df['aov'].tail(180).mean()
    current_units_per_order = df['units_per_order'].tail(180).mean()
    
    print(f"Current AOV: ₱{current_aov:.2f}")

    # Strategy Parameters
    min_spend_threshold = 600
    discount_amount = 50
    required_aov_increase = min_spend_threshold - discount_amount
    
    min_quantity = 2
    quantity_discount_pct = 10
    
    # Adoption Mix
    strategy1_adoption = 0.60
    strategy2_adoption = 0.40
    
    # Target Calculations
    strategy1_target_aov = required_aov_increase
    strategy2_target_aov = current_aov * (min_quantity / current_units_per_order) * (1 - quantity_discount_pct/100)
    
    blended_target_aov = (strategy1_adoption * strategy1_target_aov) + (strategy2_adoption * strategy2_target_aov)
    
    print(f"Target AOV (Blended): ₱{blended_target_aov:.2f} (+{((blended_target_aov - current_aov) / current_aov * 100):.1f}%)")

    # --- 3. TRAIN BASE MODEL ---
    print("\n--- 3. Training Baseline Model ---")
    trainer = TimeSeriesTrainer(df)
    future_exog_cols = ['aov', 'promo_intensity', 'units_per_order'] 
    
    # Prepare features for BASELINE forecast 
    future_df_base = utils.generate_future_features(df, FORECAST_DAYS)

    # --- FIX: MANUALLY ADD units_per_order TO FUTURE DATAFRAME ---
    # Since utils.generate_future_features doesn't include it by default
    recent_units = df['units_per_order'].tail(30).mean()
    future_df_base['units_per_order'] = recent_units
    # -----------------------------------------------------------
    
    # Train Prophet Base
    preds_base_prophet, fitted_values = trainer.train_predict(
        BASE_MODEL, 
        forecast_horizon=FORECAST_DAYS,
        future_exog=future_df_base[future_exog_cols]
    )
    
    # Train XGBoost Residuals
    residuals = trainer.df['y'] - fitted_values
    residuals = residuals.fillna(0).replace([np.inf, -np.inf], 0)
    resid_df = pd.DataFrame({'ds': trainer.df.index, 'residuals': residuals})
    
    preds_base_hybrid, _ = hybrid_xgboost.run_hybrid_forecast(
        base_forecast=preds_base_prophet,
        historical_residuals_df=resid_df,
        future_df=future_df_base,
        platform_name=PLATFORM
    )
    
    df_forecast_base = future_df_base.copy()
    df_forecast_base['hybrid_forecast'] = preds_base_hybrid

    # --- 4. TRAIN OPTIMIZED MODEL (With S-Curve) ---
    print("\n--- 4. Creating S-Curve Optimized Forecast ---")
    
    # Generate S-Curve Schedule (Daily)
    aov_schedule = create_optimization_schedule_daily(
        current=current_aov, 
        target=blended_target_aov, 
        ramp_days=180, 
        total_days=FORECAST_DAYS
    )
    
    df_forecast_opt = future_df_base.copy()
    # INJECT THE SCHEDULE
    df_forecast_opt['aov'] = aov_schedule
    
    # RE-RUN PROPHET with new AOV (Critical Step)
    preds_opt_prophet, _ = trainer.train_predict(
        BASE_MODEL, 
        forecast_horizon=FORECAST_DAYS,
        future_exog=df_forecast_opt[future_exog_cols]
    )
    
    # RE-RUN XGBOOST with new Features
    preds_opt_hybrid, _ = hybrid_xgboost.run_hybrid_forecast(
        base_forecast=preds_opt_prophet,
        historical_residuals_df=resid_df,
        future_df=df_forecast_opt,
        platform_name=PLATFORM
    )
    
    df_forecast_opt['hybrid_forecast'] = preds_opt_hybrid

    # --- 5. VISUALIZATION ---
    print("\n--- 5. Generating Exact Style Plots ---")
    
    plot_exact_style(
        df_hist=df,
        df_base=df_forecast_base,
        df_opt=df_forecast_opt,
        df_aov_traj=df_forecast_opt[['ds', 'aov']],
        strategy_start_date=df_forecast_base['ds'].min(),
        current_aov=current_aov,
        target_aov=blended_target_aov
    )

if __name__ == "__main__":
    run_what_if()