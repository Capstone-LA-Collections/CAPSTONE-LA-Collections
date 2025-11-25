from datetime import date
import os
import sys
import numpy as np
import pandas as pd
from supabase import create_client, Client
from supabase.client import ClientOptions 
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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

MAX_RETRIES = 3 
_CACHED_SALES_DATA = None 

start_date_str = date(2022, 10, 1).isoformat()  # '2022-10-01'
end_date_str = date(2025, 11, 1).isoformat()    # '2025-11-01'
platform_str = 'Lazada'

RPC_COLUMNS = [
    'ds', 
    'y',
    'total_orders',
    'total_units',
    'aov',
    'promo_intensity',
    'units_per_order',
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

    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
    
    return df

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

def main_run():
    Output_dir = 'app/Analytics/Prophet Model/outputs/tests'
    os.makedirs(Output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {Output_dir}")

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    df = load_base_sales_data()
    if df.empty:
        print("Cannot run model: Base sales data is empty.")
        return
    df['ds'] = pd.to_datetime(df['ds'])

    print("="*80)
    print("📊 PHASE 1: BASELINE FORECAST (Status Quo)")
    print("="*80)
    print(f"\nHistorical Period: {df['ds'].min().date()} to{df['ds'].max().date()}")
    print(f"Data Points: {len(df)} months")
    print(f"\nCurrent State (Last 6 Months Average):")
    print(f" Monthly Revenue: ₱{df['y'].tail(6).mean():>10,.2f}")
    print(f" Average AOV: ₱{df['aov'].tail(6).mean():>10,.2f}")
    print(f" Units per Order:{df['units_per_order'].tail(6).mean():>10,.2f}")
    print(f" Promo Intensity:{df['promo_intensity'].tail(6).mean():>10,.2f}%")

    print("\n🔮 Building Baseline Model (Time Series + Seasonality)...")

    model_baseline = Prophet(
    growth='linear',
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    seasonality_prior_scale=5.0,     
    interval_width=0.80
    )

    print(df.corrwith(df['y']))

    model_baseline.fit(df)
    print("✓ Baseline model trained")

    print("\n📅 Generating 24-month forecast...")

    future_baseline = model_baseline.make_future_dataframe(periods=24, freq='MS')
    forecast_baseline = model_baseline.predict(future_baseline)

    # Extract future only
    forecast_baseline_future = forecast_baseline[forecast_baseline['ds'] >df['ds'].max()].copy()
    print("✓ Forecast generated")

    print("\n📈 BASELINE FORECAST RESULTS (No Strategy Change):")
    print("="*80)
    baseline_summary = []
    for year in [2026, 2027]:
        year_data = forecast_baseline_future[forecast_baseline_future['ds'].dt.year == year]
        revenue = year_data['yhat'].sum()
        ci_lower = year_data['yhat_lower'].sum()
        ci_upper = year_data['yhat_upper'].sum()
        monthly_avg = revenue / 12
        baseline_summary.append({
        'Year': year,
        'Annual_Revenue': revenue,
        'Monthly_Avg': monthly_avg,
        'CI_Lower': ci_lower,
        'CI_Upper': ci_upper
        })
        print(f"\n{year}:")
        print(f" Annual Revenue: ₱{revenue:>12,.0f}")
        print(f" Monthly Average: ₱{monthly_avg:>12,.2f}")
        print(f" 80% CI: ₱{ci_lower:>12,.0f} -₱{ci_upper:>12,.0f}")

    filename_csv = os.path.join(Output_dir, f'phase1_baseline_forecast.csv')
    forecast_baseline_future.to_csv(filename_csv, index=False)
    filename_csv = os.path.join(Output_dir, f'phase1_baseline_summary.csv')
    pd.DataFrame(baseline_summary).to_csv(filename_csv, index=False)

    print("\n✓ Saved: phase1_baseline_forecast.csv")
    print("✓ Saved: phase1_baseline_summary.csv")

    print("\n📊 Generating baseline visualization...")
    fig, ax = plt.subplots(figsize=(16, 8))
    # Historical data
    ax.plot(df['ds'], df['y'],'ko', markersize=6, label='Historical Revenue', alpha=0.7, zorder=3)
    # Baseline forecast
    ax.plot(forecast_baseline_future['ds'], forecast_baseline_future['yhat'], 'r--', linewidth=3, label='Baseline Forecast (Status Quo)', alpha=0.8, zorder=2)
    # Confidence interval
    ax.fill_between(forecast_baseline_future['ds'], forecast_baseline_future['yhat_lower'], forecast_baseline_future['yhat_upper'], alpha=0.15, color='red', label='80% Confidence Interval', zorder=1)
    # Mark current date
    current_date = df['ds'].max()
    ax.axvline(x=current_date, color='orange', linestyle=':', alpha=0.7, linewidth=2, zorder=2)
    ax.text(current_date, ax.get_ylim()[1]*0.95, 'Current\n(Oct 2025)', ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    # Formatting
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Monthly Revenue (₱)', fontsize=14, fontweight='bold')
    ax.set_title('Phase 1: Baseline Revenue Forecast (2026-2028)\nStatus Quo: No Strategic Intervention', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₱{x/1000:.0f}K'))
    plt.tight_layout()
    filename_plot = os.path.join(Output_dir, f'fig_4_3_1_baseline_forecast.png')
    plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
    print(" ✓ Saved: fig_4_3_1_baseline_forecast.png")
    plt.close()
    print("\n" + "="*80)
    print("✅ PHASE 1 COMPLETE")
    print("="*80)

    print("="*80)
    print("📊 PHASE 2: CONDITIONAL DISCOUNT OPTIMIZATION")
    print("="*80)

    current_aov = df['aov'].tail(6).mean()
    current_units_per_order = df['units_per_order'].tail(6).mean()
    current_monthly_revenue = df['y'].tail(6).mean()

    print(f"\n📍 CURRENT STATE (Last 6 Months):")
    print(f" AOV: ₱{current_aov:.2f}")
    print(f" Units per Order: {current_units_per_order:.2f}")
    print(f" Monthly Revenue: ₱{current_monthly_revenue:,.2f}")

    print(f"\n🎯 CONDITIONAL DISCOUNT STRATEGIES:")
    print("-"*80)

    min_spend_threshold = 600 # ₱600 minimum
    discount_amount = 50 # ₱50 off
    required_aov_increase = min_spend_threshold - discount_amount

    print(f"\nStrategy 1: Minimum Spend Threshold")
    print(f" Offer: '₱{discount_amount} off orders ≥₱{min_spend_threshold}'")
    print(f" Current AOV: ₱{current_aov:.2f}")
    print(f" Required AOV: ₱{min_spend_threshold:.2f} (before discount)")
    print(f" Net AOV: ₱{required_aov_increase:.2f} (after discount)")
    print(f" AOV Increase: +{((min_spend_threshold - current_aov) / current_aov * 100):.1f}%")
    print(f" Mechanism: Forces customers to add ₱{min_spend_threshold - current_aov:.2f} worth of items")

    min_quantity = 2
    quantity_discount_pct = 10

    print(f"\nStrategy 2: Quantity-Linked Discount")
    print(f" Offer: 'Buy {min_quantity}+ items, Get {quantity_discount_pct}% off'")
    print(f" Current Units: {current_units_per_order:.2f} units/order")
    print(f" Required Units: {min_quantity:.1f}+ units/order")
    print(f" Units Increase: +{((min_quantity - current_units_per_order) / current_units_per_order * 100):.1f}%")
    print(f" Expected AOV: ₱{current_aov * (min_quantity / current_units_per_order) * (1 - quantity_discount_pct/100):.2f}")
    print(f" Mechanism: Forces basket size increase")

    strategy1_adoption = 0.60
    strategy2_adoption = 0.40
    strategy1_target_aov = required_aov_increase
    strategy2_target_aov = current_aov * (min_quantity / current_units_per_order) * (1 - quantity_discount_pct/100)
    blended_target_aov = (strategy1_adoption * strategy1_target_aov) + (strategy2_adoption * strategy2_target_aov)
    blended_target_units = (strategy1_adoption * (min_spend_threshold / (current_aov / current_units_per_order))) + (strategy2_adoption * min_quantity)

    print(f"\n🎯 BLENDED TARGET (Weighted Average):")

    print("-"*80)
    print(f" Adoption Mix: {strategy1_adoption*100:.0f}% Strategy 1, {strategy2_adoption*100:.0f}% Strategy 2")
    print(f" Target AOV: ₱{blended_target_aov:.2f} (+{((blended_target_aov - current_aov) / current_aov * 100):.1f}%)")
    print(f" Target Units: {blended_target_units:.2f} units/order (+{((blended_target_units - current_units_per_order) / current_units_per_order * 100):.1f}%)")

    def create_optimization_schedule(current, target, ramp_months=6, total_months=24):
        """
        Create S-curve adoption schedule for gradual strategy
        rollout
        """

        # S-curve (sigmoid) for first 6 months
        x = np.linspace(-3, 3, ramp_months)
        sigmoid = 1 / (1 + np.exp(-x))
        ramp = current + (target - current) * sigmoid

        # Maintain at target for remaining months
        steady = np.repeat(target, total_months - ramp_months)
        return np.concatenate([ramp, steady])

    aov_schedule = create_optimization_schedule(current_aov, blended_target_aov)
    units_schedule = create_optimization_schedule(current_units_per_order, blended_target_units)

    print(f"\n📅 IMPLEMENTATION TIMELINE:")
    print("-"*80)
    print(f" Month 1: AOV = ₱{aov_schedule[0]:.2f}, Units = {units_schedule[0]:.2f} (Starting)")
    print(f" Month 3: AOV = ₱{aov_schedule[2]:.2f}, Units = {units_schedule[2]:.2f} (Ramp-up)")
    print(f" Month 6: AOV = ₱{aov_schedule[5]:.2f}, Units = {units_schedule[5]:.2f} (Full adoption)")
    print(f" Month 12: AOV = ₱{aov_schedule[11]:.2f}, Units = {units_schedule[11]:.2f} (Sustained)")

    # Save schedule
    optimization_schedule = pd.DataFrame({
        'month': range(1, 25),
        'target_aov': aov_schedule,
        'target_units_per_order': units_schedule
    })

    filename_csv = os.path.join(Output_dir, f'phase2_optimization_schedule.csv')
    forecast_baseline_future.to_csv(filename_csv, index=False)
    print("\n✓ Saved: phase2_optimization_schedule.csv")

    print("\n" + "="*80)
    print("🔮 BUILDING OPTIMIZED FORECAST MODEL")
    print("="*80)

    model_optimized = Prophet(
    growth='linear',
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
    interval_width=0.80
    )

    model_optimized.add_regressor(
        'aov',
        prior_scale=20.0, # Strong impact expected
        mode='additive', # Direct revenue contribution
        standardize='auto'
    )

    print("\n🔮 Training model with AOV regressor...")
    model_optimized.fit(df[['ds', 'y', 'aov']])
    print("✓ Model trained with AOV optimization")

    future_optimized = model_optimized.make_future_dataframe(periods=24, freq='MS')
    future_optimized = future_optimized.merge(df[['ds', 'aov']],on='ds',how='left')
    forecast_start_idx = len(df)
    future_optimized.loc[forecast_start_idx:, 'aov'] = aov_schedule

    print("✓ Future dataframe prepared with optimized AOV values")
    print(f" Historical months: {forecast_start_idx}")
    print(f" Forecast months: {len(aov_schedule)}")

    print("\n📊 Generating optimized forecast...")
    forecast_optimized = model_optimized.predict(future_optimized)
    forecast_opt_future = forecast_optimized[forecast_optimized['ds'] > df['ds'].max()].copy()
    print("✓ Optimized forecast generated")

    print("\n" + "="*80)
    print("📈 FORECAST COMPARISON: BASELINE VS OPTIMIZED")
    print("="*80)

    comparison_results = []

    for year in [2026, 2027]:
        # Baseline
        base_year = forecast_baseline_future[forecast_baseline_future['ds'].dt.year == year]
        base_revenue = base_year['yhat'].sum()
        # Optimized
        opt_year = forecast_opt_future[forecast_opt_future['ds'].dt.year == year]
        opt_revenue = opt_year['yhat'].sum()
        # Calculate improvement
        improvement = opt_revenue - base_revenue
        improvement_pct = (improvement / base_revenue * 100)
        comparison_results.append({
            'Year': year,
            'Baseline_Revenue': base_revenue,
            'Optimized_Revenue': opt_revenue,
            'Improvement_Amount': improvement,
            'Improvement_Percent': improvement_pct
        })

        print(f"\n{year}:")
        print(f" Baseline (Status Quo): ₱{base_revenue:>12,.0f}")
        print(f" Optimized (Conditional): ₱{opt_revenue:>12,.0f}")
        print(f" Improvement: ₱{improvement:>12,.0f} (+{improvement_pct:>5.1f}%)")

    print("\n" + "="*80)
    print("🔍 AOV-REVENUE ELASTICITY")
    print("="*80)

    current_annual_revenue = df['y'].tail(12).sum()
    forecast_2026_baseline = forecast_baseline_future[forecast_baseline_future['ds'].dt.year == 2026]['yhat'].sum()
    forecast_2026_optimized = forecast_opt_future[forecast_opt_future['ds'].dt.year == 2026]['yhat'].sum()

    baseline_growth = ((forecast_2026_baseline / current_annual_revenue) - 1) * 100
    optimized_growth = ((forecast_2026_optimized / current_annual_revenue) - 1) * 100
    incremental_growth = optimized_growth - baseline_growth
    aov_increase_pct = ((blended_target_aov - current_aov) / current_aov) * 100

    elasticity = incremental_growth / aov_increase_pct

    print(f"\nCurrent Annual Revenue (Last 12 mo): ₱{current_annual_revenue:>12,.0f}")
    print(f"2026 Baseline Forecast: ₱{forecast_2026_baseline:>12,.0f} (+{baseline_growth:>5.1f}%)")
    print(f"2026 Optimized Forecast: ₱{forecast_2026_optimized:>12,.0f} (+{optimized_growth:>5.1f}%)")
    print(f"\nAOV Increase: +{aov_increase_pct:.1f}%")
    print(f"Incremental Revenue Growth: +{incremental_growth:.1f}% (vs baseline)")
    print(f"\n📊 AOV-Revenue Elasticity: {elasticity:.2f}")
    print(f" Interpretation: 1% ↑ AOV → {elasticity:.2f}% ↑ Revenue")


    filename_csv = os.path.join(Output_dir, f'phase2_optimized_forecast.csv')
    forecast_opt_future.to_csv(filename_csv, index=False)
    filename_csv = os.path.join(Output_dir, f'phase2_comparison_summary.csv')
    pd.DataFrame(comparison_results).to_csv(filename_csv, index=False)

    print("\n✓ Saved: phase2_optimized_forecast.csv")
    print("✓ Saved: phase2_comparison_summary.csv")

    print("\n📊 Generating comparison visualization...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    ax1.plot(df['ds'], df['y'], 'ko', markersize=6, label='Historical Revenue', alpha=0.7, zorder=4)
    ax1.plot(forecast_baseline_future['ds'], forecast_baseline_future['yhat'], 'r--', linewidth=2.5, label='Baseline (Status Quo)', alpha=0.8, zorder=3)
    ax1.fill_between(forecast_baseline_future['ds'], forecast_baseline_future['yhat_lower'], forecast_baseline_future['yhat_upper'], alpha=0.15, color='red', zorder=1)

    ax1.plot(forecast_opt_future['ds'], forecast_opt_future['yhat'], 'b-', linewidth=3, label='Optimized (Conditional Discounts)', alpha=0.9, zorder=3)
    ax1.fill_between(forecast_opt_future['ds'], forecast_opt_future['yhat_lower'], forecast_opt_future['yhat_upper'], alpha=0.2, color='blue', zorder=1)

    strategy_start = df['ds'].max()
    ax1.axvline(x=strategy_start, color='green', linestyle=':', alpha=0.7, linewidth=2.5, zorder=2)
    ax1.text(strategy_start, ax1.get_ylim()[1]*0.95, 'Conditional\nDiscount Strategy\nStarts', ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Monthly Revenue (₱)', fontsize=13, fontweight='bold')
    ax1.set_title('Revenue Forecast Comparison: Baseline vs Conditional Discount Strategy', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₱{x/1000:.0f}K'))

    ax2.plot(df['ds'], df['aov'], 'ko-', markersize=5, linewidth=1.5, label='Historical AOV', alpha=0.7, zorder=3)

    future_dates = pd.date_range(start=df['ds'].max() + pd.DateOffset(months=1), periods=24, freq='MS')

    ax2.plot(future_dates, aov_schedule, 'b-', linewidth=3, label=f'Target AOV (₱{blended_target_aov:.0f})', alpha=0.9, zorder=3)
    ax2.axhline(y=current_aov, color='red', linestyle='--', linewidth=2, label=f'Current AOV (₱{current_aov:.0f})', alpha=0.7, zorder=2)
    ax2.axhline(y=blended_target_aov, color='green', linestyle='--', linewidth=2, label=f'Target AOV (₱{blended_target_aov:.0f})', alpha=0.7, zorder=2)
    ax2.axvline(x=strategy_start, color='green', linestyle=':', alpha=0.7, linewidth=2.5, zorder=1)
    ax2.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Order Value (₱)', fontsize=13, fontweight='bold')
    ax2.set_title('AOV Optimization Trajectory (S-Curve Adoption)', fontsize=15, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₱{x:.0f}'))

    plt.tight_layout()

    filename_plot = os.path.join(Output_dir, f'fig_4_3_2_optimization_comparison.png')
    plt.savefig(filename_plot, dpi=300, bbox_inches='tight')
    print(" ✓ Saved: fig_4_3_2_optimization_comparison.png")
    plt.close()

    print("\n" + "="*80)
    print("✅ PHASE 2 COMPLETE")
    print("="*80)

    print("\n" + "="*80)
    print("📋 PREDICTIVE ANALYSIS SUMMARY")
    print("="*80)

    print(f"\n🎯 CONDITIONAL DISCOUNT STRATEGY:")
    print(f" Strategy 1: ₱50 off orders ≥₱600")
    print(f" Strategy 2: Buy 2+, Get 10% off")
    print(f" → Target AOV: ₱{blended_target_aov:.2f} (+{((blended_target_aov - current_aov) / current_aov * 100):.1f}%)")
    print(f" → Target Units/Order: {blended_target_units:.2f} (+{((blended_target_units - current_units_per_order) / current_units_per_order * 100):.1f}%)")
    print(f"\n📈 2026 FORECAST RESULTS:")
    print(f" Baseline Revenue: ₱{forecast_2026_baseline:>12,.0f}")
    print(f" Optimized Revenue: ₱{forecast_2026_optimized:>12,.0f}")
    print(f" Improvement: ₱{forecast_2026_optimized - forecast_2026_baseline:>12,.0f} (+{incremental_growth:.1f}%)")
    print(f"\n🔬 KEY FINDING:")
    print(f" AOV-Revenue Elasticity = {elasticity:.2f}")
    print(f" → Every 1% increase in AOV generates {elasticity:.2f}% revenue growth")
    print(f" → Breaking the negative β = -49.3 trade-off through conditional mechanics")
    print("\n" + "="*80)
    print("✅ PREDICTIVE ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    # NOTE: Execution will require a working Supabase connection and the prophet library installed.
    main_run()
    pass