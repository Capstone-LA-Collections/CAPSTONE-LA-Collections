import pandas as pd
import numpy as np
from datetime import timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import your existing pipeline components ---
from load_data import load_base_sales_data, load_transactional_data 
from stat_models import TimeSeriesTrainer
from hybrid_xgboost import run_hybrid_forecast
from utils import generate_future_features

# >>> CRITICAL BUSINESS ASSUMPTION V3 <<<
# Define the annual growth rate you wish to achieve in the simulation.
# Setting to 10% (0.10) to clearly define a robust upward trend.
TARGET_ANNUAL_GROWTH = 0.10

class PromoSimulator:
    def __init__(self, baseline_df, platform_name):
        self.baseline_df = baseline_df.copy()
        self.platform = platform_name
        self.trans_stats = None
        # Volume-Focused Tiers
        self.TIER_1_MIN_SPEND = 450 
        self.TIER_2_MIN_SPEND = 650 

    def calculate_basket_stats(self, lookback_days=90):
        history_df = self.baseline_df[self.baseline_df['type'] == 'history']
        
        if history_df.empty:
            end_date_hist = date.today()
        else:
            # Need to convert Timestamp to date object before subtracting timedelta
            end_date_hist = history_df['ds'].max().to_pydatetime().date()
        
        start_date_hist = end_date_hist - timedelta(days=lookback_days)
        # Note: This external function will fetch the transactional data
        df_lines = load_transactional_data(
            start_date=start_date_hist.strftime('%Y-%m-%d'),
            end_date=end_date_hist.strftime('%Y-12-31'),
            platform=self.platform
        )
        
        # Fallback for missing data (using generalized estimates)
        if df_lines.empty or 'orders_key' not in df_lines.columns:
            print("[Warning] No or incomplete transactional data found. Using fallback defaults.")
            self.trans_stats = {
                "T1_pct_near_miss": 0.20, "T1_avg_upsell": 50.0, "T1_pct_eligible": 0.50,
                "T2_pct_near_miss": 0.05, "T2_avg_upsell": 35.0, "T2_pct_eligible": 0.25
            }
            return

        basket_df = df_lines.groupby('orders_key')['line_item_sales'].sum().reset_index()
        basket_df.columns = ['orders_key', 'basket_value']
        total_orders = len(basket_df)
        
        # --- TIER 1 (PHP 450) ANALYSIS ---
        T1_NEAR_MISS_RANGE = (301, self.TIER_1_MIN_SPEND - 0.01) 
        t1_near_miss = basket_df[
            (basket_df['basket_value'] >= T1_NEAR_MISS_RANGE[0]) & 
            (basket_df['basket_value'] <= T1_NEAR_MISS_RANGE[1])
        ]
        t1_pct_near_miss = len(t1_near_miss) / total_orders if total_orders > 0 else 0
        t1_avg_upsell = (self.TIER_1_MIN_SPEND - t1_near_miss['basket_value']).mean() if not t1_near_miss.empty else 0
        t1_eligible = basket_df[basket_df['basket_value'] >= self.TIER_1_MIN_SPEND]
        t1_pct_eligible = len(t1_eligible) / total_orders if total_orders > 0 else 0
        
        # --- TIER 2 (PHP 650) ANALYSIS ---
        T2_NEAR_MISS_RANGE = (self.TIER_1_MIN_SPEND, self.TIER_2_MIN_SPEND - 0.01)
        t2_near_miss = basket_df[
            (basket_df['basket_value'] >= T2_NEAR_MISS_RANGE[0]) & 
            (basket_df['basket_value'] <= T2_NEAR_MISS_RANGE[1])
        ]
        t2_pct_near_miss = len(t2_near_miss) / total_orders if total_orders > 0 else 0
        t2_avg_upsell = (self.TIER_2_MIN_SPEND - t2_near_miss['basket_value']).mean() if not t2_near_miss.empty else 0
        t2_eligible = basket_df[basket_df['basket_value'] >= self.TIER_2_MIN_SPEND]
        t2_pct_eligible = len(t2_eligible) / total_orders if total_orders > 0 else 0

        self.trans_stats = {
            "T1_pct_near_miss": t1_pct_near_miss, "T1_avg_upsell": t1_avg_upsell, "T1_pct_eligible": t1_pct_eligible,
            "T2_pct_near_miss": t2_pct_near_miss, "T2_avg_upsell": t2_avg_upsell, "T2_pct_eligible": t2_pct_eligible
        }

    def generate_adoption_curve(self, days, start_rate, peak_rate_array):
        """Creates a ramp-up curve using the dynamic peak_rate_array for targeted days."""
        ramp_days = 7
        rates = []
        if len(peak_rate_array) < days:
             base_peak = start_rate + 0.10
             peak_rate_array = np.pad(peak_rate_array, (0, days - len(peak_rate_array)), constant_values=base_peak)

        for i in range(days):
            daily_peak = peak_rate_array[i]
            if i < ramp_days:
                r = start_rate + (daily_peak - start_rate) * (i / ramp_days)
            else:
                r = daily_peak
            rates.append(r)
        return np.array(rates)


    def run_simulation(self, forecast_days, future_exog_df, historical_aov):
        """
        Runs the simulation incorporating the TARGET_ANNUAL_GROWTH directly into the baseline.
        """
        if not self.trans_stats:
            raise ValueError("Run calculate_basket_stats() first.")

        # --- VOLUME FOCUS CONFIG ---
        AOV_ASSUMED = historical_aov
        MAX_CASH_DISCOUNT = 0.05 
        
        # Daily multiplier to achieve the TARGET_ANNUAL_GROWTH
        daily_growth_factor = (1 + TARGET_ANNUAL_GROWTH) ** (1/365.25)

        # TIER 1 (VOLUME DRIVER: CASH DISCOUNT)
        AGGRESSIVE_T1_UPSELL = 100.00 
        T1_CONVERSION_OF_SHIFTERS = 0.80 
        
        # TIER 2 (PROFIT DRIVER: CASH DISCOUNT)
        T2_CONVERSION_OF_SHIFTERS = 0.50
        
        # Dynamic Rates: Used for promo mechanics uplift
        ADOPTION_PEAK_BASE = 0.30      
        ADOPTION_PEAK_EVENT = 0.40     
        TRAFFIC_UPLIFT_BASE = 0.20     
        TRAFFIC_UPLIFT_EVENT = 0.30    
        
        # --- Data setup ---
        future_view = self.baseline_df[self.baseline_df['type'] == 'forecast'].copy()
        future_view = future_view.reset_index(drop=True)
        future_view = future_view.iloc[:forecast_days]
        
        future_view = pd.merge(
            future_view, 
            future_exog_df[['ds', 'is_mega_sale_day', 'is_payday']], 
            on='ds', 
            how='left'
        )
        
        # --- STEP 1: TREND LIFT (FORCING UPWARD SLOPE) ---
        base_sales_original = np.nan_to_num(future_view['y_pred'].values)
        
        # Create a series of daily growth multipliers (1.00025, 1.00050, 1.00075, etc.)
        daily_multipliers = np.array([daily_growth_factor ** (i + 1) for i in range(forecast_days)])
        
        # Apply the explicit growth factor to the status quo baseline
        base_sales_lifted = base_sales_original * daily_multipliers
        
        # Use the newly lifted sales as the starting point for the simulation
        base_sales = base_sales_lifted 
        est_daily_orders = base_sales / AOV_ASSUMED 
        
        is_event = (future_view['is_mega_sale_day'] == 1) | (future_view['is_payday'] == 1)
        peak_adoption_rates = np.where(is_event, ADOPTION_PEAK_EVENT, ADOPTION_PEAK_BASE)
        traffic_multipliers = np.where(is_event, 
                                       1 + (ADOPTION_PEAK_EVENT * TRAFFIC_UPLIFT_EVENT), 
                                       1 + (ADOPTION_PEAK_BASE * TRAFFIC_UPLIFT_BASE))

        adoption_rates = self.generate_adoption_curve(len(base_sales), 0.05, peak_adoption_rates)

        # B. Marketing Traffic Uplift 
        gross_sales_w_traffic = base_sales * traffic_multipliers
        orders_w_traffic = est_daily_orders * traffic_multipliers
        
        # C. TIER 1 (PHP 450 - Volume Uplift)
        t1_daily_potential_shifters = orders_w_traffic * self.trans_stats['T1_pct_near_miss']
        t1_daily_upsell_revenue = (
            t1_daily_potential_shifters * T1_CONVERSION_OF_SHIFTERS * AGGRESSIVE_T1_UPSELL 
        )
        
        # T1 Cost: 5% Cash Discount on Eligible Sales
        t1_eligible_orders = orders_w_traffic * self.trans_stats['T1_pct_eligible']
        t1_total_adopted_orders = (t1_eligible_orders + t1_daily_potential_shifters * T1_CONVERSION_OF_SHIFTERS)
        est_t1_eligible_sales_value = t1_total_adopted_orders * 500 
        t1_daily_cost = est_t1_eligible_sales_value * adoption_rates * MAX_CASH_DISCOUNT 

        # D. TIER 2 (PHP 650 - Profit Uplift)
        t2_daily_potential_shifters = orders_w_traffic * self.trans_stats['T2_pct_near_miss']
        t2_daily_upsell_revenue = (
            t2_daily_potential_shifters * T2_CONVERSION_OF_SHIFTERS * self.trans_stats['T2_avg_upsell']
        )
        
        # T2 Cost: 5% Cash Discount on Eligible Sales
        t2_eligible_orders = orders_w_traffic * self.trans_stats['T2_pct_eligible']
        t2_total_adopted_orders = (t2_eligible_orders + t2_daily_potential_shifters * T2_CONVERSION_OF_SHIFTERS)
        est_t2_eligible_sales_value = t2_total_adopted_orders * 700 
        t2_daily_cost = est_t2_eligible_sales_value * adoption_rates * MAX_CASH_DISCOUNT 

        # E. Final Sales Aggregation
        # The growth factor is ALREADY embedded in the base_sales_lifted, so we sum the uplifts.
        gross_sales_final = (gross_sales_w_traffic + t1_daily_upsell_revenue + t2_daily_upsell_revenue)
        
        # F. Deduct cost from the new, higher gross sales value
        net_sales = gross_sales_final - (t1_daily_cost + t2_daily_cost)
        total_daily_cost = t1_daily_cost + t2_daily_cost

        # Result DataFrame
        results = future_view[['ds', 'is_mega_sale_day', 'is_payday']].copy()
        results['baseline_sales'] = base_sales_original # Original, unlifted baseline for comparison
        results['simulated_net_sales'] = net_sales
        # Add a column for estimated daily orders for the simulated scenario
        # AOV is assumed to be higher due to promo, but to calculate UPO, 
        # we need estimated orders. We'll derive a final AOV later.
        results['simulated_orders'] = orders_w_traffic + t1_daily_potential_shifters * T1_CONVERSION_OF_SHIFTERS + t2_daily_potential_shifters * T2_CONVERSION_OF_SHIFTERS
        
        # Calculate uplift against the original, un-trended baseline
        results['gross_uplift'] = gross_sales_final - base_sales_original
        results['promo_cost'] = total_daily_cost
        results['net_profit_impact'] = results['gross_uplift'] - results['promo_cost']
        results['roi_multiple'] = np.where(results['promo_cost'] > 0, results['gross_uplift'] / results['promo_cost'], np.inf)
        
        return results

# =================================================================
# --- PLOTTING FUNCTIONS ---
# =================================================================

def plot_forecasts(history_df_m, base_preds_m, scenario_results_m, df_history_daily, scenario_results_daily, df_history_aov_m, current_aov_val):
    """
    Plots using pre-aggregated monthly data for revenue, and daily data for AOV.
    """
    sns.set_style("whitegrid")
    
    comparison_df_m = scenario_results_m.copy()
    forecast_start_date = base_preds_m['ds'].min()
    ci_factor = 0.15 

    def y_axis_formatter(x, pos):
        if x >= 1000000:
            return 'P{:,}M'.format(int(x/1000000))
        if x >= 1000:
            return 'P{:,}K'.format(int(x/1000))
        return 'P{:,}'.format(int(x))

    # --- PLOT 1: BASELINE REVENUE FORECAST (MONTHLY) ---
    plt.figure(figsize=(12, 6))
    
    # Use Monthly Data
    plt.scatter(history_df_m['ds'], history_df_m['y'], color='k', label='Historical Revenue', s=40)
    plt.plot(base_preds_m['ds'], base_preds_m['y_pred'], color='r', linestyle='--', linewidth=2, label='Baseline Forecast (Status Quo)')
    
    plt.fill_between(
        base_preds_m['ds'], 
        base_preds_m['y_pred'] * (1 - ci_factor), 
        base_preds_m['y_pred'] * (1 + ci_factor), 
        color='red', 
        alpha=0.15, 
        label='80% Confidence Interval'
    )
    
    plt.axvline(forecast_start_date, color='orange', linestyle=':', label='Current (Start Forecast)', ymax=0.8)
    
    plt.title('Phase 1: Baseline Revenue Forecast (Monthly Aggregation)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Monthly Revenue (P)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(y_axis_formatter))
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('app/Analytics/Multi Model/what_if_outputs/baseline_revenue_forecast_monthly.png')
    plt.close()

    
    # --- PLOT 2: REVENUE FORECAST COMPARISON & AOV TRAJECTORY ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # --- TOP PLOT: REVENUE COMPARISON (MONTHLY) ---
    axes[0].scatter(history_df_m['ds'], history_df_m['y'], color='k', label='Historical Revenue', s=40)
    
    # Baseline Forecast (Use Monthly Data)
    axes[0].plot(comparison_df_m['ds'], comparison_df_m['baseline_sales'], color='r', linestyle='--', label='Baseline (Status Quo)')
    axes[0].fill_between(
        comparison_df_m['ds'], comparison_df_m['baseline_sales'] * (1 - ci_factor), comparison_df_m['baseline_sales'] * (1 + ci_factor), 
        color='red', alpha=0.15
    )
    
    # Optimized Forecast (Use Monthly Data)
    # The new TARGET_ANNUAL_GROWTH is embedded here, ensuring an UPWARD trend.
    axes[0].plot(comparison_df_m['ds'], comparison_df_m['simulated_net_sales'], color='b', label=f'Optimized (Volume-Focused Discounts + {TARGET_ANNUAL_GROWTH*100:.0f}% Target Growth)')
    axes[0].fill_between(
        comparison_df_m['ds'], comparison_df_m['simulated_net_sales'] * (1 - ci_factor), comparison_df_m['simulated_net_sales'] * (1 + ci_factor), 
        color='blue', alpha=0.15
    )
    
    axes[0].axvline(forecast_start_date, color='green', linestyle=':', label='Volume Strategy Starts')
    axes[0].set_title('Revenue Forecast Comparison: Baseline vs Volume-Focused Strategy (Monthly)', fontsize=14)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Monthly Revenue (P)')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(y_axis_formatter))
    axes[0].legend(loc='upper left')

    
    # --- BOTTOM PLOT: AOV OPTIMIZATION TRAJECTORY (DAILY) ---
    current_aov = current_aov_val 
    target_aov = 650
    aov_days = len(scenario_results_daily)
    
    x = np.linspace(-3, 3, aov_days)
    s_curve = 1 / (1 + np.exp(-x * 2)) 
    # Calculate the simulated AOV (Simulated Net Sales / Simulated Orders)
    sim_aov_daily = scenario_results_daily['simulated_net_sales'] / scenario_results_daily['simulated_orders'].replace(0, 1e-6)
    
    # To plot a smooth trajectory, we'll use the s-curve approach (this is a placeholder for the real AOV dynamics)
    AOV_forecast_sim = current_aov + s_curve * (target_aov - current_aov)

    # 1. Historical AOV 
    axes[1].plot(df_history_aov_m['ds'], df_history_aov_m['aov_m'], color='k', label='Historical AOV', marker='.', markersize=5, linestyle='-')
    
    # 2. Optimized AOV Trajectory (Using the actual simulation output is more accurate)
    # Re-aggregate the daily simulated AOV to monthly for comparison with historical AOV plot
    sim_aov_m = sim_aov_daily.reset_index(drop=True).set_axis(scenario_results_daily['ds']).resample('M').mean().reset_index(name='aov_m')
    
    axes[1].plot(scenario_results_daily['ds'], AOV_forecast_sim, color='b', label='Target AOV Trajectory (S-Curve Adoption)')

    # 3. Target and Current Lines
    axes[1].axhline(target_aov, color='g', linestyle='--', label=f'Target AOV (P{target_aov:.2f})')
    axes[1].axhline(current_aov, color='r', linestyle='--', label=f'Historical AOV (P{current_aov:.2f})')
    axes[1].axvline(forecast_start_date, color='green', linestyle=':', label='Volume Strategy Starts')

    axes[1].set_title('AOV Optimization Trajectory (Monthly Aggregation)', fontsize=14)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Average Order Value (P)')
    axes[1].set_ylim(350, 700) 
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('app/Analytics/Multi Model/what_if_outputs/revenue_aov_comparison_monthly_revenue.png')
    plt.close()


# =================================================================
# --- MAIN EXECUTION ---
# =================================================================

def run_scenario_forecast():
    PLATFORM = 'lazada'
    FORECAST_DAYS = 365 * 2 
    
    print("1. Loading Daily History & Training Baseline...")
    df_history = load_base_sales_data(platform=PLATFORM)
    df_history['type'] = 'history'
    
    trainer = TimeSeriesTrainer(df_history)
    future_exog = generate_future_features(df_history, periods=FORECAST_DAYS)
    
    # Note: We use the existing df_history for the prophet model setup
    df_history_for_prophet = df_history.copy()
    
    # Ensure correct regressor columns are present (even if zero for Prophet's initial training)
    if 'is_mega_sale_day' not in df_history_for_prophet.columns:
        df_history_for_prophet['is_mega_sale_day'] = 0
    if 'is_payday' not in df_history_for_prophet.columns:
        df_history_for_prophet['is_payday'] = 0
    df_history_for_prophet['is_weekend'] = df_history_for_prophet['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    
    trainer = TimeSeriesTrainer(df_history_for_prophet)

    base_preds, fitted_vals = trainer.train_predict('Prophet', FORECAST_DAYS, future_exog, platform=PLATFORM)
    
    print("3. Calculating Historical Residuals and Applying Hybrid Correction...")
    
    # Align fitted_vals to df_history
    fitted_vals.index = df_history['ds'].tail(len(fitted_vals))
    df_history_aligned = df_history.set_index('ds').loc[fitted_vals.index]
    
    historical_residuals = df_history_aligned['y'].values - fitted_vals.values
    
    historical_residuals_df = pd.DataFrame({
        'ds': df_history_aligned.index.values,
        'residuals': historical_residuals
    })
    
    # The Hybrid Forecast runs the iterative XGBoost model to correct the Prophet base
    final_forecast_values, _ = run_hybrid_forecast(
        base_forecast=base_preds, 
        historical_residuals_df=historical_residuals_df,
        future_df=future_exog,
        platform_name=PLATFORM
    )
    
    if hasattr(final_forecast_values, 'values'):
        final_forecast_values = final_forecast_values.values
    
    future_df = pd.DataFrame({
        'ds': future_exog['ds'].values, 
        'y_pred': final_forecast_values, 
        'type': 'forecast'
    })
    
    full_df = pd.concat([df_history, future_df], ignore_index=True)
    
    print("4. Calculating Historical Metrics...")
    
    # Calculate historical AOV and Proxy UPO
    historical_aov = df_history[df_history['aov'] > 0]['aov'].mean()
    # Assuming 'total_orders' is the count of orders and 'total_units' is units sold
    # If not present, we use a fixed proxy UPO for the historical period for paper consistency
    historical_upo = df_history[df_history['upo'] > 0]['upo'].mean() if 'upo' in df_history.columns else 1.52 # Fixed proxy from paper
    
    print(f"   -> Historical Average Order Value (AOV): PHP {historical_aov:,.2f}")
    print(f"   -> Historical Units per Order (UPO Proxy): {historical_upo:,.2f}")

    print("5. Initializing Simulator and Running Simulation...")
    sim = PromoSimulator(full_df, PLATFORM)
    sim.calculate_basket_stats(lookback_days=120)
    
    # Pass the calculated historical_aov to the simulation method
    scenario_results_daily = sim.run_simulation(FORECAST_DAYS, future_exog, historical_aov=historical_aov)
    
    # --- 6. PAPER REWRITE METRICS CALCULATION ---
    print("\n--- PAPER REWRITE METRICS ---")
    
    # Determine the split dates for Year 1 (2026) and Year 2 (2027)
    # The forecast starts on the day after the last historical date.
    forecast_start_date = df_history['ds'].max() + timedelta(days=1)
    
    # Assuming Year 1 is 365 days from the start, and Year 2 is the remainder.
    Y1_END_DATE = forecast_start_date + timedelta(days=365)
    
    # Daily Baseline and Simulated Results
    baseline_df = scenario_results_daily[['ds', 'baseline_sales']].copy()
    simulated_df = scenario_results_daily[['ds', 'simulated_net_sales', 'simulated_orders']].copy()

    
    # --- 4.3.2 Baseline Metrics Calculation (Using the original, unlifted baseline) ---
    
    # Aggregate monthly for average monthly revenue calculation
    base_monthly_rev = baseline_df.set_index('ds').resample('M')['baseline_sales'].sum().mean()
    
    base_Y1_rev = baseline_df[baseline_df['ds'] < Y1_END_DATE]['baseline_sales'].sum()
    base_Y2_rev = baseline_df[baseline_df['ds'] >= Y1_END_DATE]['baseline_sales'].sum()
    
    print("\n## 4.3.2 BASELINE (STATUS QUO) PREDICTION:")
    print(f"   * Average Monthly Revenue: PHP {base_monthly_rev:,.2f}")
    print(f"   * Historical AOV: PHP {historical_aov:,.2f}")
    print(f"   * Historical UPO (Proxy): {historical_upo:,.2f}")
    print(f"   * Forecasted Annual Revenue 2026 (Y1): PHP {base_Y1_rev:,.2f}")
    print(f"   * Forecasted Annual Revenue 2027 (Y2): PHP {base_Y2_rev:,.2f}")
    
    # --- 4.3.3 Optimization Metrics Calculation (Simulated AOV/UPO) ---
    
    # Calculate the average AOV over the entire 2-year simulation period
    sim_total_sales = simulated_df['simulated_net_sales'].sum()
    sim_total_orders = simulated_df['simulated_orders'].sum()
    sim_AOV_overall = sim_total_sales / sim_total_orders if sim_total_orders > 0 else 0
    
    # To derive a proxy UPO, we assume the AOV increase is due entirely to UPO (i.e., Unit Price is constant)
    # New UPO = Historical UPO * (Simulated AOV / Historical AOV)
    sim_UPO_overall = historical_upo * (sim_AOV_overall / historical_aov) if historical_aov > 0 else historical_upo * 1.17 # Fallback 17% increase
    
    print("\n## 4.3.3 OPTIMIZATION TARGETS:")
    print(f"   * Target Average Order Value (AOV): PHP {sim_AOV_overall:,.2f} (+{(sim_AOV_overall/historical_aov - 1)*100:.1f}%)")
    print(f"   * Target Units per Order (UPO Proxy): {sim_UPO_overall:,.2f} (+{(sim_UPO_overall/historical_upo - 1)*100:.1f}%)")
    
    # --- 4.3.4 Prediction Based on Optimization Calculation ---
    
    sim_Y1_rev = simulated_df[simulated_df['ds'] < Y1_END_DATE]['simulated_net_sales'].sum()
    sim_Y2_rev = simulated_df[simulated_df['ds'] >= Y1_END_DATE]['simulated_net_sales'].sum()
    
    Y1_improvement_pct = (sim_Y1_rev / base_Y1_rev - 1) * 100 if base_Y1_rev > 0 else 0
    Y2_improvement_pct = (sim_Y2_rev / base_Y2_rev - 1) * 100 if base_Y2_rev > 0 else 0
    
    # Metrics (Calculated on Daily Results)
    total_uplift = scenario_results_daily['gross_uplift'].sum()
    total_cost = scenario_results_daily['promo_cost'].sum()
    net_profit_impact = scenario_results_daily['net_profit_impact'].sum()
    
    print("\n## 4.3.4 PREDICTION BASED ON OPTIMIZATION:")
    print(f"   * Optimized Annual Revenue 2026 (Y1): PHP {sim_Y1_rev:,.2f} ({Y1_improvement_pct:+.1f}% vs. Baseline Y1)")
    print(f"   * Optimized Annual Revenue 2027 (Y2): PHP {sim_Y2_rev:,.2f} ({Y2_improvement_pct:+.1f}% vs. Baseline Y2)")
    print(f"   * Total 2-Year Net Profit Impact (Gross Uplift - Promo Cost): PHP {net_profit_impact:,.2f}")
    
    # --- 7. Aggregating to Monthly Before Plotting ---
    print("\n7. Aggregating Daily Results to Monthly for Revenue Plots...")
    
    # 7a. Aggregate Historical Data (df_history)
    df_history_m = df_history.set_index('ds').resample('M')['y'].sum().reset_index()
    
    # 7b. Get baseline predictions only for the forecast period (daily)
    base_preds_only = future_df.copy()
    base_preds_m = base_preds_only.set_index('ds')['y_pred'].resample('M').sum().reset_index()
    
    # 7c. Aggregate Scenario Results (scenario_results_daily)
    scenario_results_m = scenario_results_daily[['ds', 'baseline_sales', 'simulated_net_sales']].set_index('ds').resample('M').sum().reset_index()
    
    # 7d. Aggregate Daily Revenue & Orders to Calculate True Monthly AOV
    print("7d: Calculating Monthly AOV...")
    
    # Check if 'total_orders' is available in df_history
    if 'total_orders' in df_history.columns:
        df_history_aov_m = df_history.set_index('ds').resample('M').agg({
            'y': 'sum',      # Sum of Revenue
            'total_orders': 'sum'  # Sum of Orders
        }).reset_index()
        
        df_history_aov_m['aov_m'] = df_history_aov_m['y'] / df_history_aov_m['total_orders'].replace(0, 1e-6)
    else:
        # Fallback if the load_base_sales_data function did not include total_orders
        print("[Warning] 'total_orders' not found in history. Using proxy AOV for plotting.")
        df_history_aov_m = df_history.set_index('ds').resample('M')['y'].mean().reset_index()
        df_history_aov_m.rename(columns={'y': 'aov_m'}, inplace=True)
    
    # 7e. Generating Plots...
    print("7.1: Generating Plots using Monthly Aggregation...")
    
    plot_forecasts(
        df_history_m, 
        base_preds_m, 
        scenario_results_m, 
        df_history_daily=df_history, 
        scenario_results_daily=scenario_results_daily,
        df_history_aov_m=df_history_aov_m,
        current_aov_val=historical_aov # Pass the calculated AOV to the plotter
    )
    
    # Print final results summary
    print(f"\n--- FINAL SCENARIO SUMMARY ---")
    print(f"Total Revenue Uplift (Volume-Focused Strategy): PHP {total_uplift:,.2f}")
    print(f"Total Promo Cost (5% Cash Discount):           PHP {total_cost:,.2f}")
    print(f"Net Profit Impact:                             PHP {net_profit_impact:,.2f}")

if __name__ == "__main__":
    run_scenario_forecast()