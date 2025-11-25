import pandas as pd
import numpy as np
import logging
from datetime import date, timedelta
import calendar
# Removed the unstable 'from prophet import Prophet' and 'from prophet.utilities import seasonal_components'
# Relying only on standard imports for robustness.
from sklearn.preprocessing import StandardScaler

# --- STABLE FOURIER COMPONENT HELPER ---
def _add_fourier_components(df: pd.DataFrame, period: float, n_order: int, name: str):
    """
    Manually calculates Fourier components (sin/cos pairs) for a given period and order.
    This replaces the flaky prophet.utilities.seasonal_components.
    
    Args:
        df: DataFrame containing the 'days_since_start' column.
        period: The period of the seasonality (e.g., 30.5 for monthly).
        n_order: The number of components to generate (e.g., 4 for 8 features total).
        name: Prefix for the column names.
        
    Returns:
        pd.DataFrame: A DataFrame with the new Fourier columns added.
    """
    days = df['days_since_start'].values
    for k in range(1, n_order + 1):
        # Sine components
        df[f'{name}_fourier_{2*k - 1}'] = np.sin((2 * np.pi * k * days) / period)
        # Cosine components
        df[f'{name}_fourier_{2*k}'] = np.cos((2 * np.pi * k * days) / period)
    return df

def build_time_series_features(df, target_col, lag_periods=[1, 7, 14, 30, 60], rolling_window=14):
    """
    Builds time-series features with lagged, event, Fourier, trend, and volatility signals.
    Recursion-safe version: preserves continuity, avoids flattening during forecasting.
    
    CRITICAL UPDATE: Now uses stable, manual calculation for Fourier Monthly components.
    """

    df = df.copy().sort_values("ds").reset_index(drop=True)

    # ======================
    # 🧭 1. DATE FEATURES
    # ======================
    df["year"] = df["ds"].dt.year
    df["month"] = df["ds"].dt.month
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["dayofyear"] = df["ds"].dt.dayofyear
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)
    df["quarter"] = ((df["month"] - 1) // 3) + 1
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # CRITICAL for Fourier calculation
    df["days_since_start"] = (df["ds"] - df["ds"].min()).dt.days.astype(int) 

    # ======================
    # 🎉 2. EVENT FLAGS (Simplified for robust future filling)
    # ======================
    
    # We rely heavily on the robust Prophet holiday logic for Mega_Sale_Day and Payday
    # Placeholder columns are created here to ensure other feature engineering steps don't fail.
    
    if 'is_mega_sale_day' not in df.columns:
        df['is_mega_sale_day'] = df.apply(lambda row: 1 if (row['ds'].month, row['ds'].day) in [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(11,11),(12,12)] else 0, axis=1)

    if 'is_payday' not in df.columns:
        df['is_payday'] = df.apply(lambda row: 1 if row['ds'].day == 15 or row['ds'].day >= 28 else 0, axis=1)

    # ======================
    # 📈 3. LAG AND ROLLING FEATURES (Recursive, dependent on historical data)
    # ======================
    
    # Target value for lag/rolling features
    df['daily_net_sales'] = df[target_col]
    
    # Lag Features (Past sales count)
    for lag in lag_periods:
        # Use shift(-lag) to align lag_n_net_sales with the date when sales *occurred*
        df[f'lag_{lag}_net_sales'] = df['daily_net_sales'].shift(lag)

    # Rolling Mean Features (Averaged past sales)
    df[f'rolling_7_net_sales'] = df['daily_net_sales'].shift(1).rolling(window=7, min_periods=1).mean()
    df[f'rolling_{rolling_window}_net_sales'] = df['daily_net_sales'].shift(1).rolling(window=rolling_window, min_periods=1).mean()


    # ======================
    # 💰 4. RAW GROWTH (NEW NON-RECURSIVE Feature)
    # ======================
    
    # CRITICAL: Use the daily sales column for calculating raw growth rate
    df['daily_sales_growth_smoothed'] = df['daily_net_sales'].pct_change().rolling(window=7, min_periods=1).mean()
    
    # New Raw Daily Growth Feature (Non-Recursive, as long as it's not lagged)
    # Note: Prophet is sensitive to NaNs in regressors, so fill the first day with 0.0.
    df['daily_growth_rate'] = df['daily_net_sales'].pct_change().fillna(0.0)
    df['daily_growth_rate'] = df['daily_growth_rate'].replace([np.inf, -np.inf], 0.0)

    # ======================
    # 📉 5. DISCOUNT & PRICE FEATURES (External, future-knowable)
    # ======================
    
    # We scale discount rates to prevent Prophet from over-relying on them
    if 'avg_discount_rate' in df.columns:
        scaler = StandardScaler()
        df['scaled_discount_rate'] = scaler.fit_transform(df[['avg_discount_rate']].fillna(0.0))
        
        # New Feature: scaled discount rate on event day
        df["scaled_discount_on_event"] = df["scaled_discount_rate"] * df["is_mega_sale_day"]
    else:
        # Placeholder for platforms without discount data
        df['avg_discount_rate_daily'] = 0.0 
        df['scaled_discount_rate'] = 0.0
        df["scaled_discount_on_event"] = 0.0

    # ======================
    # 🌊 6. FOURIER COMPONENTS (NOW STABLE)
    # ======================
    
    MONTHLY_COMPONENTS = 4
    MONTHLY_PERIOD = 30.5
    df = _add_fourier_components(df, period=7, n_order=3, name='fourier_week')
    df = _add_fourier_components(df, MONTHLY_PERIOD, MONTHLY_COMPONENTS, 'fourier_month')
    df = _add_fourier_components(df, period=365.25, n_order=10, name='fourier_year')

    # ======================
    # 🧹 7. FINAL CLEANUP AND DROPS
    # ======================
    
    # Fill any remaining NaNs in lag/rolling features with 0.0 (important for early history)
    lag_rolling_cols = [c for c in df.columns if c.startswith('lag_') or c.startswith('rolling_')]
    df[lag_rolling_cols] = df[lag_rolling_cols].fillna(0.0)
    
    # Drop intermediate columns
    df = df.drop(columns=[c for c in ['year', 'month', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter', 'days_since_start'] if c in df.columns], errors='ignore')    
    # Rename 'daily_net_sales' back to the target for the next step, but ensure 'ds' is present
    df = df.rename(columns={'daily_net_sales': target_col})

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return df