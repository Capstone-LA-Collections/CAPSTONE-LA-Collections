import pandas as pd
import numpy as np
import xgboost as xgb

def create_xgb_features(df, residuals=None):
    """
    Creates time-based features + Lagged Residuals.
    """
    df = df.copy()
    if not np.issubdtype(df['ds'].dtype, np.datetime64):
        df['ds'] = pd.to_datetime(df['ds'])
        
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['day_of_year'] = df['ds'].dt.dayofyear
    df['year'] = df['ds'].dt.year
    df['is_quarter_end'] = df['ds'].dt.is_quarter_end.astype(int)
    
    # If residuals are provided (Training Phase), create the lag immediately
    if residuals is not None:
        df['residuals'] = residuals
        df['resid_lag_1'] = df['residuals'].shift(1)
        
    return df

def run_hybrid_forecast(base_forecast, historical_residuals_df, future_df, platform_name):
    """
    1. Trains XGBoost on Historical Residuals (with Lag).
    2. Iteratively predicts future residuals (updating Lag step-by-step).
    """
    train_df = create_xgb_features(historical_residuals_df, residuals=historical_residuals_df['residuals'])
    train_df = train_df.dropna() 

    features = ['day_of_week', 'month', 'day_of_year', 'year', 'resid_lag_1']
    X_train = train_df[features]
    y_train = train_df['residuals']
    
    if len(X_train) < 50: # Adjusting minimum size for robust splitting
        print("   [Hybrid Warning] Not enough history for XGBoost + Early Stopping.")
        # ... (return 0s logic) ...

    # --- 2. Create Time-Series Validation Split for Early Stopping ---
    EARLY_STOPPING_ROUNDS = 25
    VAL_SIZE = 0.15 # Use 15% of historical data for validation

    val_split_index = int(len(X_train) * (1 - VAL_SIZE))

    # Actual Training Data (Older)
    X_train_actual = X_train.iloc[:val_split_index]
    y_train_actual = y_train.iloc[:val_split_index]
    
    # Validation Data (Newer)
    X_val = X_train.iloc[val_split_index:]
    y_val = y_train.iloc[val_split_index:]

    # Define the evaluators: one for training set, one for validation set (the second is used for stopping)
    eval_set = [(X_train_actual, y_train_actual), (X_val, y_val)]
    
    # --- 3. Initialize and Train XGBoost with Early Stopping ---

    if platform_name == 'shopee':
        estimators = 7800
    else:
        estimators = 5500
    
    gbm_reg = xgb.XGBRegressor(
        # pwede galawin
        n_estimators=estimators,
        learning_rate=0.1,
        max_depth=5,
        reg_alpha=0.1,
        reg_lambda=0.5,
        # di na gagalawin
        n_jobs=-1,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        eval_metric="rmse",
        verbose=False 
    )

    # The fit call now includes the early stopping parameters
    gbm_reg.fit(
        X_train_actual,
        y_train_actual,
        eval_set=eval_set
    )
    
    # Use the trained model (stopped early) to predict residuals for the full historical period
    fitted_residual_preds = gbm_reg.predict(X_train)
    fitted_residual_preds_series = pd.Series(
        fitted_residual_preds, 
        index=train_df.index
    )
    
    # --- 3. Iterative Prediction (The Loop) ---
    # We cannot predict all at once because Day 2 needs the prediction from Day 1.
    
    future_xgb = create_xgb_features(future_df)
    future_residuals = []
    
    # Initial Lag: The last known residual from history
    current_lag = historical_residuals_df['residuals'].iloc[-1]
    
    # Iterate day by day
    for i in range(len(future_xgb)):
        # Get features for this specific day
        row = future_xgb.iloc[[i]].copy()
        
        # Inject the known lag
        row['resid_lag_1'] = current_lag
        
        # Predict
        pred_resid = gbm_reg.predict(row[features])[0]
        
        # Store prediction
        future_residuals.append(pred_resid)
        
        # Update lag for the NEXT day
        current_lag = pred_resid

    future_residuals = np.array(future_residuals)

    # --- 4. Combine ---
    # Base Forecast + XGBoost Residual Correction
    # Ensure base_forecast is numpy array
    base_values = base_forecast.values if hasattr(base_forecast, 'values') else base_forecast
    
    hybrid_values = base_values + future_residuals
    
    # Clip negatives (Business Logic)
    hybrid_values = np.maximum(hybrid_values, 0)
    
    return hybrid_values, fitted_residual_preds_series