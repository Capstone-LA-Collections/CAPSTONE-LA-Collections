import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---
FORECAST_DAYS = 730
# Define the features XGBoost was trained on
FEATURES = ['is_mega_sale_day', 'is_payday', 'is_weekend', 'month', 'day_of_week', 'scaled_discount_rate', 'resid_lag_1']

# =========================================================================
# 1. DATA LOADING AND PREPARATION
# =========================================================================
print("1. Loading and Preparing Data...")
df = pd.read_csv("Lazada_hist.csv")
df['ds'] = pd.to_datetime(df['ds'])
prophet_df = df[['ds', 'y']].sort_values(by='ds')

# Define the historical training cutoff (using all but the last 60 days for a strong base model)
cutoff_date = prophet_df['ds'].max() - pd.Timedelta(days=60)
train_prophet = prophet_df[prophet_df['ds'] <= cutoff_date].copy()

# --- Feature Engineering Function (Used for Historical and Future Data) ---
def create_xgb_features(data, residuals=None):
    # Ensure 'ds' is datetime
    data['ds'] = pd.to_datetime(data['ds'])
    data['dayofweek'] = data['ds'].dt.dayofweek
    data['dayofyear'] = data['ds'].dt.dayofyear
    data['month'] = data['ds'].dt.month
    data['year'] = data['ds'].dt.year
    
    # Add Lagged Residuals if provided
    if residuals is not None:
        data['Residuals'] = residuals
        data['resid_lag_1'] = data['Residuals'].shift(1)
        data = data.dropna(subset=['resid_lag_1'])
    return data

# =========================================================================
# 2. PROPHET BASE MODEL TRAINING AND RESIDUAL CALCULATION
# =========================================================================
print("2. Training Prophet Model and Calculating Residuals...")
m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.fit(train_prophet)

# Predict on the entire historical dataset to get residuals
full_forecast = m.predict(prophet_df[['ds']])
prophet_df['Prophet_yhat'] = full_forecast['yhat']
prophet_df['Residuals'] = prophet_df['y'] - prophet_df['Prophet_yhat']

# =========================================================================
# 3. XGBOOST TRAINING ON RESIDUALS
# =========================================================================
print("3. Training XGBoost Model on Historical Residuals...")

# Merge residuals back into the feature-rich dataframe and create lag
feature_df = df.merge(prophet_df[['ds', 'Residuals', 'Prophet_yhat']], on='ds', how='left')
feature_df = create_xgb_features(feature_df.copy(), residuals=feature_df['Residuals'])
feature_df = feature_df.dropna() # Drop the first row due to the lag feature

# Define Train Data for XGBoost (aligned with Prophet training cutoff)
X_train = feature_df[feature_df['ds'] <= cutoff_date][FEATURES]
y_train = feature_df[feature_df['ds'] <= cutoff_date]['Residuals']
val_size = max(1, int(len(X_train) * 0.2))
X_valid, y_valid = X_train.iloc[-val_size:], y_train.iloc[-val_size:]

# Train XGBoost
xgb = XGBRegressor(n_estimators=500, learning_rate=0.03, 
                   random_state=42, tree_method='auto',
                   objective='reg:absoluteerror',max_depth=2, 
                   reg_alpha=0.1, reg_lambda=0.1,
                   early_stopping_rounds=25, eval_metric='rmse')

xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)])

# =========================================================================
# 3.5 HISTORICAL TEST SET EVALUATION (RMSE Checkpoint)
# =========================================================================
print("3.5 Evaluating Model Performance on Historical Test Set...")

# 3.5.1 Define Test Data and Actuals
# X_test and y_test were defined based on the feature_df > cutoff_date
X_test = feature_df[feature_df['ds'] > cutoff_date][FEATURES]

# Actual sales values and Prophet base prediction for the test period (aligned due to lag drop)
actual_test_values_aligned = feature_df[feature_df['ds'] > cutoff_date]['y'].values
prophet_test_yhat_aligned = feature_df[feature_df['ds'] > cutoff_date]['Prophet_yhat'].values

# 3.5.2 Predict residuals on the test set using the trained XGBoost model
# We must ensure X_test features are float (similar to the fix for the future loop)
xgb_test_pred_residuals = xgb.predict(X_test.astype(float))

# 3.5.3 Calculate Final Hybrid Forecast on Test Set
final_hybrid_forecast_test = prophet_test_yhat_aligned + xgb_test_pred_residuals

# 3.5.4 Calculate RMSE
# Prophet Base RMSE (Actual vs. Prophet_yhat)
prophet_rmse = mean_squared_error(actual_test_values_aligned, prophet_test_yhat_aligned, squared=False)
# Final Hybrid RMSE (Actual vs. Hybrid_Forecast)
final_rmse = mean_squared_error(actual_test_values_aligned, final_hybrid_forecast_test, squared=False)

print("\n--- Model Performance on Historical Test Set ---")
print(f"Prophet Base RMSE: {prophet_rmse:,.2f}")
print(f"Prophet + XGBoost Hybrid RMSE: {final_rmse:,.2f}")
print(f"RMSE Improvement: {((prophet_rmse - final_rmse) / prophet_rmse) * 100:.2f}%")
print("---------------------------------------------")

# The code will then proceed to Section 4 for the future prediction.

# =========================================================================
# 4. FUTURE FORECAST: Prophet and ITERATIVE XGBoost
# =========================================================================
print(f"4. Generating {FORECAST_DAYS}-Day Future Forecast (Iterative XGBoost)...")

# --- 4.1 Create Future Dataframe ---
last_date = prophet_df['ds'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='D')
future = pd.DataFrame({'ds': future_dates})

# --- 4.2 Populate Future External Features ---
# NOTE: These assumptions are placeholders. In a real scenario, you MUST provide these values.
future['dayofweek'] = future['ds'].dt.dayofweek
future['month'] = future['ds'].dt.month
future['day_of_week'] = future['dayofweek'] # Match the column name used in training
future['is_mega_sale_day'] = 0
future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
future['is_payday'] = future['ds'].dt.day.isin([15, 30]).astype(int)
future['scaled_discount_rate'] = 0.0

# --- 4.3 Prophet Prediction ---
prophet_future_forecast = m.predict(future)
future_xgb = future.merge(prophet_future_forecast[['ds', 'yhat']], on='ds')
future_xgb.rename(columns={'yhat': 'Prophet_yhat'}, inplace=True)

# --- 4.4 Initialize Lag ---
last_historical_residual = prophet_df['Residuals'].iloc[-1]
future_xgb['resid_lag_1'] = np.nan
future_xgb.loc[future_xgb.index[0], 'resid_lag_1'] = last_historical_residual

# --- Corrected Iterative XGBoost Prediction Loop ---
print("4.5 Corrected Iterative XGBoost Prediction Loop...")

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
        # Note: We must also ensure the updated 'resid_lag_1' is a float
        future_xgb.loc[future_xgb.index[i + 1], 'resid_lag_1'] = float(current_residual_pred)

future_xgb['XGBoost_Residuals'] = xgb_future_residuals

# --- 4.6 Final Hybrid Forecast ---
future_xgb['Final_Hybrid_Forecast'] = future_xgb['Prophet_yhat'] + future_xgb['XGBoost_Residuals']
print("Forecast generation complete.")

# =========================================================================
# 5. VISUALIZATION
# =========================================================================
print("5. Plotting Forecast...")
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

plt.title(f'Prophet vs. Hybrid Forecast (30 Days Beyond Historical Data)')
plt.xlabel('Date')
plt.ylabel('Sales (y)')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()

print("\n--- DONE ---")
print("The final forecast data is in the 'future_xgb' DataFrame.")