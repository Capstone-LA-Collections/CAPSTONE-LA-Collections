import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")

DEFAULT_PROPHET_PARAMS = {
    'lazada': {
        'changepoint_prior_scale': 0.1,
        'seasonality_prior_scale': 0.01,
        'seasonality_mode': 'additive',
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'yearly_seasonality': True
    },
    'shopee': {
        'changepoint_prior_scale': 0.5,
        'seasonality_prior_scale': 0.01,
        'seasonality_mode': 'additive',
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'yearly_seasonality': True
    }
}

class TimeSeriesTrainer:
    def __init__(self, df):
        self.df = df.copy()
        self.df['ds'] = pd.to_datetime(self.df['ds'])
        self.df = self.df.sort_values('ds').set_index('ds')
        self.df = self.df.asfreq('D').fillna(0)

    def get_exog(self, df_subset):
        # Dynamically grab all columns that are NOT 'y' or 'ds'
        # possible_cols = ['aov', 'promo_intensity', 'is_mega_sale_day', 'is_payday', 'is_weekend', 'units_per_order']
        possible_cols = ['aov', 'promo_intensity', 'is_mega_sale_day', 'is_payday', 'is_weekend']
        existing_cols = [c for c in possible_cols if c in df_subset.columns]
        return df_subset[existing_cols]

    def train_predict(self, model_name, forecast_horizon, future_exog=None, model_params=None, platform='default'):
        """
        model_params: Dict of hyperparameters to pass to the model (currently for Prophet).
        """
        train_y = self.df['y']
        train_exog = self.get_exog(self.df)
        
        preds = None
        fitted = None
        
        # Default empty dict if None
        if model_params is None:
            model_params = {}

        print(f"   Training {model_name}...")

        if model_name == 'SNaive':
            preds = []
            last_obs = train_y.iloc[-7:].values
            for i in range(forecast_horizon):
                preds.append(last_obs[i % 7])
            preds = pd.Series(preds)
            fitted = train_y.shift(7).fillna(method='bfill')

        elif model_name == 'Holt-Winters':
            try:
                model = ExponentialSmoothing(
                    train_y, 
                    seasonal_periods=7, 
                    trend='add', 
                    seasonal='add', 
                    damped_trend=True, 
                    initialization_method="estimated"
                ).fit()
                preds = model.forecast(forecast_horizon)
                fitted = model.fittedvalues
            except:
                model = ExponentialSmoothing(train_y, seasonal_periods=7, trend='add', seasonal='add').fit()
                preds = model.forecast(forecast_horizon)
                fitted = model.fittedvalues

        if model_name == 'Prophet':
            p_df = self.df.reset_index()[['ds', 'y']].copy()
            for col in train_exog.columns:
                p_df[col] = train_exog[col].values
            
            # --- 2. LOGIC TO LOAD AND MERGE DEFAULTS (NEW) ---
            
            # 2a. Load defaults based on platform
            # Use .get() to return an empty dict if the platform is not found, preventing errors
            default_params = DEFAULT_PROPHET_PARAMS.get(platform.lower(), {})
            
            # 2b. Merge defaults with passed params
            # Passed 'model_params' will override 'default_params' if keys overlap
            final_params = {**default_params, **model_params}
            
            # Fallback for daily_seasonality (required by Prophet if not specified)
            if 'daily_seasonality' not in final_params:
                final_params['daily_seasonality'] = True
                
            # --- APPLY FINAL PARAMS ---
            m = Prophet(**final_params)
            
            for col in train_exog.columns:
                m.add_regressor(col)
                
            m.fit(p_df)
            
            future = m.make_future_dataframe(periods=forecast_horizon)
            full_exog = pd.concat([train_exog, future_exog], axis=0).reset_index(drop=True)
            for col in train_exog.columns:
                future[col] = full_exog[col].values[:len(future)]

            forecast = m.predict(future)
            fitted = forecast.iloc[:len(train_y)]['yhat']
            fitted.index = self.df.index
            preds = forecast.iloc[len(train_y):]['yhat']

        else:
            # ARIMA / SARIMAX logic...
            seasonal = False
            m = 1
            exogenous = None
            
            if 'S' in model_name:
                seasonal = True
                m = 7
            
            if 'X' in model_name:
                exogenous = train_exog

            model = pm.auto_arima(
                train_y,
                X=exogenous,
                seasonal=seasonal,
                m=m,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
            fitted = pd.Series(model.predict_in_sample(X=exogenous), index=self.df.index)
            
            if 'X' in model_name and future_exog is not None:
                preds = model.predict(n_periods=forecast_horizon, X=future_exog)
            else:
                preds = model.predict(n_periods=forecast_horizon)

        return preds, fitted