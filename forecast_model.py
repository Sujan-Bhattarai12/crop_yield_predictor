import pandas as pd
import pickle
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

def train_and_forecast(df_clean, selected_country, save_model_path="trained_arima_model.pkl"):
    # Filter data by country
    df_country = df_clean[df_clean['Country'] == selected_country]

    # Aggregate annually
    df_agg = df_country.groupby('Year').agg({
        'Crop_Yield_MT_per_HA': 'mean',
        'Average_Temperature_C': 'mean',
        'Total_Precipitation_mm': 'mean',
        'Irrigation_Access_percent': 'mean',
        'Fertilizer_Use_KG_per_HA': 'mean',
        'Soil_Health_Index': 'mean'
    }).reset_index()

    df_agg['date'] = pd.to_datetime(df_agg['Year'], format='%Y')
    df_agg = df_agg.rename(columns={
        'Crop_Yield_MT_per_HA': 'y',
        'Average_Temperature_C': 'temp',
        'Total_Precipitation_mm': 'precip',
        'Irrigation_Access_percent': 'irrigation',
        'Fertilizer_Use_KG_per_HA': 'fertilizer',
        'Soil_Health_Index': 'soil_health'
    }).sort_values('date')

    # Prepare data for ARIMA
    y = df_agg['y']
    exog = df_agg[['temp', 'precip', 'irrigation', 'fertilizer', 'soil_health']]
    
    # Fit ARIMA model with exogenous variables
    model = SARIMAX(
        y,
        exog=exog,
        order=(1, 1, 1),           # Basic ARIMA order (p,d,q)
        seasonal_order=(0, 0, 0, 0),  # No seasonal component
        trend='c'
    )
    model_fit = model.fit(disp=False)

    # Save model
    with open(save_model_path, "wb") as f:
        pickle.dump(model_fit, f)

    # Forecast future years
    forecast_years = [2025, 2030, 2035, 2040, 2045, 2050]
    size = len(forecast_years)

    future_df = pd.DataFrame({
        'date': pd.to_datetime(forecast_years, format='%Y'),
        'temp': [df_agg['temp'].mean()] * size,
        'precip': [df_agg['precip'].mean()] * size,
        'irrigation': [df_agg['irrigation'].mean()] * size,
        'fertilizer': [df_agg['fertilizer'].mean()] * size,
        'soil_health': [df_agg['soil_health'].mean()] * size,
    }).set_index('date')

    # Generate forecasts with 95% confidence intervals
    forecast = model_fit.get_forecast(
        steps=size,
        exog=future_df[['temp', 'precip', 'irrigation', 'fertilizer', 'soil_health']]
    )
    
    # Create results DataFrame
    forecast_df = pd.DataFrame({
        'date': future_df.index,
        'prediction': forecast.predicted_mean,
        'lower': forecast.conf_int()['lower y'],
        'upper': forecast.conf_int()['upper y']
    })
    forecast_df['year'] = forecast_df['date'].dt.year

    return df_agg, forecast_df, model_fit