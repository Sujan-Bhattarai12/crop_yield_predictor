# forecast_model.py
import pandas as pd
from orbit.models import DLT

def train_and_forecast(df_clean, selected_country):
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
    })

    # Train model
    model = DLT(
        response_col='y',
        date_col='date',
        regressor_col=['temp', 'precip', 'irrigation', 'fertilizer', 'soil_health'],
        seasonality=1,
    )
    model.fit(df_agg)

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
    })

    forecast = model.predict(future_df)
    forecast['year'] = forecast['date'].dt.year

    return df_agg, forecast, model
