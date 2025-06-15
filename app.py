# Import libraries
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
from dataCleaning import load_data as load_clean_data, clean_data
from forecast_model import train_and_forecast
import pickle

# Set page configuration
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    html {
        color-scheme: light !important;
    }
    </style>
""", unsafe_allow_html=True)


# Custom CSS for professional styling
st.markdown("""
<style>
    :root {
        --primary: #2e7d32;
        --secondary: #43cea2;
        --accent: #185a9d;
        --light: #f8f9fa;
        --dark: #212529;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header-text {
        color: var(--primary);
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    
    .subheader {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .feature-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
        padding: 25px;
        margin-bottom: 25px;
        transition: transform 0.3s, box-shadow 0.3s;
        border: 1px solid #e0e0e0;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.12);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, var(--secondary) 0%, var(--accent) 100%);
        color: white;
        border-radius: 15px;
        padding: 35px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        margin: 25px 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--secondary) 0%, var(--accent) 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
        font-size: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .stSlider .thumb {
        background: var(--secondary) !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px !important;
        background: white !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--secondary) 0%, var(--accent) 100%) !important;
        color: white !important;
        font-weight: 600;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary);
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    
    .forecast-header {
        background: linear-gradient(135deg, #5c6bc0 0%, #3949ab 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    
    .insight-box {
        background: white;
        border-left: 5px solid var(--secondary);
        border-radius: 8px;
        padding: 15px 20px;
        margin: 15px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    
    .forecast-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    
    .forecast-highlight {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--accent);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
# Update your load_data function
@st.cache_data
def load_data():
    df = pd.read_csv("climate_change_impact_on_agriculture_2024.csv")
    # Apply same preprocessing as training
    df.rename(columns={'Irrigation_Access_%': 'Irrigation_Access_percent'}, inplace=True)
    
    # Drop unnecessary columns to match training
    columns_to_drop = ['Economic_Impact_Million_USD', 'Region', 'Adaptation_Strategies']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=col)
    
    return df

# Load trained model (dummy model for demo)
@st.cache_resource
def load_model():
    with open("lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Load expected feature names
def load_feature_names(filepath='model_features.txt'):
    with open(filepath, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    return features


# Load and clean data for forecasting
@st.cache_data
def load_and_clean_forecast():
    # Load data using the same file path as the main app
    df = load_data()
    # Clean data - this function should be implemented in dataCleaning.py
    df_clean = clean_data(df)
    return df_clean

def main():
    # App header
    st.markdown('<h1 class="header-text">🌾 Advanced Crop Yield Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="subheader">
        Predict, analyze, and forecast agricultural yields under changing climate conditions
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    
    # Load data and model
    model = load_model()
    df = load_data()
    expected_features = load_feature_names()
    
    # Get target column
    target_col = "Crop_Yield_MT_per_HA"
    if target_col in df.columns:
        target_mean = df[target_col].mean()
    else:
        st.error(f"Target column '{target_col}' not found in dataset!")
        st.stop()
    
    # Create tabs with updated order
    tab1, tab2, tab3 = st.tabs(["🌱 Prediction", "🔮 Forecasting", "📊 Data Explorer"])
    
    # ===================== PREDICTION TAB =====================
    with tab1:
        st.markdown("### Climate Impact Simulation")
        st.markdown("Adjust parameters to predict crop yields under different climate scenarios")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Remove the empty st.markdown calls and use proper spacing
            st.markdown("#### Climate Parameters")
    
            # Add a small spacer instead of empty markdown
            st.write("") 
            
            # Climate parameters
            avg_temp = st.slider("Average Temperature (°C)", 10.0, 45.0, 20.0, 0.5)
            precipitation = st.slider("Total Precipitation (mm)", 300.0, 2000.0, 600.0, 10.0)
            extreme_events = st.slider("Extreme Weather Events", 0, 5, 1)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### Soil Parameters")
            
            # Soil parameters
            soil_health = st.slider("Soil Health Index", 1.0, 10.0, 7.0, 0.1)
            pesticide_use = st.slider("Pesticide Use (kg/ha)", 0.1, 2.5, 1.0, 0.1)
            fertilizer_use = st.slider("Fertilizer Use (kg/ha)", 50.0, 200.0, 100.0, 5.0)
            irrigation = st.slider("Irrigation Access (%)", 50.0, 100.0, 75.0, 5.0)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Location & Crop")

            # Get unique countries and crops from your dataframe
            countries = df['Country'].unique().tolist()
            crops = df['Crop_Type'].unique().tolist()

            # Select country and crop with default selections
            country = st.selectbox("Country", countries, index=countries.index('India') if 'India' in countries else 0)
            crop = st.selectbox("Crop Type", crops, index=crops.index('Rice') if 'Rice' in crops else 0)

            st.markdown('</div>', unsafe_allow_html=True)

            # Update the prediction button section
            if st.button("Predict Yield", use_container_width=True, key="predict_button"):
                try:
                    # One-hot encode Country and Crop_Type flags
                    country_flags = {f"Country_{c}": 1 if c == country else 0 for c in countries}
                    crop_flags = {f"Crop_Type_{c}": 1 if c == crop else 0 for c in crops}

                    # Build input dictionary for model with all expected features initialized to 0
                    input_data = {
                        'Average_Temperature_C': avg_temp,
                        'Total_Precipitation_mm': precipitation,
                        'Extreme_Weather_Events': extreme_events,
                        'Irrigation_Access_percent': irrigation,
                        'Pesticide_Use_KG_per_HA': pesticide_use,
                        'Fertilizer_Use_KG_per_HA': fertilizer_use,
                        'Soil_Health_Index': soil_health,
                        # Initialize all expected features to 0
                        **{feature: 0 for feature in expected_features if feature not in [
                            'Average_Temperature_C',
                            'Total_Precipitation_mm',
                            'Extreme_Weather_Events',
                            'Irrigation_Access_percent',
                            'Pesticide_Use_KG_per_HA',
                            'Fertilizer_Use_KG_per_HA',
                            'Soil_Health_Index'
                        ]}
                    }

                    # Set the selected country and crop flags
                    input_data.update(country_flags)
                    input_data.update(crop_flags)

                    # Convert input to DataFrame
                    input_df = pd.DataFrame([input_data])

                    # Reorder columns exactly as model expects
                    input_df = input_df[expected_features]

                    # Calculate base yield from historical data for context
                    base_yield = df[
                        (df['Country'] == country) &
                        (df['Crop_Type'] == crop)
                    ]['Crop_Yield_MT_per_HA'].mean()

                    if np.isnan(base_yield):
                        base_yield = df['Crop_Yield_MT_per_HA'].mean()

                    # Predict yield with loaded model
                    prediction = model.predict(input_df)[0]
                    st.session_state.prediction = prediction

                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.error(f"Expected features: {expected_features}")
                    st.error(f"Provided features: {list(input_df.columns)}")

            # Show prediction results if available
            if st.session_state.get('prediction') is not None:
                st.markdown(f"<h2>🌾 Predicted Yield: {st.session_state.prediction:.2f} MT/HA</h2>",
                            unsafe_allow_html=True)

                diff = st.session_state.prediction - target_mean
                pct_diff = (diff / target_mean) * 100

                st.markdown(f"<p>Compared to global average: {diff:+.2f} MT/HA ({pct_diff:+.1f}%)</p>",
                            unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                if diff > 0:
                    st.success(f"✅ This yield is **{pct_diff:.1f}% above** the global average for {crop}")
                else:
                    st.warning(f"⚠️ This yield is **{abs(pct_diff):.1f}% below** the global average for {crop}")

            # Footnote about the model
            st.markdown("---")
            st.markdown(
                        """
                        <div style="padding: 0.5rem 1rem; border-left: 3px solid #666; font-size: 0.85rem; color: #222; margin-top: 2rem;">
                            <strong>Model Summary:</strong>
                            This predictive model is based on <strong>LightGBM</strong>, a gradient boosting framework that handles large-scale, mixed-type datasets.
                            It was trained on 10,000 agricultural records spanning <strong>30 years</strong> across <strong>10 countries</strong>.<br><br>
                            On the validation set, it achieved a Mean Absolute Error (MAE) of 0.23 MT/HA and a <strong>R²</strong> of <strong>0.91</strong>, better than random Forest and regression models.
                        </div>
                        """, unsafe_allow_html=True
)
    
    # ===================== FORECASTING TAB (NOW TAB 2) =====================
    # ===================== FORECASTING TAB =====================
# ===================== FORECASTING TAB =====================
    with tab2:
        st.markdown("### 🌾 Crop Yield Forecasting (2025–2050)")
        st.markdown("Long-term yield predictions based on historical trends and climate models")
        
        # Load and clean data for forecasting
        df_clean = load_and_clean_forecast()
        
        # Create columns for inputs
        col1, col2 = st.columns(2)
        
        # Country selection
        with col1:
            countries = df_clean['Country'].unique().tolist()
            selected_country = st.selectbox("Select Country", countries, key="forecast_country")
        
        # Initialize forecast to None
        forecast = None
        
        # Train model and forecast for selected country
        try:
            with st.spinner(f"Generating forecast for {selected_country}..."):
                df_agg, forecast, model = train_and_forecast(df_clean, selected_country)
        except Exception as e:
            st.error(f"Forecast generation failed: {e}")
            st.stop()
        
        # Only proceed if forecast was successfully created
        if forecast is not None:
            # Year selection
            with col2:
                forecast_years = forecast['year'].unique().tolist()  # Use 'year' column
                selected_year = st.selectbox("Select Year", forecast_years, key="forecast_year")
            
            # Filter forecast for selected year
            selected_forecast = forecast[forecast['year'] == selected_year]
            
            if not selected_forecast.empty:
                # UPDATED: Use 'year' column instead of 'date'
                pred = selected_forecast['prediction'].values[0]
                lower = selected_forecast['lower'].values[0]
                upper = selected_forecast['upper'].values[0]
                
                # Display forecast in styled card
                st.markdown(f"<h3>Forecast for {selected_country} in {selected_year}</h3>", unsafe_allow_html=True)
                st.markdown(f'<div class="forecast-highlight">{pred:.2f} MT/HA</div>', unsafe_allow_html=True)
                st.markdown(f"**95% Confidence Interval**: {lower:.2f} – {upper:.2f} MT/HA")
                
                # Historical context
                if 'Crop_Yield_MT_per_HA' in df_agg.columns:
                    last_historical = df_agg['Crop_Yield_MT_per_HA'].iloc[-1]
                    change = ((pred - last_historical) / last_historical) * 100
                    
                    if change > 0:
                        trend = f"📈 Increase of {change:.1f}% compared to last historical data"
                        st.success(trend)
                    else:
                        trend = f"📉 Decrease of {abs(change):.1f}% compared to last historical data"
                        st.warning(trend)
            
                # Forecast table expander
                with st.expander("📋 View Full Forecast Data Table"):
                    display_df = forecast[['year', 'prediction', 'lower', 'upper']].copy()
                    st.dataframe(display_df, use_container_width=True)
                
                # Visualize forecast
                st.markdown("### Forecast Trend")
                fig = px.line(
                    forecast, 
                    x='year',  # Use year instead of date
                    y='prediction',
                    title=f"Yield Forecast for {selected_country} (2025–2050)",
                    labels={'year': 'Year', 'prediction': 'Predicted Yield (MT/HA)'}
                )
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Add footnote about the forecasting model
            st.markdown("---")
            st.markdown("""
            <div style="font-size: 0.85rem; color: #666; margin-top: 2rem;">
                <strong>⏳ Forecasting Methodology:</strong> Forecasts are generated using ARIMA models with exogenous variables, 
                which provide uncertainty estimates through confidence intervals. This statistical approach effectively models 
                time series patterns while incorporating climate covariates like temperature, precipitation, irrigation access, 
                fertilizer use, and soil health. The 95% confidence intervals represent the range of probable yield outcomes.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No forecast data was generated. Please try again.")

 
   # ===================== DATA EXPLORER TAB =====================
    with tab3:
        st.markdown("### 📊 Data Explorer")
        st.markdown("Interactive exploration of agricultural data")
        
        # Reorder tabs with Yield Trends first
        tab_ex2, tab_ex1, tab_ex3 = st.tabs(["📈 Yield Trends", "📦 Numerical Analysis", "🌍 Categorical Proportions"])
        
        # ========== YIELD TRENDS TAB (NOW FIRST) ==========
        with tab_ex2:
            st.markdown("#### Crop Yield Trends Over Time")
            st.markdown("Explore historical yield patterns by country")
            
            # Country selection for trends
            countries = df['Country'].unique().tolist()
            selected_countries = st.multiselect(
                "Select Countries", 
                countries, 
                default=['USA', 'India', 'China', 'Brazil'],
                key="trend_countries"
            )
            
            # Filter data
            if selected_countries:
                trend_df = df[df['Country'].isin(selected_countries)]
                
                # Aggregate by year and country
                agg_trend = trend_df.groupby(['Year', 'Country'])['Crop_Yield_MT_per_HA'].mean().reset_index()
                
                # Line chart
                fig = px.line(
                    agg_trend, 
                    x='Year', 
                    y='Crop_Yield_MT_per_HA', 
                    color='Country',
                    title='Crop Yield Trends by Country',
                    markers=True,
                    height=500
                )
                fig.update_layout(yaxis_title="Average Yield (MT/HA)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Top and bottom countries section
                st.markdown("#### Top and Bottom Performing Countries")
                
                # Calculate country averages
                country_avg = df.groupby('Country')['Crop_Yield_MT_per_HA'].mean().reset_index()
                top_countries = country_avg.nlargest(5, 'Crop_Yield_MT_per_HA')
                bottom_countries = country_avg.nsmallest(5, 'Crop_Yield_MT_per_HA')
                
                # Create columns for top and bottom
                col_top, col_bottom = st.columns(2)
                
                with col_top:
                    st.markdown("##### 🏆 Top 5 Countries")
                    top_countries['Rank'] = range(1, 6)
                    top_countries = top_countries[['Rank', 'Country', 'Crop_Yield_MT_per_HA']]
                    top_countries.columns = ['Rank', 'Country', 'Avg Yield']
                    st.dataframe(top_countries, use_container_width=True, hide_index=True)
                    
                    # Bar chart for top countries with wider bars
                    fig_top = px.bar(
                        top_countries,
                        x='Country',
                        y='Avg Yield',
                        color='Country',
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        title='Highest Yielding Countries'
                    )
                    # Increase bar width
                    fig_top.update_traces(width=0.8)
                    fig_top.update_layout(
                        showlegend=False,
                        bargap=0.15  # Reduce gap between bars
                    )
                    st.plotly_chart(fig_top, use_container_width=True)
                
                with col_bottom:
                    st.markdown("##### ⚠️ Bottom 5 Countries")
                    bottom_countries['Rank'] = range(1, 6)
                    bottom_countries = bottom_countries[['Rank', 'Country', 'Crop_Yield_MT_per_HA']]
                    bottom_countries.columns = ['Rank', 'Country', 'Avg Yield']
                    st.dataframe(bottom_countries, use_container_width=True, hide_index=True)
                    
                    # Bar chart for bottom countries with wider bars
                    fig_bottom = px.bar(
                        bottom_countries,
                        x='Country',
                        y='Avg Yield',
                        color='Country',
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        title='Lowest Yielding Countries'
                    )
                    # Increase bar width
                    fig_bottom.update_traces(width=0.8)
                    fig_bottom.update_layout(
                        showlegend=False,
                        bargap=0.15  # Reduce gap between bars
                    )
                    st.plotly_chart(fig_bottom, use_container_width=True)
            else:
                st.info("Please select at least one country to view trends")
        
        # ========== NUMERICAL ANALYSIS TAB ==========
        with tab_ex1:
            st.markdown("#### Numerical Data Distribution")
            st.markdown("Boxplots showing distribution of numeric variables")
            
            # Select numerical column
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected_num = st.selectbox("Select Numerical Variable", num_cols, key="num_var")
            
            # Boxplot
            fig = px.box(df, y=selected_num, color_discrete_sequence=['#2e7d32'])
            fig.update_layout(title=f"Distribution of {selected_num}", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("##### Summary Statistics")
            stats = df[selected_num].describe().reset_index()
            stats.columns = ['Metric', 'Value']
            st.dataframe(stats, use_container_width=True, hide_index=True)
        
        # ========== CATEGORICAL PROPORTIONS TAB ==========
        with tab_ex3:
            st.markdown("#### Categorical Proportions by Country")
            st.markdown("Distribution of crop types and adaptation strategies")
            
            # Select categorical column
            cat_cols = ['Crop_Type']
            selected_cat = st.selectbox("Select Categorical Variable", cat_cols, key="cat_var")
            
            # Select country
            cat_country = st.selectbox("Select Country", countries, key="cat_country")
            
            # Filter data
            cat_df = df[df['Country'] == cat_country]
            
            if not cat_df.empty:
                # Calculate proportions
                proportions = cat_df[selected_cat].value_counts(normalize=True).reset_index()
                proportions.columns = [selected_cat, 'Proportion']
                
                # Create columns for charts
                col_pie, col_bar = st.columns(2)
                
                with col_pie:
                    # Pie chart
                    fig_pie = px.pie(
                        proportions,
                        names=selected_cat,
                        values='Proportion',
                        title=f'{selected_cat} Distribution in {cat_country}',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_bar:
                    # Bar chart with wider bars
                    fig_bar = px.bar(
                        proportions,
                        x=selected_cat,
                        y='Proportion',
                        color=selected_cat,
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        title=f'{selected_cat} Proportions in {cat_country}'
                    )
                    # Increase bar width
                    fig_bar.update_traces(width=0.8)
                    fig_bar.update_layout(
                        showlegend=False,
                        bargap=0.15  # Reduce gap between bars
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Detailed table
                st.markdown(f"##### Detailed {selected_cat} Distribution")
                st.dataframe(proportions, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No data available for {cat_country}")

# Run main
if __name__ == "__main__":
    main()