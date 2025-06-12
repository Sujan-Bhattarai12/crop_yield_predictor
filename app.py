import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background-image: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
    }
    .header-text {
        color: #2e7d32;
        text-align: center;
        font-size: 2.5rem !important;
    }
    .feature-card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        color: white;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 6px 10px rgba(0,0,0,0.2);
        margin: 20px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stSlider .thumb {
        background: #43cea2 !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset to get feature ranges
@st.cache_data
def load_data():
    df = pd.read_csv("climate_change_impact_on_agriculture_2024.csv")
    return df

# Load trained model
@st.cache_resource
def load_model():
    model_path = "xgboost_model.json"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {os.path.abspath(model_path)}")
        st.stop()
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model

# Load expected feature names
def load_feature_names():
    path = "model_features.txt"
    if not os.path.exists(path):
        st.error("Missing 'model_features.txt'. This file should contain the list of features used to train the model.")
        st.stop()
    with open(path, "r") as f:
        return [line.strip() for line in f]

def main():
    # App header
    st.markdown('<h1 class="header-text">🌾 Crop Yield Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; margin-bottom:30px">
        <p style="font-size:1.1rem; color:#555;">
        Predict agricultural yields under changing climate conditions using machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    
    # Load data and model
    model = load_model()
    df = load_data()
    expected_features = load_feature_names()
    
    # Get target column and its mean BEFORE modifying the DataFrame
    target_col = "Crop_Yield_MT_per_HA"
    if target_col in df.columns:
        target_mean = df[target_col].mean()
        # Create a copy for prediction inputs without target column
        input_df = df.drop(columns=[target_col])
    else:
        st.error(f"Target column '{target_col}' not found in dataset!")
        st.stop()
    
    # Create tabs - removed Model Insights tab
    tab1, tab2 = st.tabs(["Prediction", "Data Explorer"])
    
    with tab1:
        st.header("Climate Impact Simulation")
        st.write("Adjust climate parameters to predict crop yield outcomes")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("Climate Parameters")
            input_dict = {}
            
            # Group features by category
            climate_features = [col for col in input_df.columns if 'temp' in col.lower() or 'precip' in col.lower()]
            soil_features = [col for col in input_df.columns if 'soil' in col.lower()]
            other_features = [col for col in input_df.columns if col not in climate_features + soil_features]
            
            # Climate features
            for col in climate_features:
                if input_df[col].dtype == object:
                    input_dict[col] = st.selectbox(f"{col}", sorted(input_df[col].dropna().unique()))
                else:
                    min_val = float(input_df[col].min())
                    max_val = float(input_df[col].max())
                    mean_val = float(input_df[col].mean())
                    input_dict[col] = st.slider(f"{col}", min_val, max_val, mean_val)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("Soil Parameters")
            # Soil features
            for col in soil_features:
                if input_df[col].dtype == object:
                    input_dict[col] = st.selectbox(f"{col}", sorted(input_df[col].dropna().unique()))
                else:
                    min_val = float(input_df[col].min())
                    max_val = float(input_df[col].max())
                    mean_val = float(input_df[col].mean())
                    input_dict[col] = st.slider(f"{col}", min_val, max_val, mean_val)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("Other Parameters")
            # Other features
            for col in other_features:
                if input_df[col].dtype == object:
                    input_dict[col] = st.selectbox(f"{col}", sorted(input_df[col].dropna().unique()))
                else:
                    min_val = float(input_df[col].min())
                    max_val = float(input_df[col].max())
                    mean_val = float(input_df[col].mean())
                    input_dict[col] = st.slider(f"{col}", min_val, max_val, mean_val)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Prediction button
            if st.button("Predict Yield", use_container_width=True, key="predict_button"):
                try:
                    # Convert input to DataFrame
                    input_df_pred = pd.DataFrame([input_dict])
                    
                    # One-hot encode the input
                    input_df_encoded = pd.get_dummies(input_df_pred)
                    
                    # Align with model's expected features
                    for col in expected_features:
                        if col not in input_df_encoded.columns:
                            input_df_encoded[col] = 0
                    input_df_encoded = input_df_encoded[expected_features]
                    
                    prediction = model.predict(input_df_encoded)[0]
                    st.session_state.prediction = prediction
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
            
            # Display prediction
            if st.session_state.prediction is not None:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.metric(label="Predicted Crop Yield", 
                          value=f"{st.session_state.prediction:.2f} MT/HA",
                          delta="Optimal" if st.session_state.prediction > target_mean else "Suboptimal")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Yield interpretation
                diff = st.session_state.prediction - target_mean
                pct_diff = (diff / target_mean) * 100
                
                if diff > 0:
                    st.success(f"This prediction is {pct_diff:.1f}% above average yield conditions")
                else:
                    st.warning(f"This prediction is {abs(pct_diff):.1f}% below average yield conditions")
    
    with tab2:
        st.header("Data Explorer")
        st.write("Explore aggregated insights from the agricultural dataset")
        
        # Show dataframe with filtering options
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Statistics
        st.subheader("Data Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Aggregated visualizations
        st.subheader("Aggregated Analysis")
        
        # Create columns for filters
        col1, col2, col3 = st.columns([1,1,1])
        
        with col1:
            # Get categorical columns for grouping
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols:
                st.warning("No categorical columns found for grouping!")
            else:
                group_col = st.selectbox("Group by", categorical_cols)
        
        with col2:
            # Get numerical columns for aggregation
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            value_col = st.selectbox("Aggregate", numerical_cols)
        
        with col3:
            agg_method = st.selectbox("Aggregation Method", 
                                     ["Sum", "Mean", "Median", "Max", "Min", "Count"])
        
        # Perform aggregation
        if 'group_col' in locals() and 'value_col' in locals():
            try:
                if agg_method == "Sum":
                    agg_df = df.groupby(group_col)[value_col].sum().reset_index()
                elif agg_method == "Mean":
                    agg_df = df.groupby(group_col)[value_col].mean().reset_index()
                elif agg_method == "Median":
                    agg_df = df.groupby(group_col)[value_col].median().reset_index()
                elif agg_method == "Max":
                    agg_df = df.groupby(group_col)[value_col].max().reset_index()
                elif agg_method == "Min":
                    agg_df = df.groupby(group_col)[value_col].min().reset_index()
                elif agg_method == "Count":
                    agg_df = df.groupby(group_col)[value_col].count().reset_index()
                
                agg_df.columns = [group_col, f"{agg_method} of {value_col}"]
                
                # Sort and limit to top 20 for better visualization
                agg_df = agg_df.sort_values(f"{agg_method} of {value_col}", ascending=False).head(20)
                
                # Create interactive bar chart with Plotly
                fig = px.bar(
                    agg_df, 
                    x=group_col, 
                    y=f"{agg_method} of {value_col}",
                    title=f"{agg_method} of {value_col} by {group_col}",
                    color=f"{agg_method} of {value_col}",
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text_auto=True
                )
                
                fig.update_layout(
                    xaxis_title=group_col,
                    yaxis_title=f"{agg_method} of {value_col}",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Could not create visualization: {str(e)}")
        
        # Extreme events analysis section
        st.subheader("Extreme Events Analysis")
        
        # Automatically detect extreme events columns
        extreme_cols = [col for col in df.columns if 'extreme' in col.lower() or 'event' in col.lower()]
        
        if not extreme_cols:
            st.info("No columns related to extreme events found in the dataset.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                extreme_col = st.selectbox("Select Extreme Event Type", extreme_cols)
            
            with col2:
                country_col = st.selectbox("Group by Country", 
                                          df.select_dtypes(include=['object', 'category']).columns.tolist())
            
            # Aggregate extreme events
            try:
                extreme_df = df.groupby(country_col)[extreme_col].sum().reset_index()
                extreme_df.columns = [country_col, f"Total {extreme_col}"]
                
                # Sort and limit to top 20
                extreme_df = extreme_df.sort_values(f"Total {extreme_col}", ascending=False).head(20)
                
                # Create map visualization if country data is available
                if country_col == "Country" or country_col == "country":
                    st.info("Country-level visualization")
                    
                    # Create choropleth map
                    fig_map = px.choropleth(
                        extreme_df,
                        locations=country_col,
                        locationmode='country names',
                        color=f"Total {extreme_col}",
                        hover_name=country_col,
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title=f"Total {extreme_col} by Country"
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                
                # Create bar chart
                fig_bar = px.bar(
                    extreme_df,
                    x=country_col,
                    y=f"Total {extreme_col}",
                    title=f"Total {extreme_col} by {country_col}",
                    color=f"Total {extreme_col}",
                    color_continuous_scale=px.colors.sequential.Inferno,
                    text_auto=True
                )
                fig_bar.update_layout(height=500)
                st.plotly_chart(fig_bar, use_container_width=True)
                
            except Exception as e:
                st.error(f"Could not analyze extreme events: {str(e)}")

if __name__ == "__main__":
    main()