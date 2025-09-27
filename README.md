# Crop Yield Prediction and Forecasting App

## Background  
Agricultural productivity is highly sensitive to multiple factors, including environmental conditions, farming practices, and crop-specific characteristics. Accurate prediction of crop yield and forecasting of future trends are critical for decision-making by farmers, policymakers, and researchers.  

This project provides a **Streamlit web application** that enables interactive prediction of crop yield using machine learning models (LightGBM) and time series forecasting (ARIMA). It combines data preprocessing, predictive modeling, and statistical analysis in an easy-to-use dashboard.  

---

## Problem Statement  
The app addresses the following key questions:  
1. Can I accurately predict crop yield based on features such as area, production, season, and crop type?  
2. How can I forecast future yields using time series methods?  
3. What are the underlying statistical properties of the dataset (e.g., distribution, correlation, normality)?  
4. How can machine learning and forecasting methods be combined in a single interface for better insights?  

---

## Methodology  

### 1. Data Preprocessing  
- The dataset is uploaded by the user via the app.  
- Features are standardized using **StandardScaler**.  
- Missing or inconsistent values can be inspected interactively.  

### 2. Predictive Modeling  
- **LightGBM Regressor** is trained on user-selected features.  
- Model evaluation is performed using:  
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  
  - R² Score  

### 3. Forecasting  
- Time series forecasting is implemented with **ARIMA**.  
- Users can select a state, crop, and number of periods to forecast.  
- Forecasted values are plotted against historical yields.  

### 4. Statistical Analysis  
- Correlation heatmaps to visualize feature relationships.  
- Distribution plots of numeric variables.  
- Q-Q plots for testing normality of residuals.  
---

## Results & Interpretation  
- **Model Accuracy**  
  - LightGBM achieved strong predictive power on yield data, with reasonable MAE and R² values.  
  - Performance depends on feature selection (e.g., including "Area" and "Production" improves accuracy).  

- **Forecasting**  
  - ARIMA provides insights into yield trends for specific crops and states.  
  - Useful for planning agricultural policy and resource allocation.  

- **Statistical Insights**  
  - Strong correlations were observed between yield, production, and area.  
  - Some distributions deviated from normality, highlighting the importance of robust machine learning models.  
---

## Key Takeaways  
- The app integrates **machine learning (LightGBM)** and **time series forecasting (ARIMA)** into a single tool.  
- Provides both **predictive analytics** (for current yield) and **forecasting** (for future planning).  
- Offers **statistical analysis** to better understand data patterns.  
- Helps farmers, researchers, and policymakers make data-driven agricultural decisions.  

---

## Future Work  
- Extend support for additional forecasting models (Prophet, LSTM).  
- Incorporate weather and climate variables for improved predictions.  
- Deploy a cloud-based version with persistent storage.  
- Add geospatial visualizations (maps of yield across regions).  

---
