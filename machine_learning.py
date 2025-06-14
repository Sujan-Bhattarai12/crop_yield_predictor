# import libraries
import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Updated training script
def train_and_save_lgbm(filepath, target_col, model_filename):
    # Load data
    df = pd.read_csv(filepath)
    
    # Clean data
    df.rename(columns={'Irrigation_Access_%': 'Irrigation_Access_percent'}, inplace=True)
    
    # Drop unnecessary columns
    columns_to_drop = ['CO2_Emissions_MT','Economic_Impact_Million_USD', 'Region', 'Adaptation_Strategies']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # One-hot encode categorical variables
    categorical_cols = X.select_dtypes(include='object').columns
    if not categorical_cols.empty:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000
    )

    # Save feature names
    with open('model_features.txt', 'w') as f:
        for feature in X.columns:
            f.write(f"{feature}\n")
    
    # Save model
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_filename}")

train_and_save_lgbm(
    filepath='climate_change_impact_on_agriculture_2024.csv',
    target_col='Crop_Yield_MT_per_HA',
    model_filename='lgbm_model.pkl'
)
