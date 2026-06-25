import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.xgboost
import warnings

# Suppress minor warnings for clean console output
warnings.filterwarnings("ignore")

def load_and_merge_data(data_dir="data"):
    """Loads CSV files and merges them into a single DataFrame."""
    print("Loading data...")
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    features_df = pd.read_csv(f'{data_dir}/features.csv')
    stores_df = pd.read_csv(f'{data_dir}/stores.csv')

    merged_df = pd.merge(train_df, stores_df, on='Store', how='left')
    df = pd.merge(merged_df, features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
    
    return df

def engineer_features(df):
    """Cleans dates, handles missing prices, and extracts features."""
    print("Engineering features...")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pricing signals (fill missing promos with 0)
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    df[markdown_cols] = df[markdown_cols].fillna(0)

    # Time features
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    features = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'CPI', 'Unemployment', 'Week', 'Month', 'Year'] + markdown_cols
    
    # Drop rows with missing target or core features
    df = df.dropna(subset=features + ['Weekly_Sales'])
    
    return df[features], df['Weekly_Sales']

def train_and_log_model(X, y):
    """Trains the XGBoost model and logs everything to MLflow."""
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Setup MLflow Experiment
    # We use a different name so it doesn't mix with your notebook tests
    mlflow.set_experiment("DynaPrice_Production_Pipeline")

    with mlflow.start_run(run_name="Automated_Training_Run"):
        
        # Model Parameters
        params = {
            "n_estimators": 150,  # Increased from 100 in the notebook
            "learning_rate": 0.1,
            "random_state": 42
        }
        
        print("Training XGBoost model...")
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        print("Evaluating model...")
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # 2. Log everything to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.xgboost.log_model(model, "production_model")

        print("\n=== Model Training Complete ===")
        print(f"Logged to MLflow -> MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f}")

if __name__ == "__main__":
    # This is the entry point of the script.
    # It tells Python exactly what order to run the functions in.
    print("Starting DynaPrice Automated Training Pipeline...")
    raw_data = load_and_merge_data()
    X, y = engineer_features(raw_data)
    train_and_log_model(X, y)
    print("Pipeline executed successfully!")