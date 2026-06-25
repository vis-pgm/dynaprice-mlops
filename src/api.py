from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.xgboost
import mlflow

app = FastAPI(title="DynaPrice API", description="Dynamic Pricing & Demand Forecasting Engine")

# 1. Automatically find and load the latest trained model from MLflow
print("Locating latest model in MLflow...")
experiment = mlflow.get_experiment_by_name("DynaPrice_Production_Pipeline")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)

latest_run_id = runs.iloc[0]["run_id"]
model_uri = f"runs:/{latest_run_id}/production_model"

print(f"Loading Model from Run ID: {latest_run_id}")
model = mlflow.xgboost.load_model(model_uri)

# 2. Define the exact JSON structure the API expects to receive
class PricingRequest(BaseModel):
    Store: int
    Dept: int
    IsHoliday: int
    Temperature: float
    CPI: float
    Unemployment: float
    Week: int
    Month: int
    Year: int
    MarkDown1: float = 0.0
    MarkDown2: float = 0.0
    MarkDown3: float = 0.0
    MarkDown4: float = 0.0
    MarkDown5: float = 0.0

# 3. Create the endpoint that frontend applications will call
@app.post("/predict")
def predict_demand(request: PricingRequest):
    # Convert the incoming JSON request into a Pandas DataFrame
    input_data = pd.DataFrame([request.model_dump()])
    
    # Ask the XGBoost model for a prediction
    prediction = model.predict(input_data)
    
    # Return the result as JSON
    return {
        "predicted_weekly_sales": float(prediction[0]),
        "currency": "USD",
        "model_version": latest_run_id
    }