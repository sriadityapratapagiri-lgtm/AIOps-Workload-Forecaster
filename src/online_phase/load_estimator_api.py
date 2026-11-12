import torch
import pandas as pd
from fastapi import FastAPI
from prometheus_client import Gauge, generate_latest
from fastapi.responses import Response
from src.model.aiops_trs import AIOpsTrs
from src.data.feature_engineering import FeatureProcessor
from data.prometheous_extractor import PrometheusExtractor
import asyncio
from datetime import datetime, timedelta

app = FastAPI(title="AIOps Load Estimator")

# Prometheus Gauge to expose the forecasted workload[cite: 5]
FORECASTED_LOAD = Gauge('aiops_predicted_load_1m', 'Predicted 1-minute system load for the next step')

class OnlineEstimator:
    def __init__(self):
        self.extractor = PrometheusExtractor(url="http://prometheus-k8s.database.svc.cluster.local:9090")
        self.processor = FeatureProcessor.load_scaler("models/feature_processor.pkl")
        
        # Determine num_features from the saved scaler
        num_features = len(self.processor.scaler.data_max_)
        self.model = AIOpsTrs(num_features=num_features, d_model=64, nhead=4, num_layers=4)
        self.model.load_state_dict(torch.load("models/aiops_trs_weights.pt"))
        self.model.eval()

    def predict_next_step(self):
        # Fetch the last 24 timestamps (12 minutes at 30s intervals)[cite: 7, 8]
        end = datetime.now()
        start = end - timedelta(minutes=12)
        
        try:
            df = self.extractor.fetch_historical_data(start_time=str(start), end_time=str(end), step="30s")
            
            # Pad or truncate to exactly 24 steps
            if len(df) > 24:
                df = df.tail(24)
            elif len(df) < 24:
                return None
                
            processed_df = self.processor.transform(df)
            input_tensor = torch.tensor(processed_df.values, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                prediction = self.model(input_tensor).squeeze(-1)
                
            # Extract the prediction for the final time step
            return prediction[0, -1].item()
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None

estimator = None

@app.on_event("startup")
async def startup_event():
    global estimator
    estimator = OnlineEstimator()
    asyncio.create_task(inference_loop())

async def inference_loop():
    """Background loop to continuously predict and update the gauge every 30 seconds[cite: 8]."""
    while True:
        pred = estimator.predict_next_step()
        if pred is not None:
            FORECASTED_LOAD.set(pred)
            print(f"Updated forecast metric: {pred:.4f}")
        await asyncio.sleep(30)

@app.get("/metrics")
def metrics():
    """Exposes the forecasted load metric to Prometheus/KEDA."""
    return Response(generate_latest(), media_type="text/plain")