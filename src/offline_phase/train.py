import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from src.model.aiops_trs import AIOpsTrs, CombinedTrendLoss
from src.data.feature_engineering import FeatureProcessor

def create_sequences(data, target_idx, window_size=24):
    # The paper compares window sizes 24, 36, and 48[cite: 7]
    xs, ys = [], []
    for i in range(len(data) - window_size - 1):
        x = data[i:(i + window_size)]
        y = data[i + 1:(i + window_size + 1), target_idx] # One-step forward shift[cite: 4]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

def train_model():
    print("Loading extracted Prometheus telemetry...")
    df = pd.read_csv("data/raw/cluster_metrics_18h.csv", index_col=0)
    
    # Feature processing
    processor = FeatureProcessor()
    processed_df = processor.fit_transform(df)
    os.makedirs("models", exist_ok=True)
    processor.save_scaler("models/feature_processor.pkl")
    
    # Identify target metric index (e.g., 'load1m')[cite: 8]
    target_idx = processed_df.columns.get_loc('load1m')
    
    X, y = create_sequences(processed_df.values, target_idx, window_size=24)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize AIOpsTrs (4 layers, 4 heads optimal)[cite: 9]
    model = AIOpsTrs(num_features=X.shape[2], d_model=64, nhead=4, num_layers=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Alpha = 0.8 yields optimal performance for balancing absolute error and trend alignment[cite: 9]
    criterion = CombinedTrendLoss(alpha=0.8) 
    
    model.train()
    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x).squeeze(-1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
            
    torch.save(model.state_dict(), "models/aiops_trs_weights.pt")
    print("Training complete. Model weights saved.")

if __name__ == "__main__":
    train_model()