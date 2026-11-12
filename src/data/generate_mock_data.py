import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_telemetry(hours=18, interval_seconds=30):
    """Generates synthetic CI/CD workload data matching the paper's specifications[cite: 7, 8]."""
    num_rows = int((hours * 3600) / interval_seconds)
    timestamps = [datetime.now() - timedelta(hours=hours) + timedelta(seconds=i*interval_seconds) for i in range(num_rows)]
    
    # Simulate a baseline sine wave for cyclical daily load, plus random noise
    time_array = np.linspace(0, 10 * np.pi, num_rows)
    base_load = np.sin(time_array) * 0.3 + 0.5 
    noise = np.random.normal(0, 0.05, num_rows)
    
    # Inject synthetic CI/CD pipeline spikes (e.g., compilation tasks)
    spikes = np.zeros(num_rows)
    spike_indices = np.random.choice(num_rows, size=20, replace=False)
    for idx in spike_indices:
        spikes[idx:min(idx+10, num_rows)] = np.random.uniform(0.8, 1.5)

    load1m = np.clip(base_load + noise + spikes, 0, None)
    
    # Generate correlated hardware metrics
    data = {
        'timestamp': timestamps,
        'sys-mem-total': np.full(num_rows, 16 * 1024**3), # 16 GB constant
        'sys-mem-free': (16 * 1024**3) - (load1m * 8 * 1024**3), # Inverse to load
        'sys-interrupt-rate': load1m * 5000 + np.random.normal(0, 200, num_rows),
        'sys-context-switch-rate': load1m * 8000 + np.random.normal(0, 300, num_rows),
        'disk-io-time': load1m * 0.4,
        'cpu-user': load1m * 0.7,
        'cpu-system': load1m * 0.2,
        'load1m': load1m,
        'load5m': pd.Series(load1m).rolling(window=10, min_periods=1).mean(),
        'load15m': pd.Series(load1m).rolling(window=30, min_periods=1).mean(),
    }
    
    df = pd.DataFrame(data).set_index('timestamp')
    return df.astype('float32')

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_telemetry(hours=18)
    output_path = "data/raw/cluster_metrics_18h.csv"
    df.to_csv(output_path)
    print(f"Generated {len(df)} rows of synthetic telemetry at {output_path}")