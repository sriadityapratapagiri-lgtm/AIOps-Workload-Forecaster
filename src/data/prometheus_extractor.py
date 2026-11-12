import os
import pandas as pd
from datetime import datetime, timedelta
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
from prometheus_api_client.utils import parse_datetime

class PrometheusExtractor:
    def __init__(self, url="http://localhost:9090", disable_ssl=True):
        """Initializes the connection to the Prometheus server deployed in the cluster."""
        self.prom = PrometheusConnect(url=url, disable_ssl=disable_ssl)
        
        # Mapping paper metrics to standard Node Exporter PromQL queries
        self.queries = {
            "sys-mem-total": 'node_memory_MemTotal_bytes',
            "sys-mem-free": 'node_memory_MemFree_bytes',
            "sys-interrupt-rate": 'rate(node_intr_total[1m])',
            "sys-context-switch-rate": 'rate(node_context_switches_total[1m])',
            "disk-io-time": 'rate(node_disk_io_time_seconds_total[1m])',
            "cpu-user": 'sum(rate(node_cpu_seconds_total{mode="user"}[1m])) by (instance)',
            "cpu-system": 'sum(rate(node_cpu_seconds_total{mode="system"}[1m])) by (instance)',
            "load1m": 'node_load1',
            "load5m": 'node_load5',
            "load15m": 'node_load15'
        }

    def fetch_historical_data(self, start_time, end_time, step="30s"):
        """
        Extracts metrics at 30-second intervals for offline feature extraction[cite: 8].
        """
        start = parse_datetime(start_time)
        end = parse_datetime(end_time)
        
        extracted_data = []

        for metric_name, promql in self.queries.items():
            print(f"Fetching {metric_name}...")
            metric_data = self.prom.custom_query_range(
                query=promql,
                start_time=start,
                end_time=end,
                step=step
            )
            
            if metric_data:
                # Convert to DataFrame and standardize
                df = MetricRangeDataFrame(metric_data)
                df = df[['value']].rename(columns={'value': metric_name})
                extracted_data.append(df)
            else:
                print(f"Warning: No data found for {metric_name}")

        if not extracted_data:
            raise ValueError("No metrics could be extracted. Check Prometheus connection and data availability.")

        # Merge all metric DataFrames on their datetime index
        final_df = pd.concat(extracted_data, axis=1)
        
        # Forward fill missing values caused by slight scrape misalignments, then drop remaining NaNs
        final_df = final_df.ffill().dropna()
        
        # Cast all metric columns to float32 for PyTorch compatibility
        return final_df.astype('float32')

if __name__ == "__main__":
    # Example execution for offline extraction
    # The paper utilized an 18-hour data window for training[cite: 7]
    extractor = PrometheusExtractor(url="http://localhost:9090") # Port-forwarded Prometheus URL
    
    end = datetime.now()
    start = end - timedelta(hours=18) 
    
    print(f"Extracting cluster telemetry from {start} to {end} at 30s intervals...")
    
    try:
        dataset = extractor.fetch_historical_data(
            start_time=str(start),
            end_time=str(end),
            step="30s"
        )
        
        # Save to CSV for the offline training pipeline
        os.makedirs("data/raw", exist_ok=True)
        output_path = "data/raw/cluster_metrics_18h.csv"
        dataset.to_csv(output_path)
        print(f"Extraction complete. Target workload and features saved to {output_path}")
        print(dataset.head())
        
    except Exception as e:
        print(f"Extraction failed: {e}")