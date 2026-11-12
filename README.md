# On-Premise Cloud-Native AIOps Workload Forecaster

This project implements an on-premise AIOps system with workload forecasting in Kubernetes, based on the architecture proposed by Suqiong Zhang et al. (SoftwareX, 2026). It utilizes a lightweight causal transformer model (AIOpsTrs) with feature-wise normalization and a trend-aware combined loss function to predict cluster workload proactively.

## Architecture

1. **Metric Collection**: Prometheus scrapes node and pod metrics every 30 seconds.
2. **Offline Training (`model-server`)**: A PyTorch transformer model pulls historical data, normalizes heterogeneous features (e.g., memory vs. context switches), and trains using a custom MSE + difference loss function.
3. **Online Inference (`load-estimator`)**: A FastAPI service continuously polls the last 12 minutes of metrics, executes a forward pass through the trained model, and exposes the predicted load for the next step.

## Prerequisites

* Docker Desktop (with Kubernetes enabled) or Minikube.
* `kubectl` and `helm` installed and configured.
* Python 3.10+ (for local testing).

## Deployment Guide

### 1. Establish Monitoring Baseline
Create namespaces and deploy the `kube-prometheus-stack` to gather hardware telemetry.
```bash
kubectl apply -f k8s/namespace.yaml
helm repo add prometheus-community [https://prometheus-community.github.io/helm-charts](https://prometheus-community.github.io/helm-charts)
helm repo update
helm install aiops-monitor prometheus-community/kube-prometheus-stack --namespace database