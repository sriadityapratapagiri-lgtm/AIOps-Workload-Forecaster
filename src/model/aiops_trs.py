import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureWiseNormalization(nn.Module):
    """
    Implements the feature-wise normalization from the paper to handle 
    heterogeneous metric scales (e.g., memory vs context switches)[cite: 4].
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # 1. Log transformation to compress large magnitude features: X_log = log(1 + X)[cite: 4]
        x_log = torch.log1p(x)
        
        # 2. Min-Max Scaling per feature to [0, 1][cite: 4]
        x_min = x_log.min(dim=1, keepdim=True)[0]
        x_max = x_log.max(dim=1, keepdim=True)[0]
        
        x_norm = (x_log - x_min) / (x_max - x_min + self.eps)
        return x_norm

class CombinedTrendLoss(nn.Module):
    """
    Trend-aware combined loss: balances absolute prediction accuracy (MSE) 
    with temporal consistency (difference loss)[cite: 4].
    """
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha # Paper found 0.8 to be optimal[cite: 9]
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # L_MSE: Standard point-wise accuracy[cite: 4]
        loss_mse = self.mse(pred, target)
        
        # L_diff: First-order difference loss to capture temporal variations[cite: 4]
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        loss_diff = self.mse(pred_diff, target_diff)
        
        # L_total = alpha * L_MSE + (1 - alpha) * L_diff[cite: 4]
        return (self.alpha * loss_mse) + ((1 - self.alpha) * loss_diff)

class AIOpsTrs(nn.Module):
    """
    Lightweight Causal Transformer for workload forecasting[cite: 2, 4].
    """
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=4):
        super().__init__()
        # The paper suggests 4 layers and 4-8 heads for optimal performance[cite: 9]
        self.norm = FeatureWiseNormalization()
        self.embedding = nn.Linear(num_features, d_model)
        
        # Transformer Encoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, 1) # Predicting the single "load" target

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        x = self.norm(x)
        x = self.embedding(x)
        
        # Causal mask to prevent attending to future time steps[cite: 4]
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        out = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.fc_out(out)