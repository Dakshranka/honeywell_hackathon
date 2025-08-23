"""
Model training and anomaly scoring for Hackathon
Trains Isolation Forest, PCA, LSTM Autoencoder on normal period, scores analysis period, outputs abnormality scores and top_feature_1-7.

Args:
    input_train_path (str): Path to normal period CSV.
    input_analysis_path (str): Path to analysis period CSV.
    output_path (str): Path to save output CSV.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from typing import List, Tuple
import os

def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to 0-100 scale."""
    min_score = np.min(scores)
    max_score = np.max(scores)
    norm = (scores - min_score) / (max_score - min_score + 1e-8) * 100
    return norm

def get_top_features(score_row: np.ndarray, train_mean: np.ndarray, feature_names: List[str], top_k: int = 7) -> List[str]:
    """Rank features by absolute contribution (>1%), break ties alphabetically, fill up to top_k."""
    abs_contrib = np.abs(score_row - train_mean)
    total = abs_contrib.sum()
    contrib_pct = abs_contrib / (total + 1e-8)
    idx = np.where(contrib_pct > 0.01)[0]
    sorted_idx = sorted(idx, key=lambda i: (-abs_contrib[i], feature_names[i]))
    top_features = [feature_names[i] for i in sorted_idx[:top_k]]
    while len(top_features) < top_k:
        top_features.append("")
    return top_features

# LSTM Autoencoder for temporal anomaly detection
class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for temporal anomaly detection."""
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        latent = self.latent(enc_out)
        dec_out, _ = self.decoder(latent)
        out = self.output(dec_out)
        return out

def lstm_ae_score(train_X: np.ndarray, test_X: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Train LSTM Autoencoder and return reconstruction error for test set."""
    model = LSTMAutoencoder(train_X.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_tensor = torch.tensor(train_X, dtype=torch.float32).to(device)
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, train_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    test_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(test_tensor)
        mse = ((output - test_tensor) ** 2).mean(dim=-1).squeeze().cpu().numpy()
    return mse

def run_anomaly_detection(
    input_train_path: str,
    input_analysis_path: str,
    output_path: str
) -> None:
    """
    Run anomaly detection pipeline and save output CSV.
    Args:
        input_train_path: Path to normal period CSV.
        input_analysis_path: Path to analysis period CSV.
        output_path: Path to save output CSV.
    """
    train_df = pd.read_csv(input_train_path, index_col=0, parse_dates=True)
    analysis_df = pd.read_csv(input_analysis_path, index_col=0, parse_dates=True)
    feature_cols = [col for col in analysis_df.columns]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_df[feature_cols].values)
    test_X = scaler.transform(analysis_df[feature_cols].values)
    train_mean = np.mean(train_X, axis=0)

    # Isolation Forest with lower contamination
    if_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    if_model.fit(train_X)
    if_scores = -if_model.score_samples(test_X)
    abnormality_if = normalize_scores(if_scores)

    # PCA reconstruction error
    pca = PCA(n_components=min(10, len(feature_cols)))
    pca.fit(train_X)
    test_proj = pca.transform(test_X)
    test_recon = pca.inverse_transform(test_proj)
    pca_scores = np.mean((test_X - test_recon) ** 2, axis=1)
    abnormality_pca = normalize_scores(pca_scores)

    # LSTM Autoencoder (windowed)
    window = 10
    train_seq = np.expand_dims(train_X[:-(window-1)], axis=0)
    test_seq = np.expand_dims(test_X[:-(window-1)], axis=0)
    lstm_scores = lstm_ae_score(train_seq, test_seq)
    abnormality_lstm = normalize_scores(lstm_scores)

    # Truncate all arrays to the minimum length
    min_len = min(len(abnormality_if), len(abnormality_pca), len(abnormality_lstm))
    abnormality_if = abnormality_if[:min_len]
    abnormality_pca = abnormality_pca[:min_len]
    abnormality_lstm = abnormality_lstm[:min_len]
    abnormality_ensemble = normalize_scores((abnormality_if + abnormality_pca + abnormality_lstm) / 3)

    # Feature attribution for each row (using IF logic)
    top_features_list = []
    for i in range(min_len):
        top_feats = get_top_features(test_X[i], train_mean, feature_cols, top_k=7)
        top_features_list.append(top_feats)

    # Add columns to output
    output_df = analysis_df.iloc[:min_len].copy()
    output_df['abnormality_score_iforest'] = abnormality_if
    output_df['abnormality_score_pca'] = abnormality_pca
    output_df['abnormality_score_lstm'] = abnormality_lstm
    output_df['abnormality_score_ensemble'] = abnormality_ensemble
    for k in range(7):
        output_df[f'top_feature_{k+1}'] = [feats[k] for feats in top_features_list]
    # Reset index to ensure timestamp is a column and format as string
    output_df = output_df.reset_index()
    output_df['timestamp'] = output_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    output_df.to_csv(output_path, index=False)
    print(f"Anomaly scores (IF, PCA, LSTM, Ensemble) and top features saved to {output_path}")

    # Training period anomaly score check
    train_scores = -if_model.score_samples(train_X)
    train_abnormality = normalize_scores(train_scores)
    print(f"Training period abnormality score: mean={train_abnormality.mean():.2f}, max={train_abnormality.max():.2f}")
    if train_abnormality.mean() >= 10 or train_abnormality.max() >= 25:
        print("WARNING: Training period anomaly scores exceed recommended thresholds (mean < 10, max < 25)")

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    input_train_path = os.path.join(output_dir, "normal_period.csv")
    input_analysis_path = os.path.join(output_dir, "analysis_period.csv")
    output_path = os.path.join(output_dir, "anomalies_output.csv")
    run_anomaly_detection(input_train_path, input_analysis_path, output_path)

if __name__ == "__main__":
    main()
