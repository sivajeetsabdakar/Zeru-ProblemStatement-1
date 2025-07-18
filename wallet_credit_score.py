import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance

# ----------------------
# Feature Engineering
# ----------------------
def parse_transactions(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    # Flatten amount and price fields
    df['amount'] = df['actionData.amount'].astype(float)
    df['assetPriceUSD'] = df['actionData.assetPriceUSD'].astype(float)
    df['usd_value'] = df['amount'] * df['assetPriceUSD']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def engineer_features(df, time_decay=False, decay_lambda=0.01):
    features = []
    now = df['timestamp'].max()
    for wallet, group in df.groupby('userWallet'):
        f = {'wallet_address': wallet}
        # Time decay weights
        if time_decay:
            age_days = (now - group['timestamp']).dt.total_seconds() / 86400
            weights = np.exp(-decay_lambda * age_days)
        else:
            weights = np.ones(len(group))
        # Transaction counts
        f['num_deposits'] = (group['action'] == 'deposit').sum()
        f['num_borrows'] = (group['action'] == 'borrow').sum()
        f['num_redeems'] = (group['action'] == 'redeemunderlying').sum()
        f['num_repays'] = (group['action'] == 'repay').sum()
        f['num_liquidations'] = (group['action'] == 'liquidationcall').sum()
        f['total_transactions'] = len(group)
        # Value-based (time-decayed if enabled)
        f['total_deposit_usd'] = (group.loc[group['action'] == 'deposit', 'usd_value'] * weights[group['action'] == 'deposit']).sum()
        f['total_borrow_usd'] = (group.loc[group['action'] == 'borrow', 'usd_value'] * weights[group['action'] == 'borrow']).sum()
        f['total_redeem_usd'] = (group.loc[group['action'] == 'redeemunderlying', 'usd_value'] * weights[group['action'] == 'redeemunderlying']).sum()
        f['redeem_to_deposit_ratio'] = (
            f['total_redeem_usd'] / f['total_deposit_usd'] if f['total_deposit_usd'] > 0 else 0
        )
        f['avg_transaction_usd'] = (group['usd_value'] * weights).sum() / weights.sum() if weights.sum() > 0 else 0
        f['max_transaction_usd'] = (group['usd_value'] * weights).max() if len(group) > 0 else 0
        # Diversity
        f['unique_assets_interacted'] = group['actionData.assetSymbol'].nunique()
        f['num_tx_types'] = group['action'].nunique()
        # Temporal
        f['first_tx_time'] = group['timestamp'].min()
        f['last_tx_time'] = group['timestamp'].max()
        f['wallet_lifetime_days'] = max((f['last_tx_time'] - f['first_tx_time']).days, 1)
        f['txs_per_day'] = f['total_transactions'] / f['wallet_lifetime_days']
        f['num_days_active'] = group['timestamp'].dt.date.nunique()
        # Volatility: std of daily tx counts
        daily_counts = group.groupby(group['timestamp'].dt.date).size()
        f['tx_volatility'] = daily_counts.std() if len(daily_counts) > 1 else 0
        # Bot/risk patterns
        borrow_days = set(group.loc[group['action'] == 'borrow', 'timestamp'].dt.date)
        redeem_days = set(group.loc[group['action'] == 'redeemunderlying', 'timestamp'].dt.date)
        f['borrow_then_redeem_same_day'] = int(len(borrow_days & redeem_days) > 0)
        liquid_days = set(group.loc[group['action'] == 'liquidationcall', 'timestamp'].dt.date)
        f['redeem_after_liquidation'] = int(len(redeem_days & liquid_days) > 0)
        features.append(f)
    features_df = pd.DataFrame(features)
    # Fill missing
    features_df = features_df.fillna(0)
    return features_df

# ----------------------
# Preprocessing
# ----------------------
def preprocess_features(features_df):
    # Drop non-numeric columns for modeling
    non_numeric = ['wallet_address', 'first_tx_time', 'last_tx_time']
    X = features_df.drop(columns=non_numeric, errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, X.columns.tolist()

# ----------------------
# Clustering & Scoring
# ----------------------
def cluster_and_score(features_df, X_scaled, method='kmeans', n_clusters=4):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        centroids = kmeans.cluster_centers_
        best_cluster = np.argmax(centroids[:, features_df.columns.get_loc('total_deposit_usd')])
        dists = np.linalg.norm(X_scaled - centroids[best_cluster], axis=1)
        anomaly_score = (dists - dists.min()) / (dists.max() - dists.min() + 1e-9)
    elif method == 'iforest':
        iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        iforest.fit(X_scaled)
        anomaly_score = -iforest.score_samples(X_scaled)
        anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-9)
        cluster_labels = (anomaly_score > 0.7).astype(int)  # 1=outlier, 0=inlier
    elif method == 'dbscan':
        dbscan = DBSCAN(eps=1.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(X_scaled)
        anomaly_score = np.where(cluster_labels == -1, 1.0, 0.0)
    elif method == 'zscore':
        mean = X_scaled.mean(axis=0)
        std = X_scaled.std(axis=0) + 1e-9
        zscores = np.abs((X_scaled - mean) / std)
        # Use max z-score across features as anomaly score
        anomaly_score = zscores.max(axis=1)
        anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min() + 1e-9)
        cluster_labels = np.zeros(len(anomaly_score), dtype=int)
    elif method == 'mahalanobis':
        mean = X_scaled.mean(axis=0)
        cov = np.cov(X_scaled, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        mahal = np.array([distance.mahalanobis(x, mean, cov_inv) for x in X_scaled])
        anomaly_score = (mahal - mahal.min()) / (mahal.max() - mahal.min() + 1e-9)
        cluster_labels = np.zeros(len(anomaly_score), dtype=int)
    elif method == 'autoencoder':
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError('PyTorch is required for autoencoder method. Please install torch.')
        # Define a simple autoencoder
        class Autoencoder(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, max(2, input_dim // 2)),
                    nn.ReLU(),
                    nn.Linear(max(2, input_dim // 2), max(1, input_dim // 4)),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(max(1, input_dim // 4), max(2, input_dim // 2)),
                    nn.ReLU(),
                    nn.Linear(max(2, input_dim // 2), input_dim)
                )
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        model = Autoencoder(X_scaled.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        # Train autoencoder
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = loss_fn(output, X_tensor)
            loss.backward()
            optimizer.step()
        # Compute reconstruction error
        model.eval()
        with torch.no_grad():
            recon = model(X_tensor)
            errors = ((recon - X_tensor) ** 2).mean(dim=1).numpy()
        anomaly_score = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)
        cluster_labels = np.zeros(len(anomaly_score), dtype=int)
    else:
        raise ValueError('Unknown clustering method')
    return cluster_labels, anomaly_score

def compute_cluster_bonus(features_df, cluster_labels, method):
    # Compute cluster reputation as mean of total_deposit_usd + wallet_lifetime_days
    rep_feature = features_df[['total_deposit_usd', 'wallet_lifetime_days']].sum(axis=1)
    cluster_df = pd.DataFrame({'cluster': cluster_labels, 'rep': rep_feature})
    # For DBSCAN, treat noise (-1) as lowest reputation
    if method == 'dbscan':
        valid_clusters = cluster_df[cluster_df['cluster'] != -1]['cluster'].unique()
        cluster_reps = cluster_df[cluster_df['cluster'] != -1].groupby('cluster')['rep'].mean()
        min_rep = cluster_reps.min() if len(cluster_reps) > 0 else 0
        cluster_reps = pd.concat([cluster_reps, pd.Series({-1: min_rep - 1})])
    else:
        cluster_reps = cluster_df.groupby('cluster')['rep'].mean()
    # Normalize cluster reputation to [0, 1]
    rep_min, rep_max = cluster_reps.min(), cluster_reps.max()
    cluster_reps_norm = (cluster_reps - rep_min) / (rep_max - rep_min + 1e-9)
    # Map each wallet's cluster to its normalized reputation
    bonus = cluster_df['cluster'].map(cluster_reps_norm).fillna(0).values
    # Scale bonus to [-0.1, 0.1] (10% penalty/bonus)
    bonus = (bonus - 0.5) * 0.2
    return bonus

def compute_final_score(features_df, anomaly_score, cluster_labels=None, method=None):
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(features_df[['total_deposit_usd', 'wallet_lifetime_days', 'redeem_to_deposit_ratio', 'txs_per_day', 'tx_volatility']])
    total_deposit, lifetime, redeem_ratio, txs_per_day, tx_volatility = normed.T
    score = (
        0.3 * total_deposit +
        0.2 * lifetime +
        0.2 * (1 - redeem_ratio) +
        0.2 * txs_per_day -
        0.1 * anomaly_score -
        0.1 * tx_volatility
    ) * 1000
    # Add cluster bonus/penalty if available
    if cluster_labels is not None and method is not None:
        bonus = compute_cluster_bonus(features_df, cluster_labels, method)
        score = score + bonus * 1000
    # Final normalization: spread scores to [0, 1000]
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    score = score * 1000
    score = np.clip(score, 0, 1000)
    return score

# ----------------------
# Main Pipeline
# ----------------------
def main():
    parser = argparse.ArgumentParser(description='Wallet Credit Scoring Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Path to user-wallet-transactions.json')
    parser.add_argument('--output', type=str, default='wallet_scores.csv', help='Output CSV file')
    parser.add_argument('--method', type=str, choices=['kmeans', 'iforest', 'dbscan', 'zscore', 'mahalanobis', 'autoencoder'], default='kmeans', help='Clustering/outlier method')
    parser.add_argument('--time_decay', action='store_true', help='Enable time-decayed weights for recent transactions')
    parser.add_argument('--decay_lambda', type=float, default=0.01, help='Decay rate for time-decayed weights (default: 0.01)')
    args = parser.parse_args()

    print('Loading and parsing transactions...')
    df = parse_transactions(args.input)
    print('Engineering features...')
    features_df = engineer_features(df, time_decay=args.time_decay, decay_lambda=args.decay_lambda)
    print('Preprocessing features...')
    X_scaled, scaler, feature_cols = preprocess_features(features_df)
    print(f'Clustering and scoring using {args.method}...')
    cluster_labels, anomaly_score = cluster_and_score(features_df, X_scaled, method=args.method)
    print('Computing final scores...')
    final_score = compute_final_score(features_df, anomaly_score, cluster_labels, args.method)
    # Output
    features_df['score'] = final_score.astype(int)
    features_df['cluster_label'] = cluster_labels
    features_df['anomaly_score'] = anomaly_score
    out_cols = ['wallet_address', 'score', 'cluster_label', 'anomaly_score'] + [c for c in feature_cols if c != 'wallet_address']
    features_df[out_cols].to_csv(args.output, index=False)
    print(f'Saved scores to {args.output}')

    # --- Automated Score Distribution Plot ---
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        # Reload scores to ensure correct output
        scores = pd.read_csv(args.output)
        bins = list(range(0, 1100, 100))
        labels = [f'{b}-{b+100}' for b in bins[:-1]]
        scores['score_bin'] = pd.cut(scores['score'], bins=bins, labels=labels, right=False)
        sns.set(style='whitegrid')
        plt.figure(figsize=(10,6))
        scores['score_bin'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title('Wallet Credit Score Distribution')
        plt.xlabel('Score Range')
        plt.ylabel('Number of Wallets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('score_distribution.png')
        print('Saved score distribution plot as score_distribution.png')
        # Print summary statistics for each bin
        summary = scores.groupby('score_bin').agg({
            'total_deposit_usd': 'mean',
            'wallet_lifetime_days': 'mean',
            'redeem_to_deposit_ratio': 'mean',
            'tx_volatility': 'mean',
            'total_transactions': 'mean',
            'score': 'count'
        }).rename(columns={'score': 'num_wallets'})
        print('\nScore Bin Summary:')
        print(summary)
    except Exception as e:
        print(f'Could not generate score distribution plot or summary: {e}')

if __name__ == '__main__':
    main() 