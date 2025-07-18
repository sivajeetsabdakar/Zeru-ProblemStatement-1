import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

METHODS = ['kmeans', 'iforest', 'dbscan', 'zscore', 'mahalanobis', 'autoencoder']
INPUT_FILE = 'user-wallet-transactions.json'
ANALYSIS_MD = 'analysis.md'

# Run scoring for each method and save output
for method in METHODS:
    output_csv = f'wallet_scores_{method}.csv'
    print(f'Running {method}...')
    subprocess.run([
        'python', 'wallet_credit_score.py',
        '--input', INPUT_FILE,
        '--output', output_csv,
        '--method', method
    ], check=True)
    # Generate and save plot
    scores = pd.read_csv(output_csv)
    bins = list(range(0, 1100, 100))
    labels = [f'{b}-{b+100}' for b in bins[:-1]]
    scores['score_bin'] = pd.cut(scores['score'], bins=bins, labels=labels, right=False)
    sns.set(style='whitegrid')
    plt.figure(figsize=(8,5))
    scores['score_bin'].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title(f'Score Distribution: {method}')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Wallets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_file = f'score_distribution_{method}.png'
    plt.savefig(img_file)
    plt.close()

# Write analysis.md
with open(ANALYSIS_MD, 'w') as f:
    f.write('# Wallet Score Analysis (All Methods)\n\n')
    for method in METHODS:
        output_csv = f'wallet_scores_{method}.csv'
        img_file = f'score_distribution_{method}.png'
        scores = pd.read_csv(output_csv)
        bins = list(range(0, 1100, 100))
        labels = [f'{b}-{b+100}' for b in bins[:-1]]
        scores['score_bin'] = pd.cut(scores['score'], bins=bins, labels=labels, right=False)
        summary = scores.groupby('score_bin').agg({
            'total_deposit_usd': 'mean',
            'wallet_lifetime_days': 'mean',
            'redeem_to_deposit_ratio': 'mean',
            'tx_volatility': 'mean',
            'total_transactions': 'mean',
            'score': 'count'
        }).rename(columns={'score': 'num_wallets'})
        f.write(f'## {method.capitalize()}\n')
        f.write(f'![Score Distribution: {method}]({img_file})\n\n')
        f.write('### Score Bin Summary\n')
        f.write(summary.to_markdown() + '\n\n')
        # Behavior analysis
        low = summary.iloc[0:2].mean()  # 0-200
        high = summary.iloc[-2:].mean() # 800-1000
        f.write('**Behavior in Low Score Range (0-200):**\n')
        f.write(f'- Avg deposit: {low["total_deposit_usd"]:.2e}, Lifetime: {low["wallet_lifetime_days"]:.1f}, Redeem/Deposit: {low["redeem_to_deposit_ratio"]:.2f}, Volatility: {low["tx_volatility"]:.2f}, Tx count: {low["total_transactions"]:.1f}\n')
        f.write('  - Typically: high redeem-to-deposit, short lifetime, high volatility, low activity.\n')
        f.write('**Behavior in High Score Range (800-1000):**\n')
        f.write(f'- Avg deposit: {high["total_deposit_usd"]:.2e}, Lifetime: {high["wallet_lifetime_days"]:.1f}, Redeem/Deposit: {high["redeem_to_deposit_ratio"]:.2f}, Volatility: {high["tx_volatility"]:.2f}, Tx count: {high["total_transactions"]:.1f}\n')
        f.write('  - Typically: large/consistent deposits, long lifetime, low redeem-to-deposit, low volatility, steady activity.\n\n')
    f.write('---\n')
    f.write('This analysis was generated automatically for all methods.\n') 