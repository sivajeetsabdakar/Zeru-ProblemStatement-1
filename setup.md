# Setup & Usage Guide: Zeru DeFi Wallet Credit Scoring

This guide will help you install dependencies and run the wallet credit scoring pipeline with all available options.

---

## 1. Install Python (if not already installed)
- Recommended: Python 3.8+
- Download: https://www.python.org/downloads/

## 2. Install Required Packages

Open a terminal in your project directory and run:

```bash
pip install -r requirements.txt
```

This will install:
- pandas
- numpy
- scikit-learn
- torch (for autoencoder method)

## 3. Prepare Your Data
- Place your transaction data in a JSON file (e.g., `user-wallet-transactions.json`).

## 4. Run the Scoring Script

Basic usage:
```bash
python wallet_credit_score.py --input user-wallet-transactions.json --output wallet_scores.csv
```

### Method Options
Choose the clustering/outlier detection method with `--method`:
- `kmeans` (default): KMeans clustering
- `iforest`: IsolationForest anomaly detection
- `dbscan`: DBSCAN clustering
- `zscore`: Z-score outlier detection
- `mahalanobis`: Mahalanobis distance outlier detection
- `autoencoder`: Autoencoder-based anomaly detection (requires torch)

Example:
```bash
python wallet_credit_score.py --input user-wallet-transactions.json --output wallet_scores.csv --method dbscan
```

### Time-Decayed Weights (Optional)
Give more weight to recent transactions:
```bash
python wallet_credit_score.py --input user-wallet-transactions.json --output wallet_scores.csv --time_decay
```
- Adjust decay rate (default: 0.01):
```bash
python wallet_credit_score.py --input user-wallet-transactions.json --output wallet_scores.csv --time_decay --decay_lambda 0.02
```

### Full Example (All Options)
```bash
python wallet_credit_score.py --input user-wallet-transactions.json --output wallet_scores.csv --method autoencoder --time_decay --decay_lambda 0.02
```

## 5. Output
- The script will generate a CSV (default: `wallet_scores.csv`) with columns:
  - `wallet_address`, `score`, `cluster_label`, `anomaly_score`, and all engineered features.

## 6. Troubleshooting
- If you see an error about `torch` not being installed, run:
  ```bash
  pip install torch
  ```
- For large files, ensure you have enough memory and disk space.

---

For more details on the scoring logic and features, see `README.md`. 