# Zeru DeFi Wallet Credit Scoring

This project provides a robust, transparent pipeline to assign a credit score (0–1000) to DeFi wallets based on their historical transaction behavior on the Aave V2 protocol. The approach is unsupervised and uses clustering/anomaly detection to group and score wallets.

---

## Methodology Overview

### Chosen Methods
- **Unsupervised Learning:** No ground-truth labels are required. We use clustering and anomaly detection to group wallets by behavioral similarity and flag outliers.
- **Multiple Algorithms:** The pipeline supports KMeans, DBSCAN, IsolationForest, Z-score, Mahalanobis distance, and Autoencoder-based anomaly detection. This flexibility allows for robust, extensible scoring.
- **Feature Engineering:** We extract a rich set of features from raw transaction logs, including behavioral counts, value-based metrics, diversity, temporal activity, and volatility.
- **Scoring:** Scores are a weighted combination of normalized features, anomaly/cluster information, volatility, and cluster reputation, all mapped to a 0–1000 scale.

### Why Unsupervised?
- DeFi lacks labeled “good” or “bad” wallet data. Unsupervised methods let us group and score wallets based on real behavioral patterns, not arbitrary rules.
- Outlier detection helps flag risky, bot-like, or exploitative behavior.

---

## Architecture

```mermaid
graph TD;
    A[Input JSON: Raw Transactions] --> B[Feature Engineering]
    B --> C[Preprocessing (Scaling, Cleaning)]
    C --> D[Clustering/Outlier Detection]
    D --> E[Score Calculation]
    E --> F[Output CSV/JSON]
```

- **Input:** JSON file with transaction records (wallet, action, amount, asset, timestamp, etc.)
- **Feature Engineering:** Aggregates per-wallet features (counts, values, diversity, volatility, etc.)
- **Preprocessing:** Handles missing values, normalizes features, applies time-decay if enabled
- **Clustering/Outlier Detection:** Groups wallets or flags outliers using the chosen method
- **Score Calculation:** Combines features, anomaly/cluster info, volatility, and cluster reputation into a final score
- **Output:** CSV/JSON with wallet scores, cluster labels, anomaly scores, and all features

---

## Processing Flow

1. **Data Ingestion:**
   - Reads a JSON file of raw transaction logs (one record per transaction).
2. **Feature Engineering:**
   - For each wallet, computes:
     - Transaction counts (deposits, borrows, redeems, etc.)
     - Value-based features (total deposit/borrow/redeem in USD, ratios)
     - Diversity (unique assets, action types)
     - Temporal (lifetime, activity frequency)
     - Volatility (std of daily tx counts)
     - Bot/risk patterns (e.g., borrow and redeem same day)
     - Optionally, applies time-decayed weights to recent transactions
3. **Preprocessing:**
   - Fills missing values, normalizes all numerical features
4. **Clustering/Outlier Detection:**
   - Groups wallets or flags outliers using the selected method (KMeans, DBSCAN, IsolationForest, Z-score, Mahalanobis, Autoencoder)
   - Computes cluster reputation for bonus/penalty
5. **Scoring:**
   - Weighted sum of normalized features, anomaly/cluster info, volatility, and cluster bonus/penalty
   - Final score is clipped to [0, 1000]
6. **Output:**
   - Saves a CSV (or JSON) with wallet address, score, cluster label, anomaly score, and all features

---

## Why Different Methods?

Each unsupervised method (KMeans, DBSCAN, IsolationForest, Z-score, Mahalanobis, Autoencoder) detects patterns, clusters, and outliers in fundamentally different ways.

### 1. Different Mathematical Approaches
- **KMeans:** Groups wallets into a fixed number of clusters based on feature similarity. Scores are influenced by distance to the “best” cluster centroid. Assumes clusters are spherical and similar in size.
- **DBSCAN:** Finds clusters of arbitrary shape and labels points in low-density regions as “noise” (outliers). If your data is mostly “noise” to DBSCAN, most wallets will be penalized, and only a few will be in clusters.
- **IsolationForest:** Treats outliers as those that are easiest to “isolate” in a random forest. Sensitive to the global distribution and can flag rare behaviors as highly anomalous.
- **Z-score/Mahalanobis:** Measures how far each wallet is from the “center” of the feature space. If your data is skewed, most points will be close to the mean, and a few will be far away.
- **Autoencoder:** Learns to reconstruct “normal” behavior. Wallets that are hard to reconstruct (i.e., unusual) get high anomaly scores.

### 2. Feature Sensitivity
- Each method reacts differently to the distribution and scaling of your features. If your data has many wallets with very similar behavior, some methods (like KMeans) may group them all together, while others (like IsolationForest or DBSCAN) may see most as outliers.

### 3. Parameter Sensitivity
- **KMeans:** Number of clusters (`n_clusters`)
- **DBSCAN:** `eps` (distance threshold), `min_samples`
- **IsolationForest:** `contamination` (expected outlier proportion)
- **Autoencoder:** Network size, training epochs, etc.

Small changes in these parameters can lead to very different groupings and, therefore, different score distributions.

### 4. Score Normalization
- After anomaly/cluster scoring, the pipeline normalizes scores to [0, 1000]. If the underlying anomaly/cluster scores are distributed differently, the normalization will stretch or compress the results accordingly.

### 5. Interpretation
- **No method is “right” or “wrong”**—each provides a different lens on the data. One may want to compare methods and choose the one that best matches the intuition or business goals (e.g., do you want to penalize rare behaviors, or reward cluster membership?).

---

## Extensibility & Transparency
- All logic is in a single, well-commented script
- Features and scoring logic are documented for auditability
- Easy to add new features, methods, or adjust weights

---

For setup and usage, see `setup.md`. For scoring logic and feature details, see below or the code comments. 