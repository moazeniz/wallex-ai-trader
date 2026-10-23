"""
Whale Profiler

Uses K-Means clustering to profile whale wallet behavior:
- Classify wallets by trading patterns
- Identify accumulation vs distribution
- Detect smart money movements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from dataclasses import dataclass
import pickle
from pathlib import Path
from datetime import datetime, timedelta


@dataclass
class WalletProfile:
    """Profile of a wallet based on clustering."""
    address: str
    cluster: int
    cluster_name: str
    features: Dict[str, float]
    confidence: float


class WhaleProfiler:
    """
    Whale wallet profiler using K-Means clustering.

    Classifies whale wallets by behavior patterns to identify
    smart money movements and predict accumulation/distribution.
    """

    # Cluster labels
    CLUSTER_NAMES = {
        0: "accumulator",      # Steady buyers
        1: "distributor",      # Steady sellers
        2: "trader",           # Active trading
        3: "holder",           # Long-term holder
        4: "smart_money",      # Profitable timing
    }

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize the whale profiler.

        Args:
            n_clusters: Number of behavior clusters
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

        self._model: Optional[KMeans] = None
        self._scaler: Optional[StandardScaler] = None
        self._feature_columns: List[str] = []
        self._cluster_centers: Optional[np.ndarray] = None
        self._cluster_labels: Dict[int, str] = {}
        self._fitted = False

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    def extract_wallet_features(
        self,
        transactions: List[Dict[str, Any]],
        current_balance: float,
        token_price: float = 1.0,
    ) -> Dict[str, float]:
        """
        Extract behavioral features from wallet transactions.

        Args:
            transactions: List of transaction dicts with 'type', 'amount', 'timestamp'
            current_balance: Current token balance
            token_price: Current token price for USD calculations

        Returns:
            Dictionary of behavioral features
        """
        if not transactions:
            return self._get_empty_features()

        df = pd.DataFrame(transactions)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

        # Calculate features
        features = {}

        # Transaction counts
        buys = df[df['type'] == 'buy'] if 'type' in df.columns else pd.DataFrame()
        sells = df[df['type'] == 'sell'] if 'type' in df.columns else pd.DataFrame()

        features['buy_count'] = len(buys)
        features['sell_count'] = len(sells)
        features['total_tx_count'] = len(df)

        # Buy/sell ratio
        if len(sells) > 0:
            features['buy_sell_ratio'] = len(buys) / len(sells)
        else:
            features['buy_sell_ratio'] = len(buys) if len(buys) > 0 else 1.0

        # Volume features
        if 'amount' in df.columns:
            total_bought = buys['amount'].sum() if len(buys) > 0 else 0
            total_sold = sells['amount'].sum() if len(sells) > 0 else 0
            features['net_flow'] = total_bought - total_sold
            features['total_volume'] = total_bought + total_sold
            features['buy_volume_pct'] = total_bought / (total_bought + total_sold) if (total_bought + total_sold) > 0 else 0.5
        else:
            features['net_flow'] = 0
            features['total_volume'] = 0
            features['buy_volume_pct'] = 0.5

        # Timing features
        if 'timestamp' in df.columns and len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            if len(time_diffs) > 0:
                avg_seconds = time_diffs.dt.total_seconds().mean()
                features['avg_tx_frequency'] = avg_seconds / 3600  # Hours between tx
            else:
                features['avg_tx_frequency'] = 168  # Weekly default
        else:
            features['avg_tx_frequency'] = 168

        # Recency
        if 'timestamp' in df.columns and len(df) > 0:
            last_tx = df['timestamp'].max()
            hours_since = (datetime.now() - last_tx).total_seconds() / 3600
            features['hours_since_last_tx'] = min(hours_since, 720)  # Cap at 30 days
        else:
            features['hours_since_last_tx'] = 720

        # Balance features
        features['current_balance'] = current_balance
        features['balance_usd'] = current_balance * token_price

        # Activity trend (recent vs old)
        if 'timestamp' in df.columns and len(df) > 1:
            mid_point = df['timestamp'].median()
            recent = df[df['timestamp'] >= mid_point]
            old = df[df['timestamp'] < mid_point]
            features['activity_trend'] = len(recent) / max(len(old), 1)
        else:
            features['activity_trend'] = 1.0

        return features

    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty feature dict."""
        return {
            'buy_count': 0,
            'sell_count': 0,
            'total_tx_count': 0,
            'buy_sell_ratio': 1.0,
            'net_flow': 0,
            'total_volume': 0,
            'buy_volume_pct': 0.5,
            'avg_tx_frequency': 168,
            'hours_since_last_tx': 720,
            'current_balance': 0,
            'balance_usd': 0,
            'activity_trend': 1.0,
        }

    def prepare_features_dataframe(
        self,
        wallet_features: List[Dict[str, float]],
    ) -> pd.DataFrame:
        """
        Prepare features DataFrame for clustering.

        Args:
            wallet_features: List of feature dicts per wallet

        Returns:
            DataFrame ready for clustering
        """
        df = pd.DataFrame(wallet_features)

        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._feature_columns = numeric_cols

        # Handle infinities and missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        return df[numeric_cols]

    # =========================================================================
    # Model Training
    # =========================================================================

    def fit(
        self,
        wallet_features: List[Dict[str, float]],
    ) -> "WhaleProfiler":
        """
        Fit the clustering model.

        Args:
            wallet_features: List of feature dicts per wallet

        Returns:
            Self for chaining
        """
        df = self.prepare_features_dataframe(wallet_features)

        # Scale features
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(df)

        # Fit K-Means
        self._model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self._model.fit(scaled)

        self._cluster_centers = self._model.cluster_centers_
        self._assign_cluster_labels(df)
        self._fitted = True

        return self

    def _assign_cluster_labels(self, df: pd.DataFrame) -> None:
        """Assign meaningful labels to clusters based on centers."""
        if self._cluster_centers is None:
            return

        # Get feature indices
        feature_indices = {col: i for i, col in enumerate(self._feature_columns)}

        for cluster_id in range(self.n_clusters):
            center = self._cluster_centers[cluster_id]

            # Determine cluster type based on feature values
            buy_sell_idx = feature_indices.get('buy_sell_ratio', -1)
            net_flow_idx = feature_indices.get('net_flow', -1)
            tx_count_idx = feature_indices.get('total_tx_count', -1)
            activity_idx = feature_indices.get('activity_trend', -1)

            # Simple heuristic classification
            if buy_sell_idx >= 0 and center[buy_sell_idx] > 0.5:
                if net_flow_idx >= 0 and center[net_flow_idx] > 0.3:
                    label = "accumulator"
                elif tx_count_idx >= 0 and center[tx_count_idx] > 0.5:
                    label = "trader"
                else:
                    label = "holder"
            elif buy_sell_idx >= 0 and center[buy_sell_idx] < -0.3:
                label = "distributor"
            elif activity_idx >= 0 and center[activity_idx] > 0.5:
                label = "smart_money"
            else:
                label = "holder"

            self._cluster_labels[cluster_id] = label

    def find_optimal_clusters(
        self,
        wallet_features: List[Dict[str, float]],
        k_range: Tuple[int, int] = (2, 10),
    ) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            wallet_features: List of feature dicts
            k_range: Range of k values to try

        Returns:
            Optimal number of clusters
        """
        df = self.prepare_features_dataframe(wallet_features)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)

        best_k = k_range[0]
        best_score = -1

        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(scaled)
            score = silhouette_score(scaled, labels)

            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    # =========================================================================
    # Prediction
    # =========================================================================

    def profile_wallet(
        self,
        features: Dict[str, float],
        address: str = "unknown",
    ) -> WalletProfile:
        """
        Profile a single wallet.

        Args:
            features: Wallet feature dictionary
            address: Wallet address

        Returns:
            WalletProfile with cluster assignment
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        # Prepare features
        df = pd.DataFrame([features])
        df = df.reindex(columns=self._feature_columns, fill_value=0)

        # Scale
        scaled = self._scaler.transform(df)

        # Predict cluster
        cluster = self._model.predict(scaled)[0]

        # Get distance to cluster center for confidence
        distances = self._model.transform(scaled)[0]
        min_dist = distances[cluster]
        max_dist = distances.max()
        confidence = 1 - (min_dist / max_dist) if max_dist > 0 else 1.0

        cluster_name = self._cluster_labels.get(cluster, f"cluster_{cluster}")

        return WalletProfile(
            address=address,
            cluster=int(cluster),
            cluster_name=cluster_name,
            features=features,
            confidence=float(confidence),
        )

    def profile_wallets(
        self,
        wallet_data: List[Tuple[str, Dict[str, float]]],
    ) -> List[WalletProfile]:
        """
        Profile multiple wallets.

        Args:
            wallet_data: List of (address, features) tuples

        Returns:
            List of WalletProfiles
        """
        return [
            self.profile_wallet(features, address)
            for address, features in wallet_data
        ]

    def get_cluster_distribution(
        self,
        profiles: List[WalletProfile],
    ) -> Dict[str, int]:
        """Get distribution of wallets across clusters."""
        distribution = {}
        for profile in profiles:
            name = profile.cluster_name
            distribution[name] = distribution.get(name, 0) + 1
        return distribution

    def get_accumulation_score(
        self,
        profiles: List[WalletProfile],
    ) -> float:
        """
        Calculate overall accumulation score from profiles.

        Args:
            profiles: List of wallet profiles

        Returns:
            Score from 0 (distribution) to 100 (accumulation)
        """
        if not profiles:
            return 50.0

        weights = {
            "accumulator": 100,
            "smart_money": 80,
            "holder": 60,
            "trader": 50,
            "distributor": 0,
        }

        total_score = 0
        total_confidence = 0

        for profile in profiles:
            weight = weights.get(profile.cluster_name, 50)
            total_score += weight * profile.confidence
            total_confidence += profile.confidence

        if total_confidence > 0:
            return total_score / total_confidence
        return 50.0

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        """Save model to disk."""
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_columns": self._feature_columns,
            "cluster_centers": self._cluster_centers,
            "cluster_labels": self._cluster_labels,
            "n_clusters": self.n_clusters,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> "WhaleProfiler":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]
        self._scaler = data["scaler"]
        self._feature_columns = data["feature_columns"]
        self._cluster_centers = data["cluster_centers"]
        self._cluster_labels = data["cluster_labels"]
        self.n_clusters = data["n_clusters"]
        self._fitted = True

        return self


def create_whale_profiler(n_clusters: int = 5) -> WhaleProfiler:
    """Create a WhaleProfiler instance."""
    return WhaleProfiler(n_clusters=n_clusters)


if __name__ == "__main__":
    print("Testing WhaleProfiler...")

    # Create sample wallet data
    np.random.seed(42)

    # Generate synthetic wallet transactions
    wallets = []
    for i in range(100):
        # Random wallet type
        wallet_type = np.random.choice(['accumulator', 'distributor', 'trader', 'holder'])

        if wallet_type == 'accumulator':
            features = {
                'buy_count': np.random.randint(10, 50),
                'sell_count': np.random.randint(1, 10),
                'total_tx_count': np.random.randint(15, 60),
                'buy_sell_ratio': np.random.uniform(2, 10),
                'net_flow': np.random.uniform(1000, 10000),
                'total_volume': np.random.uniform(2000, 15000),
                'buy_volume_pct': np.random.uniform(0.7, 0.95),
                'avg_tx_frequency': np.random.uniform(24, 72),
                'hours_since_last_tx': np.random.uniform(1, 48),
                'current_balance': np.random.uniform(10000, 100000),
                'balance_usd': np.random.uniform(50000, 500000),
                'activity_trend': np.random.uniform(1.2, 2.0),
            }
        elif wallet_type == 'distributor':
            features = {
                'buy_count': np.random.randint(1, 10),
                'sell_count': np.random.randint(10, 50),
                'total_tx_count': np.random.randint(15, 60),
                'buy_sell_ratio': np.random.uniform(0.1, 0.5),
                'net_flow': np.random.uniform(-10000, -1000),
                'total_volume': np.random.uniform(2000, 15000),
                'buy_volume_pct': np.random.uniform(0.1, 0.3),
                'avg_tx_frequency': np.random.uniform(24, 72),
                'hours_since_last_tx': np.random.uniform(1, 48),
                'current_balance': np.random.uniform(1000, 10000),
                'balance_usd': np.random.uniform(5000, 50000),
                'activity_trend': np.random.uniform(0.5, 1.0),
            }
        elif wallet_type == 'trader':
            features = {
                'buy_count': np.random.randint(50, 200),
                'sell_count': np.random.randint(50, 200),
                'total_tx_count': np.random.randint(100, 400),
                'buy_sell_ratio': np.random.uniform(0.8, 1.2),
                'net_flow': np.random.uniform(-1000, 1000),
                'total_volume': np.random.uniform(50000, 200000),
                'buy_volume_pct': np.random.uniform(0.4, 0.6),
                'avg_tx_frequency': np.random.uniform(1, 12),
                'hours_since_last_tx': np.random.uniform(0.5, 12),
                'current_balance': np.random.uniform(5000, 50000),
                'balance_usd': np.random.uniform(25000, 250000),
                'activity_trend': np.random.uniform(0.9, 1.5),
            }
        else:  # holder
            features = {
                'buy_count': np.random.randint(1, 5),
                'sell_count': np.random.randint(0, 2),
                'total_tx_count': np.random.randint(1, 7),
                'buy_sell_ratio': np.random.uniform(2, 20),
                'net_flow': np.random.uniform(5000, 50000),
                'total_volume': np.random.uniform(5000, 50000),
                'buy_volume_pct': np.random.uniform(0.9, 1.0),
                'avg_tx_frequency': np.random.uniform(168, 720),
                'hours_since_last_tx': np.random.uniform(168, 720),
                'current_balance': np.random.uniform(50000, 500000),
                'balance_usd': np.random.uniform(250000, 2500000),
                'activity_trend': np.random.uniform(0.3, 0.7),
            }

        wallets.append(features)

    # Fit profiler
    profiler = WhaleProfiler(n_clusters=4)
    profiler.fit(wallets)

    # Profile wallets
    wallet_data = [(f"0x{i:040x}", w) for i, w in enumerate(wallets)]
    profiles = profiler.profile_wallets(wallet_data)

    # Get distribution
    distribution = profiler.get_cluster_distribution(profiles)
    print(f"\nCluster distribution: {distribution}")

    # Get accumulation score
    acc_score = profiler.get_accumulation_score(profiles)
    print(f"Overall accumulation score: {acc_score:.1f}")

    # Test single wallet
    single_profile = profiler.profile_wallet(wallets[0], "0xTest")
    print(f"\nSingle wallet profile: {single_profile.cluster_name} (confidence: {single_profile.confidence:.2f})")
