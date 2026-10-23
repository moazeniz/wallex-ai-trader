"""
Pattern Matcher

Finds similar historical patterns for current market conditions:
- Price pattern similarity
- Feature-based pattern matching
- Outcome prediction based on historical matches
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from datetime import datetime
import pickle
from pathlib import Path


@dataclass
class PatternMatch:
    """A matched historical pattern."""
    match_index: int
    match_date: datetime
    similarity: float  # 0-1, higher is more similar
    outcome_return: float  # Return after pattern (in %)
    outcome_direction: str  # "up", "down", "sideways"
    pattern_features: Dict[str, float]


@dataclass
class PatternAnalysis:
    """Analysis of pattern matches."""
    current_features: Dict[str, float]
    matches: List[PatternMatch]
    avg_similarity: float
    bullish_probability: float  # % of matches that went up
    avg_return: float  # Average return after similar patterns
    expected_direction: str
    confidence: float


class PatternMatcher:
    """
    Historical pattern matcher.

    Finds similar historical setups and analyzes their outcomes
    to predict potential future movements.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        pattern_length: int = 20,
        forward_length: int = 5,
    ):
        """
        Initialize the pattern matcher.

        Args:
            n_neighbors: Number of similar patterns to find
            pattern_length: Length of pattern to match (lookback)
            forward_length: Bars to look ahead for outcome
        """
        self.n_neighbors = n_neighbors
        self.pattern_length = pattern_length
        self.forward_length = forward_length

        self._model: Optional[NearestNeighbors] = None
        self._scaler: Optional[StandardScaler] = None
        self._pattern_store: Optional[pd.DataFrame] = None
        self._outcomes: Optional[np.ndarray] = None
        self._dates: Optional[List[datetime]] = None
        self._feature_columns: List[str] = []

        self._fitted = False

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    def _extract_pattern_features(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> Dict[str, float]:
        """
        Extract features for a single pattern.

        Args:
            df: Full OHLCV DataFrame
            index: Index to extract pattern ending at

        Returns:
            Feature dictionary
        """
        if index < self.pattern_length:
            return {}

        window = df.iloc[index - self.pattern_length:index]

        features = {}

        # Normalized price changes
        close = window['close'].values
        norm_close = (close - close[0]) / close[0] * 100

        for i, pct in enumerate([25, 50, 75, 100]):
            idx = int(len(norm_close) * pct / 100) - 1
            features[f'price_pct_{pct}'] = norm_close[idx]

        # Returns at different points
        features['return_total'] = (close[-1] - close[0]) / close[0] * 100
        features['return_first_half'] = (close[len(close)//2] - close[0]) / close[0] * 100
        features['return_second_half'] = (close[-1] - close[len(close)//2]) / close[len(close)//2] * 100

        # Volatility
        returns = pd.Series(close).pct_change().dropna()
        features['volatility'] = returns.std() * 100

        # Trend consistency
        up_moves = (returns > 0).sum()
        features['trend_consistency'] = up_moves / len(returns) if len(returns) > 0 else 0.5

        # High/Low position
        high = window['high'].max()
        low = window['low'].min()
        features['price_position'] = (close[-1] - low) / (high - low) if high != low else 0.5

        # Volume trend (if available)
        if 'volume' in window.columns:
            vol = window['volume'].values
            vol_first = vol[:len(vol)//2].mean()
            vol_second = vol[len(vol)//2:].mean()
            features['volume_trend'] = vol_second / vol_first if vol_first > 0 else 1.0

        # Pattern shape metrics
        features['max_drawdown'] = ((pd.Series(close).cummax() - close) / pd.Series(close).cummax()).max() * 100
        features['max_runup'] = ((close - pd.Series(close).cummin()) / pd.Series(close).cummin()).max() * 100

        return features

    def _calculate_outcome(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> Tuple[float, str]:
        """
        Calculate the outcome after a pattern.

        Args:
            df: Full DataFrame
            index: Pattern end index

        Returns:
            Tuple of (return_pct, direction)
        """
        if index + self.forward_length >= len(df):
            return 0.0, "unknown"

        start_price = df.iloc[index]['close']
        end_price = df.iloc[index + self.forward_length]['close']

        return_pct = (end_price - start_price) / start_price * 100

        if return_pct > 1.0:
            direction = "up"
        elif return_pct < -1.0:
            direction = "down"
        else:
            direction = "sideways"

        return return_pct, direction

    # =========================================================================
    # Model Training
    # =========================================================================

    def fit(self, df: pd.DataFrame) -> "PatternMatcher":
        """
        Build pattern database from historical data.

        Args:
            df: OHLCV DataFrame

        Returns:
            Self for chaining
        """
        patterns = []
        outcomes = []
        dates = []

        # Extract all patterns
        for i in range(self.pattern_length, len(df) - self.forward_length):
            features = self._extract_pattern_features(df, i)
            if features:
                patterns.append(features)
                outcome_return, outcome_dir = self._calculate_outcome(df, i)
                outcomes.append((outcome_return, outcome_dir))
                dates.append(df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now())

        if not patterns:
            raise ValueError("Not enough data to extract patterns")

        # Create DataFrame
        self._pattern_store = pd.DataFrame(patterns)
        self._feature_columns = self._pattern_store.columns.tolist()
        self._outcomes = outcomes
        self._dates = dates

        # Scale features
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(self._pattern_store)

        # Fit nearest neighbors
        self._model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(patterns)),
            metric='euclidean',
        )
        self._model.fit(scaled)

        self._fitted = True
        return self

    # =========================================================================
    # Pattern Matching
    # =========================================================================

    def find_similar_patterns(
        self,
        df: pd.DataFrame,
        n_matches: Optional[int] = None,
    ) -> List[PatternMatch]:
        """
        Find similar historical patterns.

        Args:
            df: Current OHLCV DataFrame
            n_matches: Number of matches to return

        Returns:
            List of PatternMatch objects
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        n_matches = n_matches or self.n_neighbors

        # Extract current pattern features
        current_features = self._extract_pattern_features(df, len(df))
        if not current_features:
            return []

        # Prepare for matching
        feature_vector = pd.DataFrame([current_features]).reindex(
            columns=self._feature_columns, fill_value=0
        )
        scaled = self._scaler.transform(feature_vector)

        # Find neighbors
        distances, indices = self._model.kneighbors(
            scaled, n_neighbors=min(n_matches, len(self._pattern_store))
        )

        # Build matches
        matches = []
        max_dist = distances[0].max() if distances[0].max() > 0 else 1

        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            similarity = 1 - (dist / max_dist)
            outcome_return, outcome_dir = self._outcomes[idx]

            match = PatternMatch(
                match_index=int(idx),
                match_date=self._dates[idx],
                similarity=float(similarity),
                outcome_return=float(outcome_return),
                outcome_direction=outcome_dir,
                pattern_features=self._pattern_store.iloc[idx].to_dict(),
            )
            matches.append(match)

        return matches

    def analyze_patterns(
        self,
        df: pd.DataFrame,
        n_matches: Optional[int] = None,
    ) -> PatternAnalysis:
        """
        Analyze current pattern against historical matches.

        Args:
            df: Current OHLCV DataFrame
            n_matches: Number of matches to analyze

        Returns:
            PatternAnalysis with predictions
        """
        matches = self.find_similar_patterns(df, n_matches)

        if not matches:
            return PatternAnalysis(
                current_features={},
                matches=[],
                avg_similarity=0,
                bullish_probability=0.5,
                avg_return=0,
                expected_direction="unknown",
                confidence=0,
            )

        # Calculate statistics
        avg_similarity = np.mean([m.similarity for m in matches])
        returns = [m.outcome_return for m in matches]
        avg_return = np.mean(returns)

        # Direction probabilities
        up_count = sum(1 for m in matches if m.outcome_direction == "up")
        down_count = sum(1 for m in matches if m.outcome_direction == "down")
        bullish_prob = up_count / len(matches)

        # Expected direction
        if bullish_prob > 0.6:
            expected_direction = "bullish"
        elif bullish_prob < 0.4:
            expected_direction = "bearish"
        else:
            expected_direction = "neutral"

        # Confidence based on similarity and consistency
        direction_consistency = max(bullish_prob, 1 - bullish_prob)
        confidence = avg_similarity * 0.5 + direction_consistency * 0.5

        current_features = self._extract_pattern_features(df, len(df))

        return PatternAnalysis(
            current_features=current_features,
            matches=matches,
            avg_similarity=avg_similarity,
            bullish_probability=bullish_prob,
            avg_return=avg_return,
            expected_direction=expected_direction,
            confidence=confidence,
        )

    def get_pattern_summary(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Get a summary of pattern analysis.

        Args:
            df: Current OHLCV DataFrame

        Returns:
            Dictionary with pattern analysis summary
        """
        analysis = self.analyze_patterns(df)

        # Calculate win rate by direction
        if analysis.matches:
            positive_returns = sum(1 for m in analysis.matches if m.outcome_return > 0)
            win_rate = positive_returns / len(analysis.matches)
        else:
            win_rate = 0.5

        return {
            "n_matches": len(analysis.matches),
            "avg_similarity": analysis.avg_similarity,
            "bullish_probability": analysis.bullish_probability,
            "bearish_probability": 1 - analysis.bullish_probability,
            "avg_expected_return": analysis.avg_return,
            "expected_direction": analysis.expected_direction,
            "historical_win_rate": win_rate,
            "confidence": analysis.confidence,
            "top_matches": [
                {
                    "date": m.match_date.isoformat() if hasattr(m.match_date, 'isoformat') else str(m.match_date),
                    "similarity": m.similarity,
                    "outcome": m.outcome_return,
                    "direction": m.outcome_direction,
                }
                for m in analysis.matches[:3]
            ],
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        """Save model to disk."""
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "pattern_store": self._pattern_store,
            "outcomes": self._outcomes,
            "dates": self._dates,
            "feature_columns": self._feature_columns,
            "config": {
                "n_neighbors": self.n_neighbors,
                "pattern_length": self.pattern_length,
                "forward_length": self.forward_length,
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> "PatternMatcher":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]
        self._scaler = data["scaler"]
        self._pattern_store = data["pattern_store"]
        self._outcomes = data["outcomes"]
        self._dates = data["dates"]
        self._feature_columns = data["feature_columns"]

        config = data["config"]
        self.n_neighbors = config["n_neighbors"]
        self.pattern_length = config["pattern_length"]
        self.forward_length = config["forward_length"]

        self._fitted = True
        return self


def create_pattern_matcher(
    n_neighbors: int = 10,
    pattern_length: int = 20,
    forward_length: int = 5,
) -> PatternMatcher:
    """Create a PatternMatcher instance."""
    return PatternMatcher(
        n_neighbors=n_neighbors,
        pattern_length=pattern_length,
        forward_length=forward_length,
    )


if __name__ == "__main__":
    print("Testing PatternMatcher...")

    # Create sample data with recognizable patterns
    np.random.seed(42)
    n = 1000

    # Create price series with some repeating patterns
    close = [100]
    for i in range(1, n):
        # Add some pattern structure
        if i % 50 < 25:
            # Uptrend pattern
            change = np.random.randn() * 0.3 + 0.1
        else:
            # Downtrend pattern
            change = np.random.randn() * 0.3 - 0.05

        close.append(close[-1] * (1 + change / 100))

    close = np.array(close)
    dates = pd.date_range("2024-01-01", periods=n, freq="1H")

    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    # Test pattern matcher
    matcher = PatternMatcher(n_neighbors=10, pattern_length=20, forward_length=5)
    matcher.fit(df.iloc[:-50])  # Fit on historical data

    # Match on recent data
    summary = matcher.get_pattern_summary(df.iloc[-50:])

    print(f"\nPattern Analysis Summary:")
    print(f"  Matches found: {summary['n_matches']}")
    print(f"  Average similarity: {summary['avg_similarity']:.2f}")
    print(f"  Bullish probability: {summary['bullish_probability']:.1%}")
    print(f"  Expected direction: {summary['expected_direction']}")
    print(f"  Historical win rate: {summary['historical_win_rate']:.1%}")
    print(f"  Confidence: {summary['confidence']:.2f}")

    print(f"\n  Top 3 matches:")
    for match in summary['top_matches']:
        print(f"    - Date: {match['date'][:10]}, Similarity: {match['similarity']:.2f}, Outcome: {match['outcome']:.1f}%")
