"""
Enhanced Feature Extractor

Adds high-value features beyond basic technical indicators:
- Funding rate (crypto perpetuals)
- Open Interest changes
- Multi-timeframe features (1h, 4h, 1d combined)
- Cross-asset correlations
- Whale activity indicators
- Market regime features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/wallex')


class EnhancedFeatureExtractor:
    """
    Extracts enhanced features for ML models.

    Features are grouped by category:
    - Price features (normalized, returns, volatility)
    - Technical indicators (RSI, MACD, BB, etc.)
    - Market microstructure (funding, OI)
    - Multi-timeframe (1h, 4h, 1d trends)
    - Cross-asset (BTC correlation, dominance)
    - Regime indicators
    """

    def __init__(self):
        self.feature_names: List[str] = []
        self._bybit_client = None
        self._exchange_client = None

    @property
    def bybit_client(self):
        if self._bybit_client is None:
            try:
                from data.bybit_client import BybitDataClient
                self._bybit_client = BybitDataClient()
            except Exception as e:
                print(f"Warning: BybitDataClient not available: {e}")
        return self._bybit_client

    @property
    def exchange_client(self):
        if self._exchange_client is None:
            try:
                from data.exchange_client import ExchangeClient
                self._exchange_client = ExchangeClient()
            except:
                pass
        return self._exchange_client

    def extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic price-based features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # Normalized price (relative to SMAs)
        for period in [10, 20, 50]:
            sma = close.rolling(period).mean()
            features[f'price_vs_sma_{period}'] = (close - sma) / (sma + 1e-10)

        # Returns at multiple horizons
        for period in [1, 4, 12, 24]:
            features[f'return_{period}h'] = close.pct_change(period)

        # Volatility at multiple windows
        for window in [6, 12, 24, 48]:
            features[f'volatility_{window}h'] = close.pct_change().rolling(window).std()

        # Trend strength
        features['trend_strength'] = (close.rolling(5).mean() - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-10)

        return features

    def extract_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Technical indicator features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']
        high = df.get('high', close)
        low = df.get('low', close)
        volume = df.get('volume', pd.Series(1, index=df.index))

        # RSI at multiple periods
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            features[f'rsi_{period}'] = (100 - (100 / (1 + rs))) / 100  # Normalized 0-1

        # RSI divergence (price vs RSI trend)
        features['rsi_divergence'] = features['rsi_14'].diff(5) - close.pct_change(5)

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        features['macd_norm'] = macd / (close + 1e-10)
        features['macd_hist'] = (macd - macd_signal) / (close + 1e-10)
        features['macd_crossover'] = (macd > macd_signal).astype(int).diff()

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)
        features['bb_width'] = (4 * bb_std) / (bb_mid + 1e-10)  # Volatility indicator

        # Volume features
        vol_ma = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (vol_ma + 1e-10)
        features['volume_trend'] = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-10)
        features['price_volume_corr'] = close.pct_change().rolling(20).corr(volume.pct_change())

        # ATR (Average True Range)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        features['atr_ratio'] = tr.rolling(14).mean() / (close + 1e-10)

        # Momentum
        features['momentum_12'] = close / close.shift(12) - 1
        features['momentum_24'] = close / close.shift(24) - 1
        features['momentum_acceleration'] = features['momentum_12'].diff(6)

        # Support/Resistance proximity
        features['dist_from_high_24'] = (close - high.rolling(24).max()) / (close + 1e-10)
        features['dist_from_low_24'] = (close - low.rolling(24).min()) / (close + 1e-10)
        features['dist_from_high_72'] = (close - high.rolling(72).max()) / (close + 1e-10)
        features['dist_from_low_72'] = (close - low.rolling(72).min()) / (close + 1e-10)

        return features

    def extract_funding_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Funding rate and Open Interest features using Bybit API."""
        features = pd.DataFrame(index=df.index)

        # Initialize with zeros (will be filled if data available)
        features['funding_rate'] = 0.0
        features['funding_ma'] = 0.0
        features['funding_zscore'] = 0.0
        features['oi_change_1h'] = 0.0
        features['oi_change_24h'] = 0.0
        features['oi_price_divergence'] = 0.0

        if self.bybit_client is None:
            return features

        try:
            # Get all derivatives data from Bybit
            base_symbol = symbol.replace('/USDT', '').replace('USDT', '').replace('/', '')
            data = self.bybit_client.get_all_derivatives_data(base_symbol)

            if data:
                # Funding features
                features['funding_rate'] = data.get('funding_rate', 0.0)
                features['funding_ma'] = data.get('funding_ma', 0.0)
                features['funding_zscore'] = data.get('funding_zscore', 0.0)
                
                # Open Interest features
                features['oi_change_1h'] = data.get('oi_change_1h', 0.0)
                features['oi_change_24h'] = data.get('oi_change_24h', 0.0)

                # OI-price divergence
                price_change = df['close'].pct_change(24).iloc[-1] if len(df) > 24 else 0
                features['oi_price_divergence'] = features['oi_change_24h'].iloc[0] * np.sign(price_change)

        except Exception as e:
            print(f"Warning: Bybit data fetch failed for {symbol}: {e}")

        return features

    def extract_multiframe_features(
        self,
        df_1h: pd.DataFrame,
        df_4h: Optional[pd.DataFrame] = None,
        df_1d: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Multi-timeframe trend alignment features."""
        features = pd.DataFrame(index=df_1h.index)

        # 1h trend
        close_1h = df_1h['close']
        sma_20_1h = close_1h.rolling(20).mean()
        sma_50_1h = close_1h.rolling(50).mean()
        features['trend_1h'] = ((close_1h > sma_20_1h) & (sma_20_1h > sma_50_1h)).astype(int) - \
                               ((close_1h < sma_20_1h) & (sma_20_1h < sma_50_1h)).astype(int)

        # 4h trend (if available)
        if df_4h is not None and len(df_4h) > 0:
            close_4h = df_4h['close']
            sma_20_4h = close_4h.rolling(20).mean()
            trend_4h = (close_4h > sma_20_4h).astype(int) * 2 - 1
            # Map 4h values to 1h timeframe (forward fill)
            features['trend_4h'] = 0.0
            try:
                # Simple approach: use last known 4h trend
                features['trend_4h'] = float(trend_4h.iloc[-1]) if len(trend_4h) > 0 else 0
            except:
                pass
        else:
            features['trend_4h'] = 0.0

        # 1d trend (if available)
        if df_1d is not None and len(df_1d) > 0:
            close_1d = df_1d['close']
            sma_10_1d = close_1d.rolling(10).mean()
            trend_1d = (close_1d > sma_10_1d).astype(int) * 2 - 1
            features['trend_1d'] = float(trend_1d.iloc[-1]) if len(trend_1d) > 0 else 0
        else:
            features['trend_1d'] = 0.0

        # Trend alignment score (-3 to +3)
        features['trend_alignment'] = features['trend_1h'] + features['trend_4h'] + features['trend_1d']

        # Multi-timeframe momentum
        features['mtf_momentum'] = (
            close_1h.pct_change(4) * 0.5 +  # Short-term
            close_1h.pct_change(24) * 0.3 +  # Medium-term
            close_1h.pct_change(72) * 0.2    # Long-term
        )

        return features

    def extract_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market regime detection features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # Volatility regime
        vol_short = close.pct_change().rolling(12).std()
        vol_long = close.pct_change().rolling(48).std()
        features['vol_regime'] = vol_short / (vol_long + 1e-10)  # >1 = increasing vol

        # Trend regime (ADX-like)
        high = df.get('high', close)
        low = df.get('low', close)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr + 1e-10)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        features['adx'] = dx.rolling(14).mean() / 100  # Normalized 0-1

        # Range detection
        range_high = high.rolling(24).max()
        range_low = low.rolling(24).min()
        features['range_position'] = (close - range_low) / (range_high - range_low + 1e-10)

        # Mean reversion vs trending
        features['mean_reversion_score'] = -features['range_position'].diff(6)  # Positive = reverting to mean

        return features

    def extract_cross_asset_features(
        self,
        df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Cross-asset correlation features."""
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # Initialize with defaults
        features['btc_correlation'] = 0.0
        features['btc_beta'] = 1.0
        features['relative_strength'] = 0.0

        if btc_df is not None and len(btc_df) > 0:
            try:
                btc_close = btc_df['close']

                # Align indices (take last N that match)
                min_len = min(len(close), len(btc_close))
                if min_len > 24:
                    close_aligned = close.iloc[-min_len:].reset_index(drop=True)
                    btc_aligned = btc_close.iloc[-min_len:].reset_index(drop=True)

                    # Rolling correlation with BTC
                    corr = close_aligned.pct_change().rolling(24).corr(btc_aligned.pct_change())
                    features['btc_correlation'] = float(corr.iloc[-1]) if len(corr) > 0 else 0

                    # Beta to BTC
                    cov = close_aligned.pct_change().rolling(24).cov(btc_aligned.pct_change())
                    btc_var = btc_aligned.pct_change().rolling(24).var()
                    beta = cov / (btc_var + 1e-10)
                    features['btc_beta'] = float(beta.iloc[-1]) if len(beta) > 0 else 1

                    # Relative strength vs BTC
                    token_return = close_aligned.pct_change(24)
                    btc_return = btc_aligned.pct_change(24)
                    rs = token_return - btc_return
                    features['relative_strength'] = float(rs.iloc[-1]) if len(rs) > 0 else 0
            except:
                pass

        return features

    def extract_all_features(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        df_4h: pd.DataFrame = None,
        df_1d: pd.DataFrame = None,
        btc_df: pd.DataFrame = None,
        include_funding: bool = True
    ) -> pd.DataFrame:
        """Extract all enhanced features."""

        # Price features
        price_features = self.extract_price_features(df)

        # Technical features
        tech_features = self.extract_technical_features(df)

        # Funding/OI features (crypto only)
        if include_funding and symbol and 'USD' in symbol:
            funding_features = self.extract_funding_features(symbol, df)
        else:
            funding_features = pd.DataFrame(index=df.index)

        # Multi-timeframe features
        mtf_features = self.extract_multiframe_features(df, df_4h, df_1d)

        # Regime features
        regime_features = self.extract_regime_features(df)

        # Cross-asset features
        cross_features = self.extract_cross_asset_features(df, btc_df)

        # Combine all
        all_features = pd.concat([
            price_features,
            tech_features,
            funding_features,
            mtf_features,
            regime_features,
            cross_features
        ], axis=1)

        # Store feature names
        self.feature_names = all_features.columns.tolist()

        # Fill NaN
        all_features = all_features.fillna(0)

        # Replace inf with large values
        all_features = all_features.replace([np.inf, -np.inf], 0)

        return all_features

    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.feature_names)

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by category."""
        groups = {
            'price': [f for f in self.feature_names if 'price' in f or 'return' in f or 'volatility' in f],
            'technical': [f for f in self.feature_names if any(x in f for x in ['rsi', 'macd', 'bb_', 'volume', 'atr', 'momentum'])],
            'funding': [f for f in self.feature_names if 'funding' in f or 'oi_' in f],
            'multiframe': [f for f in self.feature_names if 'trend_' in f or 'mtf_' in f],
            'regime': [f for f in self.feature_names if 'regime' in f or 'adx' in f or 'range' in f or 'mean_reversion' in f],
            'cross_asset': [f for f in self.feature_names if 'btc_' in f or 'relative' in f],
        }
        return groups


# Singleton
_extractor = None

def get_feature_extractor() -> EnhancedFeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = EnhancedFeatureExtractor()
    return _extractor
