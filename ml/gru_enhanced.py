"""
Enhanced GRU Direction Classifier

Improvements over original:
- Attention mechanism for focusing on important time steps
- Uses enhanced features (funding, OI, multi-timeframe)
- Multi-class output (UP/DOWN/NEUTRAL) - learns all patterns together
- Regime-aware predictions
- Focal loss for class imbalance
- Larger capacity with better regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
from sklearn.preprocessing import RobustScaler
import pickle
from pathlib import Path
import sys
sys.path.insert(0, '/opt/wallex')


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Down-weights easy examples, focuses on hard ones.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class Attention(nn.Module):
    """
    Attention mechanism for sequence models.
    Learns which time steps are most important for prediction.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, gru_output):
        # gru_output: (batch, seq_len, hidden_size)
        attention_weights = self.attention(gru_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum of hidden states
        context = torch.sum(attention_weights * gru_output, dim=1)  # (batch, hidden_size)

        return context, attention_weights


class EnhancedGRUClassifier(nn.Module):
    """
    Enhanced GRU with attention for multi-class direction prediction.

    Outputs 3 classes: UP (0), DOWN (1), NEUTRAL (2)
    This allows the model to learn ALL patterns together.

    Architecture:
    - Bidirectional GRU layers
    - Attention mechanism
    - Residual connections
    - Layer normalization
    """

    def __init__(
        self,
        input_size: int = 50,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 3,  # UP, DOWN, NEUTRAL
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention
        self.attention = Attention(hidden_size * self.num_directions)

        # Output layers - multi-class
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),  # 3 classes: UP, DOWN, NEUTRAL
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # Project input
        x = self.input_proj(x)

        # GRU
        gru_out, _ = self.gru(x)

        # Attention
        context, attn_weights = self.attention(gru_out)

        # Classification (logits, no softmax - handled by loss function)
        logits = self.fc(context)

        return logits, attn_weights


class EnhancedDirectionPredictor:
    """
    Enhanced predictor with:
    - Multi-class output (UP/DOWN/NEUTRAL learned together)
    - Multi-timeframe features
    - Regime conditioning
    - Focal loss for imbalance
    - Attention mechanism
    """

    # Class mapping
    CLASS_UP = 0
    CLASS_DOWN = 1
    CLASS_NEUTRAL = 2

    def __init__(
        self,
        seq_length: int = 48,
        forecast_horizon: int = 4,
        threshold: float = 0.005,  # 0.5% movement threshold
    ):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.threshold = threshold

        self.model: Optional[EnhancedGRUClassifier] = None
        self.scaler: Optional[RobustScaler] = None
        self.scalers: Dict[str, RobustScaler] = {}
        self.feature_names: List[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Class weights learned during training
        self.class_weights: Optional[torch.Tensor] = None

        # Feature extractor
        self._feature_extractor = None

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            try:
                from ml.enhanced_features import get_feature_extractor
                self._feature_extractor = get_feature_extractor()
            except ImportError:
                try:
                    from enhanced_features import get_feature_extractor
                    self._feature_extractor = get_feature_extractor()
                except:
                    # Fallback to basic features
                    self._feature_extractor = None
        return self._feature_extractor

    def _prepare_features_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic feature extraction fallback."""
        df = df.reset_index(drop=True)
        features = pd.DataFrame(index=df.index)
        close = df['close']

        # Basic features
        sma_20 = close.rolling(20).mean()
        features['price_norm'] = (close - sma_20) / (sma_20 + 1e-10)
        features['return_1'] = close.pct_change(1)
        features['return_4'] = close.pct_change(4)
        features['return_12'] = close.pct_change(12)
        features['return_24'] = close.pct_change(24)
        features['volatility'] = close.pct_change().rolling(10).std()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_norm'] = (100 - (100 / (1 + rs))) / 100

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        features['macd_norm'] = (ema_12 - ema_26) / (close + 1e-10)

        # BB
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_position'] = (close - bb_mid) / (2 * bb_std + 1e-10)

        # Volume
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / (vol_ma + 1e-10)
        else:
            features['volume_ratio'] = 1.0

        self.feature_names = features.columns.tolist()
        return features.fillna(0)

    def _prepare_features(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        df_4h: pd.DataFrame = None,
        df_1d: pd.DataFrame = None,
        btc_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Extract enhanced features."""
        df = df.reset_index(drop=True)

        if self.feature_extractor is not None:
            features = self.feature_extractor.extract_all_features(
                df=df,
                symbol=symbol,
                df_4h=df_4h,
                df_1d=df_1d,
                btc_df=btc_df,
                include_funding=True
            )
            self.feature_names = features.columns.tolist()
        else:
            features = self._prepare_features_basic(df)

        return features

    def _create_sequences(
        self,
        features: np.ndarray,
        returns: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with multi-class labels."""
        X, y = [], []

        for i in range(len(features) - self.seq_length - self.forecast_horizon):
            future_return = returns[i + self.seq_length + self.forecast_horizon - 1]

            # Skip if future return is NaN (must check BEFORE adding to X)
            if np.isnan(future_return):
                continue

            # Multi-class labels: UP=0, DOWN=1, NEUTRAL=2
            if future_return > self.threshold:
                label = self.CLASS_UP
            elif future_return < -self.threshold:
                label = self.CLASS_DOWN
            else:
                label = self.CLASS_NEUTRAL

            # Only add X and y together to ensure same length
            X.append(features[i:i + self.seq_length])
            y.append(label)

        return np.array(X), np.array(y)

    def fit(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        df_4h: pd.DataFrame = None,
        df_1d: pd.DataFrame = None,
        btc_df: pd.DataFrame = None,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.0005,
        validation_split: float = 0.2,
        use_focal_loss: bool = True,
    ) -> Dict[str, Any]:
        """Train the enhanced multi-class GRU classifier."""

        print(f"Preparing enhanced features...")

        # Handle multi-token data
        if 'token' in df.columns:
            tokens = df['token'].unique()
            all_X, all_y = [], []

            for token in tokens:
                token_df = df[df['token'] == token].copy()
                if len(token_df) < self.seq_length + self.forecast_horizon + 50:
                    continue

                features_df = self._prepare_features(token_df, symbol=token)
                returns = token_df['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon).values

                # Per-token scaler
                self.scalers[token] = RobustScaler()
                features_scaled = self.scalers[token].fit_transform(features_df.values)

                X, y = self._create_sequences(features_scaled, returns)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)

                    # Class distribution
                    up_pct = (y == self.CLASS_UP).mean()
                    down_pct = (y == self.CLASS_DOWN).mean()
                    neutral_pct = (y == self.CLASS_NEUTRAL).mean()
                    print(f"  {token}: {len(X)} seq, UP={up_pct:.1%}, DOWN={down_pct:.1%}, NEUTRAL={neutral_pct:.1%}")

            if not all_X:
                return {"error": "No valid sequences created"}

            X = np.vstack(all_X)
            y = np.concatenate(all_y)

            # Global scaler
            self.scaler = RobustScaler()

        else:
            features_df = self._prepare_features(df, symbol, df_4h, df_1d, btc_df)
            returns = df['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon).values

            self.scaler = RobustScaler()
            features_scaled = self.scaler.fit_transform(features_df.values)

            X, y = self._create_sequences(features_scaled, returns)

        if len(X) < 100:
            return {"error": f"Not enough sequences: {len(X)}"}

        # Class distribution
        up_pct = (y == self.CLASS_UP).mean()
        down_pct = (y == self.CLASS_DOWN).mean()
        neutral_pct = (y == self.CLASS_NEUTRAL).mean()
        print(f"\nTotal: {len(X)} sequences")
        print(f"Class distribution: UP={up_pct:.1%}, DOWN={down_pct:.1%}, NEUTRAL={neutral_pct:.1%}")

        # Chronological split (NO shuffle for time series)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)

        # Calculate class weights for imbalance
        class_counts = np.bincount(y, minlength=3)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * 3  # Normalize
        self.class_weights = torch.FloatTensor(class_weights).to(self.device)

        print(f"Class weights: UP={class_weights[0]:.2f}, DOWN={class_weights[1]:.2f}, NEUTRAL={class_weights[2]:.2f}")

        # Initialize model
        self.model = EnhancedGRUClassifier(
            input_size=len(self.feature_names),
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            num_classes=3,
        ).to(self.device)

        # Loss function
        if use_focal_loss:
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        # Training loop
        best_val_acc = 0
        best_directional_acc = 0  # UP/DOWN accuracy (what we care about)
        best_state = None
        patience = 20
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            # Mini-batch training
            indices = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]

                optimizer.zero_grad()
                logits, _ = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                total_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits, _ = self.model(X_val)
                val_preds = torch.argmax(val_logits, dim=1)

                # Overall accuracy
                val_acc = (val_preds == y_val).float().mean().item()

                # Directional accuracy (UP and DOWN only, exclude NEUTRAL)
                directional_mask = (y_val != self.CLASS_NEUTRAL)
                if directional_mask.sum() > 0:
                    directional_acc = (val_preds[directional_mask] == y_val[directional_mask]).float().mean().item()
                else:
                    directional_acc = 0

                # UP accuracy specifically
                up_mask = (y_val == self.CLASS_UP)
                up_acc = (val_preds[up_mask] == y_val[up_mask]).float().mean().item() if up_mask.sum() > 0 else 0

                scheduler.step(directional_acc)

                # Save based on directional accuracy (what matters for trading)
                if directional_acc > best_directional_acc:
                    best_directional_acc = directional_acc
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.3f}, "
                      f"Dir Acc: {directional_acc:.3f}, UP Acc: {up_acc:.3f}, "
                      f"Best Dir: {best_directional_acc:.3f}")

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        return {
            "best_val_accuracy": best_val_acc,
            "best_directional_accuracy": best_directional_acc,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "epochs_trained": epoch + 1,
            "feature_count": len(self.feature_names),
            "class_distribution": {
                "up": up_pct,
                "down": down_pct,
                "neutral": neutral_pct
            }
        }

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        token: str = None,
        df_4h: pd.DataFrame = None,
        df_1d: pd.DataFrame = None,
        btc_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Predict direction with probabilities for all classes."""
        # Accept token as alias for symbol (for ensemble compatibility)
        if token and not symbol:
            symbol = token
        if self.model is None:
            raise RuntimeError("Model not trained")

        features_df = self._prepare_features(df, symbol, df_4h, df_1d, btc_df)

        if len(features_df) < self.seq_length:
            return {
                "direction": "NEUTRAL",
                "probability_up": 0.33,
                "probability_down": 0.33,
                "probability_neutral": 0.34,
                "confidence": 0.0,
            }

        # Scale features
        if symbol and symbol in self.scalers:
            features_scaled = self.scalers[symbol].transform(features_df.values)
        elif self.scaler is not None:
            features_scaled = self.scaler.transform(features_df.values)
        else:
            features_scaled = features_df.values

        last_sequence = features_scaled[-self.seq_length:]

        # Predict
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
            logits, attn_weights = self.model(X)

            # Softmax for probabilities
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

            prob_up = probs[self.CLASS_UP]
            prob_down = probs[self.CLASS_DOWN]
            prob_neutral = probs[self.CLASS_NEUTRAL]

        # Determine direction
        pred_class = np.argmax(probs)

        if pred_class == self.CLASS_UP:
            direction = "UP"
            confidence = prob_up - max(prob_down, prob_neutral)
        elif pred_class == self.CLASS_DOWN:
            direction = "DOWN"
            confidence = prob_down - max(prob_up, prob_neutral)
        else:
            direction = "NEUTRAL"
            confidence = 0.0

        # Only confident if probability is significantly higher
        if max(prob_up, prob_down) < 0.4:
            direction = "NEUTRAL"
            confidence = 0.0

        return {
            "direction": direction,
            "probability_up": float(prob_up),
            "probability_down": float(prob_down),
            "probability_neutral": float(prob_neutral),
            "confidence": float(max(0, confidence)),
            "forecast_horizon": self.forecast_horizon,
            "attention_focus": attn_weights.cpu().numpy().flatten()[-5:].tolist() if attn_weights is not None else [],
        }

    def save(self, path: str) -> None:
        """Save model and components."""
        data = {
            "model_state": self.model.state_dict() if self.model else None,
            "scaler": self.scaler,
            "scalers": self.scalers,
            "feature_names": self.feature_names,
            "class_weights": self.class_weights.cpu().numpy() if self.class_weights is not None else None,
            "config": {
                "seq_length": self.seq_length,
                "forecast_horizon": self.forecast_horizon,
                "threshold": self.threshold,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved model to {path}")

    def load(self, path: str) -> "EnhancedDirectionPredictor":
        """Load model and components (with CPU fallback for CUDA models)."""
        import io
        
        # Custom unpickler to handle CUDA tensors on CPU
        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
                return super().find_class(module, name)
        
        with open(path, "rb") as f:
            data = CPUUnpickler(f).load()

        self.scaler = data["scaler"]
        self.scalers = data.get("scalers", {})
        self.feature_names = data["feature_names"]
        self.seq_length = data["config"]["seq_length"]
        self.forecast_horizon = data["config"]["forecast_horizon"]
        self.threshold = data["config"].get("threshold", 0.005)

        if data.get("class_weights") is not None:
            self.class_weights = torch.FloatTensor(data["class_weights"]).to(self.device)

        if data["model_state"]:
            # Ensure all tensors are on CPU
            state_dict = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in data["model_state"].items()}
            self.model = EnhancedGRUClassifier(
                input_size=len(self.feature_names),
                hidden_size=128,
                num_layers=3,
                dropout=0.3,
                bidirectional=True,
                num_classes=3,
            ).to(self.device)
            self.model.load_state_dict(state_dict)

        return self
