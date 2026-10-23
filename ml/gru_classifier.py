"""
GRU Direction Classifier

Predicts price direction (UP/DOWN) for next N periods.
Uses GRU instead of LSTM based on research showing better
performance for crypto prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class GRUDirectionClassifier(nn.Module):
    """
    GRU-based binary classifier for price direction.

    Architecture:
    - Input: (batch, seq_len, features)
    - GRU layers with dropout
    - Fully connected output layer
    - Sigmoid for probability
    """

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        gru_out, _ = self.gru(x)
        # Take last time step
        last_hidden = gru_out[:, -1, :]
        # Classify
        return self.fc(last_hidden)


class DirectionPredictor:
    """
    Wrapper class for training and inference.
    """

    def __init__(
        self,
        seq_length: int = 24,  # 24 hours of hourly data
        forecast_horizon: int = 4,  # Predict 4 hours ahead
        threshold: float = 0.005,  # FIX: 0.5% minimum return to consider "UP" (was 0.0)
    ):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.threshold = threshold

        self.model: Optional[GRUDirectionClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.scalers: Dict[str, StandardScaler] = {}  # FIX: Per-token scalers
        self.feature_columns: list = []
        self.device = torch.device("cpu")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for GRU input."""
        # Reset index to ensure unique indexing
        df = df.reset_index(drop=True)
        
        features = pd.DataFrame(index=df.index)

        # Normalized price (relative to SMA)
        sma_20 = df['close'].rolling(20).mean()
        features['price_norm'] = (df['close'] - sma_20) / (sma_20 + 1e-10)

        # Returns
        features['return_1'] = df['close'].pct_change(1)
        features['return_4'] = df['close'].pct_change(4)

        # Volatility
        features['volatility'] = df['close'].pct_change().rolling(10).std()

        # RSI normalized (0-1)
        if 'rsi_14' in df.columns:
            features['rsi_norm'] = df['rsi_14'] / 100
        else:
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            features['rsi_norm'] = (100 - (100 / (1 + rs))) / 100

        # MACD normalized
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        features['macd_norm'] = macd / (df['close'] + 1e-10)

        # Bollinger Band position (0-1)
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)

        # Volume ratio
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / (vol_ma + 1e-10)
        else:
            features['volume_ratio'] = 1.0

        # High-low range
        if 'high' in df.columns and 'low' in df.columns:
            features['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        else:
            features['hl_range'] = 0.0

        # Trend (SMA slope)
        features['trend'] = sma_20.pct_change(5)

        self.feature_columns = features.columns.tolist()
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features

    def _create_sequences(
        self,
        features: np.ndarray,
        returns: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        X, y = [], []

        for i in range(len(features) - self.seq_length - self.forecast_horizon):
            X.append(features[i:i + self.seq_length])
            # Target: 1 if price goes up, 0 if down
            future_return = returns[i + self.seq_length + self.forecast_horizon - 1]
            if not np.isnan(future_return):
                y.append(1 if future_return > self.threshold else 0)
                # Only add X if we have valid y
            else:
                X.pop()  # Remove last X if no valid y

        return np.array(X), np.array(y)

    def _process_token_data(self, df: pd.DataFrame, token: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Process data for a single token and create sequences."""
        # Prepare features
        features_df = self._prepare_features(df)
        
        # Calculate future returns for labels
        returns = df['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon).values
        
        # Use per-token scaler if available, else global scaler
        if token and token in self.scalers:
            features_scaled = self.scalers[token].transform(features_df.values)
        else:
            features_scaled = self.scaler.transform(features_df.values)
        
        # Create sequences
        return self._create_sequences(features_scaled, returns)

    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the GRU classifier."""
        
        # Check if data has multiple tokens
        if 'token' in df.columns:
            tokens = df['token'].unique()
            print(f"Processing {len(tokens)} tokens separately...")
            
            # FIX: Fit per-token scalers for better normalization
            all_features = []
            for token in tokens:
                token_df = df[df['token'] == token].copy()
                if len(token_df) >= self.seq_length + self.forecast_horizon + 20:
                    features_df = self._prepare_features(token_df)
                    # Per-token scaler
                    self.scalers[token] = StandardScaler()
                    self.scalers[token].fit(features_df.values)
                    all_features.append(features_df.values)
            
            if not all_features:
                return {"error": "Not enough data for any token"}
            
            # Also keep a global scaler for unknown tokens
            self.scaler = StandardScaler()
            all_features_combined = np.vstack(all_features)
            self.scaler.fit(all_features_combined)
            
            # Second pass: create sequences for each token
            all_X, all_y = [], []
            for token in tokens:
                token_df = df[df['token'] == token].copy()
                if len(token_df) >= self.seq_length + self.forecast_horizon + 20:
                    X, y = self._process_token_data(token_df, token)
                    if len(X) > 0:
                        all_X.append(X)
                        all_y.append(y)
                        print(f"  {token}: {len(X)} sequences")
            
            if not all_X:
                return {"error": "No valid sequences created"}
            
            X = np.vstack(all_X)
            y = np.concatenate(all_y)
            
        else:
            # Single token data
            features_df = self._prepare_features(df)
            
            if len(features_df) < self.seq_length + self.forecast_horizon + 10:
                return {"error": "Not enough data for training"}
            
            # Calculate future returns for labels
            returns = df['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon).values

            # Scale features
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_df.values)

            # Create sequences
            X, y = self._create_sequences(features_scaled, returns)

        if len(X) < 100:
            return {"error": f"Not enough sequences: {len(X)}"}

        print(f"Total sequences: {len(X)}, Class distribution: {np.mean(y):.2%} UP")

        # FIX: REMOVED SHUFFLE - time series must NOT be shuffled (prevents data leakage)
        # Train/val split - chronological to prevent data leakage
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Initialize model
        self.model = GRUDirectionClassifier(
            input_size=len(self.feature_columns),
        ).to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        best_val_acc = 0
        best_state = None
        patience = 15
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
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_preds = (val_outputs > 0.5).float()
                val_acc = (val_preds == y_val).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.4f}, Best: {best_val_acc:.4f}")

            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        return {
            "best_val_accuracy": best_val_acc,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "epochs_trained": epoch + 1,
        }

    def predict(self, df: pd.DataFrame, token: str = None) -> Dict[str, Any]:
        """Predict direction for current market state."""
        if self.model is None:
            raise RuntimeError("Model not trained")

        # Prepare features
        features_df = self._prepare_features(df)

        if len(features_df) < self.seq_length:
            return {
                "direction": "NEUTRAL",
                "probability_up": 0.5,
                "probability_down": 0.5,
                "confidence": 0.0,
            }

        # Get last sequence - use per-token scaler if available
        if token and token in self.scalers:
            features_scaled = self.scalers[token].transform(features_df.values)
        else:
            features_scaled = self.scaler.transform(features_df.values)
        last_sequence = features_scaled[-self.seq_length:]

        # Predict
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
            prob_up = self.model(X).item()

        # FIX: Narrow neutral band from 0.4-0.6 to 0.45-0.55
        # FIX: Uppercase direction for consistency
        if prob_up > 0.55:  # was 0.6
            direction = "UP"  # was "up"
            confidence = (prob_up - 0.5) * 2
        elif prob_up < 0.45:  # was 0.4
            direction = "DOWN"  # was "down"
            confidence = (0.5 - prob_up) * 2
        else:
            direction = "NEUTRAL"  # was "neutral"
            confidence = 0.0

        return {
            "direction": direction,
            "probability_up": prob_up,
            "probability_down": 1 - prob_up,
            "confidence": confidence,
            "forecast_horizon": self.forecast_horizon,
        }

    def save(self, path: str) -> None:
        """Save model and scaler."""
        data = {
            "model_state": self.model.state_dict() if self.model else None,
            "scaler": self.scaler,
            "scalers": self.scalers,  # FIX: Save per-token scalers
            "feature_columns": self.feature_columns,
            "config": {
                "seq_length": self.seq_length,
                "forecast_horizon": self.forecast_horizon,
                "threshold": self.threshold,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> "DirectionPredictor":
        """Load model and scaler (with CPU fallback for CUDA models)."""
        import io
        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
                return super().find_class(module, name)
        with open(path, "rb") as f:
            data = CPUUnpickler(f).load()
        self.scaler = data["scaler"]
        self.scalers = data.get("scalers", {})
        self.feature_columns = data["feature_columns"]
        self.seq_length = data["config"]["seq_length"]
        self.forecast_horizon = data["config"]["forecast_horizon"]
        self.threshold = data["config"]["threshold"]
        if data["model_state"]:
            state_dict = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in data["model_state"].items()}
            self.model = GRUDirectionClassifier(
                input_size=len(self.feature_columns),
            ).to(self.device)
            self.model.load_state_dict(state_dict)
        return self