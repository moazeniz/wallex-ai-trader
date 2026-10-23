"""
LSTM Direction Classifier

Similar to GRU but with different memory cell architecture.
May capture different patterns than GRU.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class LSTMDirectionClassifier(nn.Module):
    """LSTM-based binary classifier for price direction."""
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
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
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class LSTMPredictor:
    """Wrapper for LSTM training and inference."""
    
    def __init__(
        self,
        seq_length: int = 24,
        forecast_horizon: int = 4,
        threshold: float = 0.005,
    ):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.threshold = threshold
        
        self.model: Optional[LSTMDirectionClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: list = []
        self.device = torch.device("cpu")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for LSTM input."""
        df = df.reset_index(drop=True)
        features = pd.DataFrame(index=df.index)
        
        sma_20 = df['close'].rolling(20).mean()
        features['price_norm'] = (df['close'] - sma_20) / (sma_20 + 1e-10)
        features['return_1'] = df['close'].pct_change(1)
        features['return_4'] = df['close'].pct_change(4)
        features['volatility'] = df['close'].pct_change().rolling(10).std()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_norm'] = (100 - (100 / (1 + rs))) / 100
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd_norm'] = (ema_12 - ema_26) / (df['close'] + 1e-10)
        
        # Bollinger
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
        
        # Volume
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / (vol_ma + 1e-10)
        else:
            features['volume_ratio'] = 1.0
        
        # Range
        if 'high' in df.columns and 'low' in df.columns:
            features['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        else:
            features['hl_range'] = 0.0
        
        features['trend'] = sma_20.pct_change(5)
        
        self.feature_columns = features.columns.tolist()
        return features.fillna(0)
    
    def _create_sequences(self, features: np.ndarray, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        X, y = [], []
        for i in range(len(features) - self.seq_length - self.forecast_horizon):
            X.append(features[i:i + self.seq_length])
            future_return = returns[i + self.seq_length + self.forecast_horizon - 1]
            if not np.isnan(future_return):
                y.append(1 if future_return > self.threshold else 0)
            else:
                X.pop()
        return np.array(X), np.array(y)
    
    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the LSTM classifier."""
        
        if 'token' in df.columns:
            tokens = df['token'].unique()
            all_X, all_y = [], []
            
            for token in tokens:
                token_df = df[df['token'] == token].copy()
                if len(token_df) < self.seq_length + self.forecast_horizon + 20:
                    continue
                
                features_df = self._prepare_features(token_df)
                returns = token_df['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon).values
                
                self.scalers[token] = StandardScaler()
                features_scaled = self.scalers[token].fit_transform(features_df.values)
                
                X, y = self._create_sequences(features_scaled, returns)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    print(f"  {token}: {len(X)} sequences")
            
            X = np.vstack(all_X)
            y = np.concatenate(all_y)
            
            # Global scaler for unknown tokens
            self.scaler = StandardScaler()
            all_features = np.vstack([self.scalers[t].mean_.reshape(1, -1) for t in self.scalers])
            self.scaler.fit(all_features)
        else:
            features_df = self._prepare_features(df)
            returns = df['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon).values
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_df.values)
            X, y = self._create_sequences(features_scaled, returns)
        
        print(f"Total sequences: {len(X)}, Class distribution: {np.mean(y):.2%} UP")
        
        # Chronological split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Model
        self.model = LSTMDirectionClassifier(
            input_size=len(self.feature_columns),
        ).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_acc = 0
        best_state = None
        patience = 15
        no_improve = 0
        
        for epoch in range(epochs):
            self.model.train()
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
            
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        return {
            "best_val_accuracy": best_val_acc,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "epochs_trained": epoch + 1,
        }
    
    def predict(self, df: pd.DataFrame, token: str = None) -> Dict[str, Any]:
        """Predict direction."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        features_df = self._prepare_features(df)
        
        if len(features_df) < self.seq_length:
            return {"direction": "NEUTRAL", "probability_up": 0.5, "confidence": 0.0}
        
        if token and token in self.scalers:
            features_scaled = self.scalers[token].transform(features_df.values)
        else:
            features_scaled = self.scaler.transform(features_df.values)
        
        last_sequence = features_scaled[-self.seq_length:]
        
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
            prob_up = self.model(X).item()
        
        if prob_up > 0.55:
            direction = "UP"
            confidence = (prob_up - 0.5) * 2
        elif prob_up < 0.45:
            direction = "DOWN"
            confidence = (0.5 - prob_up) * 2
        else:
            direction = "NEUTRAL"
            confidence = 0.0
        
        return {
            "direction": direction,
            "probability_up": prob_up,
            "probability_down": 1 - prob_up,
            "confidence": confidence,
        }
    
    def save(self, path: str) -> None:
        """Save model."""
        data = {
            "model_state": self.model.state_dict() if self.model else None,
            "scaler": self.scaler,
            "scalers": self.scalers,
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
    
    def load(self, path: str) -> "LSTMPredictor":
        """Load model (with CPU fallback for CUDA models)."""
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
        self.feature_columns = data["feature_columns"]
        self.seq_length = data["config"]["seq_length"]
        self.forecast_horizon = data["config"]["forecast_horizon"]
        self.threshold = data["config"]["threshold"]
        
        if data["model_state"]:
            # Ensure all tensors are on CPU
            state_dict = {k: v.cpu() if hasattr(v, "cpu") else v for k, v in data["model_state"].items()}
            self.model = LSTMDirectionClassifier(
                input_size=len(self.feature_columns),
            ).to(self.device)
            self.model.load_state_dict(state_dict)
        
        return self
