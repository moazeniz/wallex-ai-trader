#!/usr/bin/env python3
"""
Local GPU Training Script
Trains models using RTX 4070 Ti
"""

import sys
import os

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
import json
import torch

# Check CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Import components
from ml.enhanced_features import get_feature_extractor
from ml.gru_enhanced import EnhancedDirectionPredictor

# Paths
DATA_DIR = Path(BASE_DIR) / "data" / "historical"
MODEL_DIR = Path(BASE_DIR) / "models"
BACKUP_DIR = Path(BASE_DIR) / "models" / "backups"
LOG_FILE = Path(BASE_DIR) / "logs" / "local_train.log"

# Tokens to train on
TOKENS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LUNC/USDT", "PAXG/USDT"]


def log(message: str):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(log_msg + "\n")
    except:
        pass


def backup_models():
    """Backup current models before retraining."""
    log("Backing up current models...")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_file in MODEL_DIR.glob("*.pkl"):
        backup_path = BACKUP_DIR / f"{model_file.stem}_{timestamp}.pkl"
        shutil.copy(model_file, backup_path)
        log(f"  Backed up: {model_file.name}")


def load_training_data() -> pd.DataFrame:
    """Load all historical data into single DataFrame."""
    log("Loading training data...")

    all_data = []

    for symbol in TOKENS:
        token_name = symbol.replace("/", "_")
        filepath = DATA_DIR / f"{token_name}_1h_365d.csv"

        if not filepath.exists():
            log(f"  Warning: {filepath} not found")
            continue

        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["token"] = symbol

        log(f"  {symbol}: {len(df)} candles")
        all_data.append(df)

    if not all_data:
        raise ValueError("No training data found!")

    combined = pd.concat(all_data, ignore_index=True)
    log(f"Total: {len(combined)} candles")

    return combined


def train_enhanced_gru(df: pd.DataFrame) -> dict:
    """Train the enhanced GRU model."""
    log("Training Enhanced GRU Model...")
    log(f"  Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    model = EnhancedDirectionPredictor(
        seq_length=48,
        forecast_horizon=4,
        threshold=0.005
    )

    start_time = datetime.now()

    result = model.fit(
        df,
        epochs=100,
        batch_size=64,
        learning_rate=0.0005,
        validation_split=0.2,
        use_focal_loss=True
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"  Training time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    log(f"  Val Accuracy: {result.get('best_val_accuracy', 0):.2%}")
    log(f"  Dir Accuracy: {result.get('best_directional_accuracy', 0):.2%}")
    log(f"  Features: {result.get('feature_count', 0)}")

    # Save model
    model_path = MODEL_DIR / "gru_direction.pkl"
    model.save(str(model_path))
    log(f"  Saved to: {model_path}")

    result['training_time'] = elapsed
    return result


def main():
    """Main training pipeline."""
    log("=" * 60)
    log("LOCAL GPU TRAINING PIPELINE")
    log("=" * 60)

    results = {}

    try:
        # 1. Backup existing models
        backup_models()

        # 2. Load training data (already downloaded)
        df = load_training_data()

        # 3. Train enhanced GRU
        results['gru'] = train_enhanced_gru(df)

        log("=" * 60)
        log("TRAINING COMPLETE!")
        log(f"Time: {results['gru'].get('training_time', 0)/60:.1f} minutes")
        log("=" * 60)

    except Exception as e:
        log(f"TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)

    # Save results
    results_path = MODEL_DIR / "last_retrain_results.json"
    with open(results_path, "w") as f:
        def convert(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.float64, np.float32)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        json.dump(results, f, indent=2, default=convert)

    return results


if __name__ == "__main__":
    main()
