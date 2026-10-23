#!/usr/bin/env python3
"""Train GRU direction classifier on historical data."""

import sys
sys.path.insert(0, "/opt/wallex")

from data.exchange_client import ExchangeClient
from ml.gru_classifier import DirectionPredictor
from settings import TRACKED_TOKENS
import pandas as pd
import time


def main():
    print("=" * 50)
    print("GRU Direction Classifier Training")
    print("=" * 50)
    
    client = ExchangeClient()
    
    # Collect training data from tracked tokens
    all_data = []
    for token, config in TRACKED_TOKENS.items():
        symbol = config["symbol"]
        print(f"\nFetching {token} ({symbol})...")
        
        try:
            # Fetch extended OHLCV data (30 days for training)
            df = client.fetch_ohlcv_extended(
                symbol=symbol,
                timeframe="1h",
                days=60,  # 60 days of hourly data
                use_cache=False
            )
            
            if df is not None and len(df) > 100:
                df["token"] = token
                all_data.append(df)
                print(f"  Got {len(df)} candles for {token}")
            else:
                print(f"  Insufficient data for {token}")
                
            # Small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Error fetching {token}: {e}")
            continue
    
    if not all_data:
        print("\nERROR: No data collected. Cannot train.")
        return
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=False)
    combined = combined.sort_index()
    print(f"\nTotal training samples: {len(combined)}")
    
    # Train predictor
    print("\nInitializing GRU Direction Predictor...")
    predictor = DirectionPredictor(
        seq_length=24,      # 24 hours lookback
        forecast_horizon=4,  # Predict 4 hours ahead
        threshold=0.005,     # 0.5% minimum move for "UP"
    )
    
    print("\nTraining...")
    results = predictor.fit(
        combined,
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        validation_split=0.2,
    )
    
    print("\n" + "=" * 50)
    print("Training Results:")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    # Save model
    model_path = "/opt/wallex/models/gru_direction.pkl"
    predictor.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Quick test prediction
    print("\nTesting prediction on latest data...")
    test_df = client.fetch_ohlcv("BTC/USDT", "1h", limit=50, use_cache=False)
    
    # Reload model to test save/load
    test_predictor = DirectionPredictor()
    test_predictor.load(model_path)
    
    prediction = test_predictor.predict(test_df)
    print(f"Test prediction: {prediction}")
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
