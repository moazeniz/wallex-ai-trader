#!/usr/bin/env python3
"""Test GRU integration in the orchestrator."""

import sys
sys.path.insert(0, "/opt/wallex")

from agents.orchestrator import TradingOrchestrator

print("Testing GRU-integrated orchestrator...")
orch = TradingOrchestrator(exchange="binance", chains=["ethereum"])

print("\nRunning analysis on AAVE/USDT...")
result = orch.analyze(
    symbol="AAVE/USDT",
    token_address="0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
    chain="ethereum",
    timeframe="1h"
)

print("\n" + "=" * 50)
print("Results")
print("=" * 50)
print(f"Symbol: {result.get('symbol')}")
print(f"Latest Price: ${result.get('latest_price', 0):,.2f}")

# GRU prediction
gru = result.get("gru_prediction", {})
print(f"\nGRU Prediction:")
print(f"  Direction: {gru.get('direction', 'N/A')}")
print(f"  Prob UP: {gru.get('probability_up', 0):.1%}")
print(f"  Confidence: {gru.get('confidence', 0):.1%}")

# Signal
signal = result.get("aggregated_signal")
if signal:
    print(f"\nFinal Signal: {signal.signal_type.value.upper()} ({signal.strength.value})")
    print(f"Confidence: {signal.confidence:.0%}")
    print(f"Reasoning: {signal.reasoning}")

print(f"\nRisk Approved: {result.get('risk_approved')}")

# Timing
timing = result.get("timing", {})
print(f"\nTiming:")
for step, seconds in timing.items():
    print(f"  {step}: {seconds:.2f}s")
print(f"  Total: {sum(timing.values()):.2f}s")
