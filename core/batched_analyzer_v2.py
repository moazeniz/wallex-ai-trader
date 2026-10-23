"""
Batched Analyzer V2 - Enhanced Version

Uses:
- Enhanced GRU with attention and multi-class output
- Enhanced ensemble with dynamic weighting and confidence gating
- Multi-timeframe features
- Focus on bullish predictions

Flow:
1. Gather multi-timeframe data for all tokens
2. Run enhanced ensemble predictions (local, no API)
3. Use LLM for interpretation only (1 API call)
4. Return ensemble predictions with interpretations
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import time

try:
    from google import genai
    USE_NEW_API = True
except ImportError:
    import google.generativeai as genai
    USE_NEW_API = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.exchange_client import ExchangeClient
from tools.technical_tools import TechnicalTools
from ml.regime_detector import RegimeDetector
from settings import load_config, TRACKED_TOKENS

# Try to import enhanced components, fallback to original
try:
    from ml.gru_enhanced import EnhancedDirectionPredictor
    from ml.ensemble_enhanced import EnhancedEnsembleCombiner, get_enhanced_ensemble
    from ml.multiframe_fetcher import get_multiframe_fetcher
    USE_ENHANCED = True
    print("[BatchedV2] Using enhanced ML components")
except ImportError as e:
    print(f"[BatchedV2] Enhanced components not available: {e}")
    from ml.gru_classifier import DirectionPredictor
    from ml.ensemble import EnsembleCombiner
    USE_ENHANCED = False

# Original models as fallback
try:
    from ml.gru_classifier import DirectionPredictor as GRUPredictor
    from ml.lstm_classifier import LSTMPredictor
    from ml.xgboost_classifier import XGBoostDirectionClassifier
except ImportError:
    pass


@dataclass
class TokenSignal:
    """Signal for a single token."""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    # Ensemble details
    ensemble_prediction: str
    model_agreement: float
    probability_up: float = 0.5
    probability_down: float = 0.5
    # Model predictions
    gru_direction: str = ""
    lstm_direction: str = ""
    xgboost_direction: str = ""
    # LLM interpretation
    llm_interpretation: str = ""
    llm_risks: str = ""
    reasoning: str = ""


class BatchedAnalyzerV2:
    """
    Enhanced batched analyzer with:
    - Multi-timeframe data
    - Enhanced ensemble
    - Focus on bullish predictions
    """

    def __init__(
        self,
        exchange: str = "binance",
        model_name: str = "gemini-2.0-flash",
        max_output_tokens: int = 8192,
    ):
        self.exchange = exchange
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens

        # Initialize components
        self.exchange_client = ExchangeClient(exchange=exchange)
        self.technical = TechnicalTools(exchange=exchange)

        # Multi-timeframe fetcher
        if USE_ENHANCED:
            self.mtf_fetcher = get_multiframe_fetcher()
        else:
            self.mtf_fetcher = None

        # Enhanced or original ensemble
        self._init_ensemble()

        # Regime detector
        try:
            self.regime = RegimeDetector()
        except:
            self.regime = None

        # Configure Gemini
        self.config = load_config()
        api_key = self.config.llm.get_api_key()
        print(f"[BatchedV2] API key pool: {self.config.llm.key_count} keys available")

        if USE_NEW_API:
            self.client = genai.Client(api_key=api_key)
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "max_output_tokens": max_output_tokens,
                    "temperature": 0.3,
                }
            )

        self._max_output_tokens = max_output_tokens
        self.tokens = [v['symbol'] for v in TRACKED_TOKENS.values()]

    def _init_ensemble(self):
        """Initialize the ensemble with all models."""
        if USE_ENHANCED:
            self.ensemble = get_enhanced_ensemble()
        else:
            self.ensemble = EnsembleCombiner()

        models_loaded = []

        # Load enhanced GRU if available
        if USE_ENHANCED:
            try:
                gru = EnhancedDirectionPredictor()
                gru.load("/opt/wallex/models/gru_enhanced.pkl")
                self.ensemble.register_model("gru", gru, weight=0.50)
                models_loaded.append("GRU-Enhanced")
            except Exception as e:
                print(f"Enhanced GRU not available: {e}")

        # Load original models as fallback/ensemble members
        try:
            gru_orig = GRUPredictor()
            gru_orig.load("/opt/wallex/models/gru_direction.pkl")
            if "GRU-Enhanced" not in models_loaded:
                self.ensemble.register_model("gru", gru_orig, weight=0.50)
                models_loaded.append("GRU")
        except Exception as e:
            print(f"Original GRU not available: {e}")

        try:
            lstm = LSTMPredictor()
            lstm.load("/opt/wallex/models/lstm_direction.pkl")
            self.ensemble.register_model("lstm", lstm, weight=0.25)
            models_loaded.append("LSTM")
        except Exception as e:
            print(f"LSTM not available: {e}")

        try:
            xgb = XGBoostDirectionClassifier()
            xgb.load("/opt/wallex/models/xgboost_direction.pkl")
            self.ensemble.register_model("xgboost", xgb, weight=0.25)
            models_loaded.append("XGBoost")
        except Exception as e:
            print(f"XGBoost not available: {e}")

        print(f"[BatchedV2] Ensemble models: {models_loaded}")

    def gather_all_token_data(self) -> Dict[str, Dict]:
        """Gather multi-timeframe data and predictions for all tokens."""
        all_data = {}

        # Get BTC data for cross-asset features
        btc_df = self.exchange_client.fetch_ohlcv("BTC/USDT", "1h", limit=200, use_cache=True)

        for symbol in self.tokens:
            try:
                # Get multi-timeframe data if available
                if self.mtf_fetcher:
                    mtf_data = self.mtf_fetcher.fetch_multi_timeframe(symbol, include_btc=False)
                    df = mtf_data.get('1h')
                    df_4h = mtf_data.get('4h')
                    df_1d = mtf_data.get('1d')
                else:
                    df = self.exchange_client.fetch_ohlcv(symbol, "1h", limit=200, use_cache=True)
                    df_4h = None
                    df_1d = None

                if df is None or len(df) < 50:
                    all_data[symbol] = {"error": "Insufficient data"}
                    continue

                price = float(df["close"].iloc[-1])
                change_24h = (df["close"].iloc[-1] / df["close"].iloc[-24] - 1) * 100 if len(df) >= 24 else 0

                # Technical indicators
                indicators = self.technical.get_indicators(symbol, "1h")

                rsi = 50
                macd_hist = 0
                bb_position = 0.5

                if indicators and "indicators" in indicators:
                    ind = indicators["indicators"]
                    rsi = ind.get("rsi", {}).get("value", 50)
                    macd = ind.get("macd", {})
                    macd_hist = macd.get("histogram", 0)
                    bb = ind.get("bollinger_bands", {})
                    if bb.get("upper") and bb.get("lower"):
                        bb_range = bb["upper"] - bb["lower"]
                        if bb_range > 0:
                            bb_position = (price - bb["lower"]) / bb_range

                # Regime detection
                regime = "unknown"
                if self.regime:
                    try:
                        regime_info = self.regime.get_regime_summary(df)
                        regime = regime_info.get("market_regime", "unknown")
                    except:
                        pass

                # ENSEMBLE PREDICTION
                if USE_ENHANCED:
                    ensemble_result = self.ensemble.predict(
                        df=df,
                        token=symbol,
                        df_4h=df_4h,
                        df_1d=df_1d,
                        btc_df=btc_df,
                        regime=regime
                    )
                else:
                    ensemble_result = self.ensemble.predict(df, token=symbol)

                model_preds = ensemble_result.get("model_predictions", {})

                all_data[symbol] = {
                    "price": price,
                    "change_24h": change_24h,
                    "rsi": rsi,
                    "macd_histogram": macd_hist,
                    "bb_position": bb_position,
                    "regime": regime,
                    "df": df,
                    "df_4h": df_4h,
                    "df_1d": df_1d,
                    # Ensemble prediction
                    "ensemble": ensemble_result,
                    "ensemble_direction": ensemble_result.get("direction", "NEUTRAL"),
                    "ensemble_prob_up": ensemble_result.get("probability_up", 0.5),
                    "ensemble_prob_down": ensemble_result.get("probability_down", 0.5),
                    "ensemble_confidence": ensemble_result.get("confidence", 0),
                    "model_agreement": ensemble_result.get("agreement_ratio", 0),
                    "model_predictions": model_preds,
                    "models_included": ensemble_result.get("models_included", []),
                }

            except Exception as e:
                all_data[symbol] = {"error": str(e)}

        return all_data

    def build_interpretation_prompt(self, all_data: Dict[str, Dict]) -> str:
        """Build prompt for LLM interpretation (not decision-making)."""
        prompt = """You are interpreting ML ensemble predictions. You DO NOT make trading decisions.

## YOUR ROLE
The ML ensemble has made predictions. Your job is to:
1. EXPLAIN why the models might be predicting this
2. IDENTIFY 2-3 risks or counter-arguments
3. FLAG any unusual conditions

## CRITICAL RULES
- Do NOT change model predictions
- Focus on BULLISH signals (UP predictions are more reliable)
- Be specific about risks

## TOKEN DATA:
"""

        for symbol, data in all_data.items():
            if "error" in data:
                continue

            model_preds = data.get("model_predictions", {})
            models_included = data.get("models_included", [])

            prompt += f"""
{symbol}:
  Price: ${data.get('price', 0):,.2f} ({data.get('change_24h', 0):+.1f}% 24h)
  RSI: {data.get('rsi', 50):.1f} | Regime: {data.get('regime', 'unknown')}

  ENSEMBLE: {data.get('ensemble_direction', 'NEUTRAL')}
  P(UP): {data.get('ensemble_prob_up', 0.5):.1%} | P(DOWN): {data.get('ensemble_prob_down', 0.5):.1%}
  Agreement: {data.get('model_agreement', 0):.0%} | Models: {', '.join(models_included)}
"""
            for name, pred in model_preds.items():
                if isinstance(pred, dict):
                    prompt += f"    {name}: {pred.get('direction', 'N/A')} ({pred.get('probability_up', 0.5):.1%})\n"

        prompt += """

## OUTPUT FORMAT
For each token with a BUY or SELL signal, provide:

SYMBOL: [symbol]
INTERPRETATION: [1-2 sentences why models predict this]
RISKS: [2-3 specific risks]

For HOLD signals, you can skip or provide brief note.
"""
        return prompt

    def parse_interpretation(self, response_text: str) -> Dict[str, Dict]:
        """Parse LLM interpretation."""
        interpretations = {}
        current_symbol = None
        current_interp = {}

        for line in response_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("SYMBOL:"):
                if current_symbol:
                    interpretations[current_symbol] = current_interp
                current_symbol = line.replace("SYMBOL:", "").strip()
                current_interp = {}
            elif line.startswith("INTERPRETATION:"):
                current_interp["interpretation"] = line.replace("INTERPRETATION:", "").strip()
            elif line.startswith("RISKS:"):
                current_interp["risks"] = line.replace("RISKS:", "").strip()

        if current_symbol:
            interpretations[current_symbol] = current_interp

        return interpretations

    def _rotate_api_key(self):
        """Rotate to next API key."""
        new_key = self.config.llm.rotate_key()
        key_idx = self.config.llm._current_key_index
        print(f"[BatchedV2] Rotating to API key #{key_idx + 1}/{self.config.llm.key_count}")

        if USE_NEW_API:
            self.client = genai.Client(api_key=new_key)
        else:
            genai.configure(api_key=new_key)

    def analyze_all(self) -> List[TokenSignal]:
        """Analyze all tokens with enhanced ensemble."""
        print(f"[BatchedV2] Gathering data for {len(self.tokens)} tokens...")

        # Gather data and predictions
        all_data = self.gather_all_token_data()
        valid_tokens = [s for s, d in all_data.items() if "error" not in d]
        print(f"[BatchedV2] Got predictions for {len(valid_tokens)} tokens")

        # Get LLM interpretation
        prompt = self.build_interpretation_prompt(all_data)
        response_text = ""

        max_retries = self.config.llm.key_count
        for attempt in range(max_retries):
            try:
                if USE_NEW_API:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "max_output_tokens": self._max_output_tokens,
                            "temperature": 0.3,
                        }
                    )
                    response_text = response.text
                else:
                    response = self.model.generate_content(prompt)
                    response_text = response.text
                break

            except Exception as e:
                if '429' in str(e).lower() or 'quota' in str(e).lower():
                    if attempt < max_retries - 1:
                        self._rotate_api_key()
                        time.sleep(1)
                        continue
                print(f"[BatchedV2] LLM error: {e}")
                break

        interpretations = self.parse_interpretation(response_text) if response_text else {}

        # Build signals
        signals = []
        for symbol, data in all_data.items():
            if "error" in data:
                continue

            price = data.get("price", 0)
            ensemble_dir = data.get("ensemble_direction", "NEUTRAL")
            ensemble_prob_up = data.get("ensemble_prob_up", 0.5)
            ensemble_prob_down = data.get("ensemble_prob_down", 0.5)
            ensemble_conf = data.get("ensemble_confidence", 0)
            model_agreement = data.get("model_agreement", 0)
            model_preds = data.get("model_predictions", {})

            # Signal from ensemble
            if ensemble_dir == "UP" and ensemble_conf > 0.1:
                signal = "BUY"
                confidence = 50 + ensemble_conf * 50
            elif ensemble_dir == "DOWN" and ensemble_conf > 0.15:  # Higher threshold for DOWN
                signal = "SELL"
                confidence = 50 + ensemble_conf * 50
            else:
                signal = "HOLD"
                confidence = 50

            # Boost confidence for high agreement
            if model_agreement >= 0.8:
                confidence = min(confidence * 1.1, 95)
            elif model_agreement < 0.5:
                confidence *= 0.85

            # Stop loss and take profit
            sl_pct = 5
            tp_pct = 10

            interp = interpretations.get(symbol, {})

            # Extract model directions
            gru_dir = ""
            lstm_dir = ""
            xgb_dir = ""
            for name, pred in model_preds.items():
                if isinstance(pred, dict):
                    if 'gru' in name.lower():
                        gru_dir = pred.get("direction", "")
                    elif 'lstm' in name.lower():
                        lstm_dir = pred.get("direction", "")
                    elif 'xgb' in name.lower() or 'xgboost' in name.lower():
                        xgb_dir = pred.get("direction", "")

            signals.append(TokenSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                entry_price=price,
                stop_loss=price * (1 - sl_pct/100) if signal == "BUY" else price * (1 + sl_pct/100),
                take_profit=price * (1 + tp_pct/100) if signal == "BUY" else price * (1 - tp_pct/100),
                ensemble_prediction=ensemble_dir,
                model_agreement=model_agreement,
                probability_up=ensemble_prob_up,
                probability_down=ensemble_prob_down,
                gru_direction=gru_dir,
                lstm_direction=lstm_dir,
                xgboost_direction=xgb_dir,
                llm_interpretation=interp.get("interpretation", ""),
                llm_risks=interp.get("risks", ""),
                reasoning=interp.get("interpretation", ""),
            ))

        # Summary
        actionable = [s for s in signals if s.signal != "HOLD" and s.confidence >= 60]
        buy_signals = [s for s in actionable if s.signal == "BUY"]
        print(f"[BatchedV2] Generated {len(signals)} signals, {len(actionable)} actionable, {len(buy_signals)} BUY")

        return signals


# Singleton
_analyzer_v2 = None

def get_batched_analyzer_v2() -> BatchedAnalyzerV2:
    global _analyzer_v2
    if _analyzer_v2 is None:
        _analyzer_v2 = BatchedAnalyzerV2()
    return _analyzer_v2


if __name__ == "__main__":
    print("Testing Batched Analyzer V2...")
    analyzer = BatchedAnalyzerV2()
    signals = analyzer.analyze_all()

    print("\n" + "=" * 60)
    for sig in signals:
        if sig.signal != "HOLD":
            print(f"{sig.symbol}: {sig.signal} ({sig.confidence:.0f}%)")
            print(f"  Ensemble: {sig.ensemble_prediction} | Agreement: {sig.model_agreement:.0%}")
            print(f"  P(UP): {sig.probability_up:.1%} | P(DOWN): {sig.probability_down:.1%}")
            if sig.llm_interpretation:
                print(f"  {sig.llm_interpretation[:100]}...")
            print()
