"""
Batched Analyzer - Interpretation Focused

PHILOSOPHY:
- ML ensemble (GRU + LSTM + XGBoost) makes predictions
- LLM provides interpretation, context, and risk identification
- LLM NEVER overrides model predictions
- Reduces API calls to ~100/day

Flow:
1. Gather data for all tokens
2. Run ensemble predictions (local, no API)
3. Use LLM to interpret and add context (1 API call for all tokens)
4. Return ensemble predictions with LLM interpretations
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
from ml.gru_classifier import DirectionPredictor as GRUPredictor
from ml.lstm_classifier import LSTMPredictor
from ml.xgboost_classifier import XGBoostDirectionClassifier
from ml.ensemble import EnsembleCombiner
from ml.regime_detector import RegimeDetector
from settings import load_config, TRACKED_TOKENS


@dataclass
class TokenSignal:
    """Signal for a single token - prediction from ensemble, context from LLM."""
    symbol: str
    signal: str  # BUY, SELL, HOLD (from ensemble)
    confidence: float  # From ensemble
    entry_price: float
    stop_loss: float
    take_profit: float
    # Ensemble details
    ensemble_prediction: str
    model_agreement: float
    gru_direction: str = ""
    lstm_direction: str = ""
    xgboost_direction: str = ""
    # LLM interpretation (context, not decision)
    llm_interpretation: str = ""
    llm_risks: str = ""
    llm_counter_arguments: str = ""


class BatchedAnalyzer:
    """
    Batched analyzer that:
    1. Uses ML ensemble for predictions
    2. Uses LLM for interpretation only
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

        # ML Ensemble (makes predictions)
        self.ensemble = EnsembleCombiner()
        self._load_ensemble_models()

        # Regime detector
        try:
            self.regime = RegimeDetector()
        except:
            self.regime = None

        # Configure Gemini for interpretation
        self.config = load_config()
        api_key = self.config.llm.get_api_key()
        print(f"[Batched] API key pool: {self.config.llm.key_count} keys available")

        if USE_NEW_API:
            self.client = genai.Client(api_key=api_key)
            self.model_name = model_name
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

    def _load_ensemble_models(self):
        """Load all ensemble models."""
        models_loaded = []
        
        try:
            gru = GRUPredictor()
            gru.load("/opt/wallex/models/gru_direction.pkl")
            self.ensemble.register_model("gru", gru, weight=0.35)
            models_loaded.append("GRU")
        except Exception as e:
            print(f"Warning: Could not load GRU: {e}")
        
        try:
            lstm = LSTMPredictor()
            lstm.load("/opt/wallex/models/lstm_direction.pkl")
            self.ensemble.register_model("lstm", lstm, weight=0.30)
            models_loaded.append("LSTM")
        except Exception as e:
            print(f"Warning: Could not load LSTM: {e}")
        
        try:
            xgb = XGBoostDirectionClassifier()
            xgb.load("/opt/wallex/models/xgboost_direction.pkl")
            self.ensemble.register_model("xgboost", xgb, weight=0.35)
            models_loaded.append("XGBoost")
        except Exception as e:
            print(f"Warning: Could not load XGBoost: {e}")
        
        print(f"[Batched] Ensemble models loaded: {models_loaded}")

    def gather_all_token_data(self) -> Dict[str, Dict]:
        """Gather OHLCV and indicator data for all tokens."""
        all_data = {}

        for symbol in self.tokens:
            try:
                df = self.exchange_client.fetch_ohlcv(
                    symbol, timeframe="1h", limit=200, use_cache=True
                )
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

                # ENSEMBLE PREDICTION (the actual prediction)
                ensemble_result = self.ensemble.predict(df, token=symbol)
                
                # Regime detection
                regime = "unknown"
                if self.regime:
                    try:
                        regime_info = self.regime.get_regime_summary(df)
                        regime = regime_info.get("market_regime", "unknown")
                    except:
                        pass

                all_data[symbol] = {
                    "price": price,
                    "change_24h": change_24h,
                    "rsi": rsi,
                    "macd_histogram": macd_hist,
                    "bb_position": bb_position,
                    "regime": regime,
                    "df": df,
                    # Ensemble prediction
                    "ensemble": ensemble_result,
                    "ensemble_direction": ensemble_result.get("direction", "NEUTRAL"),
                    "ensemble_prob_up": ensemble_result.get("probability_up", 0.5),
                    "ensemble_confidence": ensemble_result.get("confidence", 0),
                    "model_agreement": ensemble_result.get("agreement_ratio", 0),
                    "model_predictions": ensemble_result.get("model_predictions", {}),
                }

            except Exception as e:
                all_data[symbol] = {"error": str(e)}

        return all_data

    def build_interpretation_prompt(self, all_data: Dict[str, Dict]) -> str:
        """
        Build prompt for LLM INTERPRETATION (not decision-making).
        
        The ensemble has already made predictions.
        LLM's job: explain, add context, identify risks.
        """
        prompt = """You are interpreting ML model predictions. You DO NOT make trading decisions.

## YOUR ROLE
The ML ensemble (GRU + LSTM + XGBoost) has made predictions for each token.
Your job is to:
1. EXPLAIN why the models might be predicting this
2. IDENTIFY 2-3 risks or reasons the prediction could be wrong
3. FLAG any unusual conditions

## CRITICAL RULES
- Do NOT change or override model predictions
- Do NOT give your own BUY/SELL recommendations
- Express uncertainty when appropriate
- Be specific about risks, not generic

## TOKEN DATA WITH MODEL PREDICTIONS:
"""

        for symbol, data in all_data.items():
            if "error" in data:
                continue

            model_preds = data.get("model_predictions", {})
            gru_pred = model_preds.get("gru", {})
            lstm_pred = model_preds.get("lstm", {})
            xgb_pred = model_preds.get("xgboost", {})

            prompt += f"""
{symbol}:
  Price: ${data.get('price', 0):,.2f} ({data.get('change_24h', 0):+.1f}% 24h)
  RSI: {data.get('rsi', 'N/A'):.1f}
  MACD Histogram: {data.get('macd_histogram', 0):.4f}
  Market Regime: {data.get('regime', 'unknown')}
  
  ENSEMBLE PREDICTION: {data.get('ensemble_direction', 'NEUTRAL')} (prob UP: {data.get('ensemble_prob_up', 0.5):.1%})
  Model Agreement: {data.get('model_agreement', 0):.0%}
  - GRU: {gru_pred.get('direction', 'N/A')} ({gru_pred.get('probability_up', 0.5):.1%})
  - LSTM: {lstm_pred.get('direction', 'N/A')} ({lstm_pred.get('probability_up', 0.5):.1%})
  - XGBoost: {xgb_pred.get('direction', 'N/A')} ({xgb_pred.get('probability_up', 0.5):.1%})
"""

        prompt += """

## OUTPUT FORMAT
For each token, provide interpretation in this format (one block per token):

SYMBOL: [exact symbol]
INTERPRETATION: [1-2 sentences explaining why models predict this]
RISKS: [2-3 specific risks or counter-arguments]
UNUSUAL: [any unusual conditions to flag, or "None"]

Example:
SYMBOL: BTC/USDT
INTERPRETATION: Models predict DOWN likely due to overbought RSI (72) and bearish MACD crossover. All 3 models agree suggesting high confidence in short-term weakness.
RISKS: 1) Strong support at $85k could trigger bounce. 2) Models don't see pending ETF news. 3) Weekend liquidity is low, increasing false signal risk.
UNUSUAL: None

Now provide interpretation for each token:
"""
        return prompt

    def parse_interpretation(self, response_text: str, all_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Parse LLM interpretation response."""
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
            elif line.startswith("UNUSUAL:"):
                current_interp["unusual"] = line.replace("UNUSUAL:", "").strip()
        
        if current_symbol:
            interpretations[current_symbol] = current_interp
        
        return interpretations

    def _rotate_api_key(self):
        """Rotate to next API key."""
        new_key = self.config.llm.rotate_key()
        key_idx = self.config.llm._current_key_index
        print(f"[Batched] Rotating to API key #{key_idx + 1}/{self.config.llm.key_count}")

        if USE_NEW_API:
            self.client = genai.Client(api_key=new_key)
        else:
            genai.configure(api_key=new_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "max_output_tokens": self._max_output_tokens,
                    "temperature": 0.3,
                }
            )

    def analyze_all(self) -> List[TokenSignal]:
        """
        Analyze all tokens:
        1. Ensemble models make predictions
        2. LLM provides interpretation
        """
        print(f"[Batched] Gathering data and ensemble predictions for {len(self.tokens)} tokens...")

        # Gather data and get ensemble predictions
        all_data = self.gather_all_token_data()
        valid_tokens = [s for s, d in all_data.items() if "error" not in d]
        print(f"[Batched] Got ensemble predictions for {len(valid_tokens)} tokens")

        # Build interpretation prompt
        prompt = self.build_interpretation_prompt(all_data)

        # Get LLM interpretation
        max_retries = self.config.llm.key_count
        response_text = None

        for attempt in range(max_retries):
            print(f"[Batched] Getting LLM interpretation (attempt {attempt + 1}/{max_retries})...")
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
                print(f"[Batched] Got interpretation ({len(response_text)} chars)")
                break

            except Exception as e:
                error_str = str(e).lower()
                if '429' in error_str or 'quota' in error_str or 'rate' in error_str:
                    print(f"[Batched] Rate limit hit: {e}")
                    if attempt < max_retries - 1:
                        self._rotate_api_key()
                        time.sleep(1)
                        continue
                print(f"[Batched] LLM error: {e}")
                response_text = ""
                break

        # Parse interpretations
        interpretations = {}
        if response_text:
            interpretations = self.parse_interpretation(response_text, all_data)

        # Build signals from ENSEMBLE predictions (not LLM)
        signals = []
        for symbol, data in all_data.items():
            if "error" in data:
                continue

            price = data.get("price", 0)
            ensemble_dir = data.get("ensemble_direction", "NEUTRAL")
            ensemble_prob = data.get("ensemble_prob_up", 0.5)
            ensemble_conf = data.get("ensemble_confidence", 0)
            model_agreement = data.get("model_agreement", 0)
            model_preds = data.get("model_predictions", {})

            # Signal from ensemble (not LLM)
            if ensemble_dir == "UP" and ensemble_conf > 0.1:
                signal = "BUY"
                confidence = 50 + ensemble_conf * 50
            elif ensemble_dir == "DOWN" and ensemble_conf > 0.1:
                signal = "SELL"
                confidence = 50 + ensemble_conf * 50
            else:
                signal = "HOLD"
                confidence = 50

            # Reduce confidence if models disagree
            if model_agreement < 0.67:
                confidence *= 0.8

            # Stop loss and take profit
            sl_pct = 5
            tp_pct = 10

            # Get LLM interpretation
            interp = interpretations.get(symbol, {})

            signals.append(TokenSignal(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                entry_price=price,
                stop_loss=price * (1 - sl_pct/100) if signal == "BUY" else price * (1 + sl_pct/100),
                take_profit=price * (1 + tp_pct/100) if signal == "BUY" else price * (1 - tp_pct/100),
                ensemble_prediction=ensemble_dir,
                model_agreement=model_agreement,
                gru_direction=model_preds.get("gru", {}).get("direction", ""),
                lstm_direction=model_preds.get("lstm", {}).get("direction", ""),
                xgboost_direction=model_preds.get("xgboost", {}).get("direction", ""),
                llm_interpretation=interp.get("interpretation", ""),
                llm_risks=interp.get("risks", ""),
                llm_counter_arguments=interp.get("unusual", ""),
            ))

        print(f"[Batched] Generated {len(signals)} signals from ensemble")
        actionable = [s for s in signals if s.signal != "HOLD" and s.confidence >= 60]
        print(f"[Batched] Actionable signals: {len(actionable)}")

        return signals


# Singleton
_batched_analyzer = None

def get_batched_analyzer() -> BatchedAnalyzer:
    global _batched_analyzer
    if _batched_analyzer is None:
        _batched_analyzer = BatchedAnalyzer()
    return _batched_analyzer


if __name__ == "__main__":
    analyzer = BatchedAnalyzer()
    signals = analyzer.analyze_all()

    print("\n" + "=" * 60)
    print("BATCHED ANALYSIS (Ensemble + LLM Interpretation)")
    print("=" * 60)

    for sig in signals:
        print(f"\n{sig.symbol}:")
        print(f"  Signal: {sig.signal} ({sig.confidence:.0f}%)")
        print(f"  Ensemble: {sig.ensemble_prediction} (Agreement: {sig.model_agreement:.0%})")
        print(f"  Models: GRU={sig.gru_direction}, LSTM={sig.lstm_direction}, XGB={sig.xgboost_direction}")
        if sig.llm_interpretation:
            print(f"  Interpretation: {sig.llm_interpretation}")
        if sig.llm_risks:
            print(f"  Risks: {sig.llm_risks}")
