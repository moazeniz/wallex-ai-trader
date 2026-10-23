"""Deterministic capital controls around externally proposed research actions."""
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_DOWN
import math
import re

D = Decimal
def valid_tmn_symbol(symbol):
    return isinstance(symbol,str) and re.fullmatch(r'[A-Z0-9]{1,20}TMN',symbol) is not None


SEVERE_FLAGS = {"market_manipulation", "exchange_incident", "security_incident"}

@dataclass(frozen=True)
class RiskConfig:
    initial_cash_tmn: str = "5000000"
    fee_rate: str = "0.0035"
    slippage_rate: str = "0.001"
    max_book_impact_fraction: str = "0.003"
    min_order_tmn: str = "100000"
    max_position_fraction: str = "0.20"
    max_total_exposure: str = "0.40"
    risk_per_trade: str = "0.005"
    max_drawdown: str = "0.03"
    minimum_stop_fraction: str = "0.01"
    maximum_stop_fraction: str = "0.025"
    atr_stop_multiple: str = "1.5"
    reward_risk: str = "2"
    minimum_net_edge: float = 0.002
    maximum_spread: float = 0.004
    maximum_snapshot_age_seconds: int = 60
    minimum_model_confidence: float = 0.6
    max_positions: int = 2
    trading_cutoff_epoch: float | None = None
    settlement_buffer_seconds: int = 1800
    research_universe: tuple[str,...] = ()

    def __post_init__(self):
        if self.trading_cutoff_epoch is not None and (not math.isfinite(self.trading_cutoff_epoch) or self.trading_cutoff_epoch<=0):raise ValueError('Invalid official cutoff')
        if self.settlement_buffer_seconds<60:raise ValueError('Settlement buffer too short')
        if not isinstance(self.research_universe,(tuple,list)) or any(not valid_tmn_symbol(symbol) for symbol in self.research_universe):
            raise ValueError('Invalid TMN research universe')
        if len(set(self.research_universe))!=len(self.research_universe):raise ValueError('Duplicate research universe symbol')
        object.__setattr__(self,'research_universe',tuple(self.research_universe))
        for name in ["initial_cash_tmn", "min_order_tmn", "atr_stop_multiple", "reward_risk"]:
            if not D(getattr(self,name)).is_finite() or D(getattr(self,name)) <= 0: raise ValueError(name)
        for name in ["fee_rate", "slippage_rate", "max_position_fraction", "max_total_exposure", "risk_per_trade", "max_drawdown", "minimum_stop_fraction", "maximum_stop_fraction", "max_book_impact_fraction"]:
            x=D(getattr(self,name))
            if not x.is_finite() or not 0<x<1: raise ValueError(name)
        if D(self.max_position_fraction)>D(self.max_total_exposure): raise ValueError("Position cap exceeds total cap")
        if D(self.minimum_stop_fraction)>D(self.maximum_stop_fraction): raise ValueError("Stop bounds")
        for x in [self.minimum_net_edge,self.maximum_spread,self.minimum_model_confidence]:
            if not math.isfinite(x) or not 0<x<1:raise ValueError("Invalid finite risk threshold")
        if D(self.slippage_rate)>D(self.max_book_impact_fraction):raise ValueError("Configured impact exceeds execution price limit")
        if type(self.max_positions) is not int or type(self.maximum_snapshot_age_seconds) is not int or self.max_positions<1 or self.maximum_snapshot_age_seconds<1:raise ValueError("Invalid position/time setting")


def net_expected_return(gross, spread, config):
    values=[gross,spread]
    if not all(math.isfinite(x) for x in values) or gross<=-1 or not 0<=spread<1:raise ValueError("Invalid expected return or spread")
    fee=float(config.fee_rate);slip=float(config.slippage_rate)
    # Quotes are bid/ask around the mid; spread and configured impact are separate explicit assumptions.
    spread_factor=(1-spread/2)/(1+spread/2)
    return (1+gross)*(1-slip)*(1-fee)*spread_factor/((1+slip)*(1+fee))-1


def validate_context(context, now, config):
    """Validate both observation clocks before using captured quotes for any action."""
    current=float(now)
    if not math.isfinite(current):raise ValueError('Invalid execution clock')
    symbol=context['symbol'];book=context['book']
    if not valid_tmn_symbol(symbol) or context['market']['symbol']!=symbol or (config.research_universe and symbol not in config.research_universe):
        raise ValueError('Unsupported or mismatched paper symbol')
    outer=float(context['observed_at']);inner=float(book['observed_at'])
    for stamp in (outer,inner):
        if not math.isfinite(stamp) or stamp>current+5 or current-stamp>config.maximum_snapshot_age_seconds:
            raise ValueError('Cannot use stale/future snapshot or book')
    if abs(inner-outer)>config.maximum_snapshot_age_seconds:
        raise ValueError('Snapshot/book acquisition delay exceeds limit')
    bid,ask,spread=(float(book[key]) for key in ('bid','ask','spread_fraction'))
    if not all(math.isfinite(value) for value in (bid,ask,spread)) or not 0<bid<ask or not 0<=spread<1:
        raise ValueError('Invalid or crossed best quotes')
    actual=(ask-bid)/((ask+bid)/2)
    if not math.isclose(actual,spread,rel_tol=1e-7,abs_tol=1e-9):
        raise ValueError('Reported spread differs from best quotes')


def resolve_action(context, forecast, assessments, portfolio, now, config):
    """Combine specialist proposals and hard constraints without giving models execution authority."""
    try:validate_context(context,now,config)
    except (ValueError,KeyError,TypeError):
        return {'action':'HOLD','reason':['stale_future_or_invalid_market_data'],'forced_risk_exit':False}
    symbol=context['symbol'];position=portfolio['positions'].get(symbol)
    reasons=[];book=context['book'];action='HOLD'
    if book['spread_fraction']>config.maximum_spread:reasons.append('spread_above_limit')
    if position:
        if config.trading_cutoff_epoch is not None and now>=config.trading_cutoff_epoch-config.settlement_buffer_seconds:
            return {'action':'SELL','reason':['final_toman_settlement_window'],'forced_risk_exit':True}
        bid=D(str(book['bid']))
        if portfolio['halted']:return {'action':'SELL','reason':['portfolio_drawdown_halt'],'forced_risk_exit':True}
        if bid<=D(position['stop_price']):return {'action':'SELL','reason':['stop_loss'],'forced_risk_exit':True}
        if bid>=D(position['take_price']):return {'action':'SELL','reason':['take_profit'],'forced_risk_exit':True}
        if now>=position['expires_at']:return {'action':'SELL','reason':['holding_horizon_expired'],'forced_risk_exit':True}
    if reasons:return {'action':'HOLD','reason':reasons,'forced_risk_exit':False}
    if not assessments or any(x.get('status')!='ok' for x in assessments):return {'action':'HOLD','reason':['incomplete_research_council'],'forced_risk_exit':False}
    votes=[x['proposed_action'] for x in assessments]
    severe=sorted({f for x in assessments for f in x['risk_flags']} & SEVERE_FLAGS)
    if position:
        if severe or votes.count('SELL')>=2:
            return {'action':'SELL','reason':severe or ['council_exit_consensus'],'forced_risk_exit':False}
        return {'action':'HOLD','reason':['retain_position_until_exit_rule'],'forced_risk_exit':False}
    if config.trading_cutoff_epoch is not None and now>=config.trading_cutoff_epoch-config.settlement_buffer_seconds:reasons.append('settlement_window_no_new_entries')
    if portfolio['halted']:reasons.append('portfolio_drawdown_halt')
    if len(portfolio['positions'])>=config.max_positions:reasons.append('position_count_limit')
    if severe:reasons.extend(severe)
    if votes.count('BUY')<2:reasons.append('no_buy_consensus')
    approving=[x for x in assessments if x['proposed_action']=='BUY']
    if approving and any(x['confidence']<config.minimum_model_confidence for x in approving):reasons.append('low_analyst_confidence')
    gross=float(forecast['predicted_gross_return'])
    expected=net_expected_return(gross,float(book['spread_fraction']),config)
    if expected<=config.minimum_net_edge:reasons.append('expected_edge_does_not_cover_costs_and_margin')
    # A refreshed book must not make an older analysis bar look current. The latest
    # fully completed UTC hourly candle closes exactly at floor(now/hour)*hour.
    # This entry-only gate follows all forced-exit branches, so stale analysis
    # never prevents fresh-quote stops, holding timeouts, or drawdown liquidation.
    closed=float(context['technical']['last_closed_at'])
    if not math.isfinite(closed) or not closed.is_integer() or int(closed)%3600 or int(closed)!=int(now)//3600*3600:
        reasons.append('completed_analysis_bar_not_current')
    if int(forecast['training_cutoff'])>=int(context['technical']['last_closed_at']):reasons.append('model_training_overlaps_decision')
    if int(forecast.get('calibration_cutoff',0))>=int(context['technical']['last_closed_at']):reasons.append('model_calibration_overlaps_decision')
    if forecast.get('feature_cutoff') is not None and int(forecast['feature_cutoff'])!=int(context['technical']['last_closed_at']):reasons.append('forecast_snapshot_bar_mismatch')
    if not 1<=int(forecast['horizon_hours'])<=24:reasons.append('unsupported_forecast_horizon')
    if not reasons:action='BUY'
    return {'action':action,'reason':reasons or ['quantitative_edge_and_council_agree'],
            'expected_net_return_including_observed_spread':expected,'forced_risk_exit':False,
            'confidence_is_not_calibrated_win_probability':True}


def entry_plan(context, portfolio, config):
    symbol=context['symbol'];price=D(str(context['book']['ask']));equity=D(portfolio['equity_tmn'])
    exposure=sum((D(p['quantity'])*D(p['mark_price']) for p in portfolio['positions'].values()),D(0))
    atr=D(str(context['technical']['one_hour'].get('atr_fraction') or 0))
    stop=max(D(config.minimum_stop_fraction),min(D(config.maximum_stop_fraction),atr*D(config.atr_stop_multiple)))
    cost=2*(D(config.fee_rate)+D(config.slippage_rate))
    budget=min(D(portfolio['cash_tmn'])/(1+D(config.fee_rate)),equity*D(config.max_position_fraction),
               max(D(0),equity*D(config.max_total_exposure)-exposure),equity*D(config.risk_per_trade)/(stop+cost))
    return {'maximum_notional_tmn':str(max(D(0),budget)), 'stop_fraction':str(stop),
            'take_fraction':str(stop*D(config.reward_risk)),'currency':'TMN','symbol':symbol}
