from __future__ import annotations
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
import hashlib, json, math, statistics

D = Decimal
FEATURE_NAMES = ["return_1", "return_3", "return_12", "volatility_12", "mean_gap_6_24", "relative_volume", "range"]

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)

@dataclass(frozen=True)
class Candle:
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float


def validate_candles(rows, interval):
    if interval <= 0 or not rows:
        raise ValueError("Empty data or invalid interval")
    previous = None
    for b in rows:
        if not all(math.isfinite(x) for x in [b.o,b.h,b.l,b.c,b.v]):
            raise ValueError("Non-finite OHLCV")
        if b.t % interval or (previous is not None and b.t - previous != interval):
            raise ValueError("Candle times must be aligned, unique, and contiguous")
        if min(b.o,b.h,b.l,b.c) <= 0 or b.v < 0 or b.h < max(b.o,b.c) or b.l > min(b.o,b.c):
            raise ValueError("Invalid OHLCV bounds")
        previous = b.t


def normalize_history(payload, target_interval, closed_before):
    if payload.get("s") != "ok":
        raise ValueError("History API did not return s=ok")
    keys = ["t", "o", "h", "l", "c", "v"]
    if len({len(payload[k]) for k in keys}) != 1:
        raise ValueError("History arrays have different lengths")
    raw = [Candle(int(t), *map(float, rest)) for t,*rest in zip(*(payload[k] for k in keys))]
    if len(raw) < 2:
        raise ValueError("Insufficient raw candles")
    if any(b.t <= a.t for a,b in zip(raw,raw[1:])):
        raise ValueError("Raw candles are duplicated or unsorted")
    source_interval = min(b.t-a.t for a,b in zip(raw,raw[1:]))
    if target_interval % source_interval or source_interval > target_interval:
        raise ValueError("Cannot aggregate source interval into target interval")
    groups = {}
    for b in raw:
        if not all(math.isfinite(x) for x in [b.o,b.h,b.l,b.c,b.v]) or min(b.o,b.h,b.l,b.c)<=0 or b.v<0 or b.h<max(b.o,b.c) or b.l>min(b.o,b.c):
            raise ValueError("Invalid raw candle")
        if b.t % source_interval:
            raise ValueError("Unaligned raw candle")
        bucket = b.t//target_interval*target_interval
        if bucket+target_interval <= closed_before:
            groups.setdefault(bucket,[]).append(b)
    rows=[]; incomplete=0
    expected=target_interval//source_interval
    for t,bars in sorted(groups.items()):
        # Refuse partial buckets; do not fabricate missing prices or volume.
        if len(bars)!=expected or bars[0].t!=t or bars[-1].t!=t+target_interval-source_interval:
            incomplete+=1;continue
        rows.append(Candle(t,bars[0].o,max(b.h for b in bars),min(b.l for b in bars),bars[-1].c,sum(b.v for b in bars)))
    if not rows:
        raise ValueError("No complete candles")
    # Retain the newest uninterrupted segment so evaluation never crosses a data gap.
    start=0
    for i in range(1,len(rows)):
        if rows[i].t-rows[i-1].t!=target_interval:start=i
    selected=rows[start:]
    validate_candles(selected,target_interval)
    return selected, {"source_interval_seconds":source_interval,"target_interval_seconds":target_interval,
        "raw_candles":len(raw),"complete_candles":len(rows),"incomplete_buckets_dropped":incomplete,
        "older_candles_before_last_gap":start,"selected_candles":len(selected),
        "policy":"complete buckets only; newest contiguous segment; no imputation"}


@dataclass(frozen=True)
class Settings:
    initial_cash: str = "5000000"
    fee_rate: str = "0.0035"
    slippage: str = "0.001"
    min_order_tmn: str = "100000"
    max_position_fraction: str = "0.20"
    risk_fraction: str = "0.005"
    stop_fraction: str = "0.01"
    take_fraction: str = "0.02"
    max_drawdown: str = "0.03"
    volume_participation: str = "0.01"
    buy_probability: float = 0.60
    sell_probability: float = 0.45
    max_hold_bars: int = 8
    horizon: int = 4
    interval_seconds: int = 900

    def __post_init__(self):
        for name in ["initial_cash","min_order_tmn"]:
            value=D(getattr(self,name))
            if not value.is_finite() or value<=0:raise ValueError(name)
        for name in ["fee_rate","slippage","max_position_fraction","risk_fraction","stop_fraction","take_fraction","max_drawdown","volume_participation"]:
            value=D(getattr(self,name))
            if not value.is_finite() or value<=0 or value>=1:raise ValueError(name)
        if not (0<self.sell_probability<self.buy_probability<1):raise ValueError("Probability thresholds")
        if self.interval_seconds<=0 or self.horizon<1 or self.max_hold_bars<1:raise ValueError("Invalid time setting")

@dataclass(frozen=True)
class Market:
    symbol: str
    amount_decimals: int
    minimum_quantity: str
    minimum_notional: str

    def __post_init__(self):
        if not self.symbol.endswith("TMN"):raise ValueError("Research broker requires a TMN quote")
        if not 0<=self.amount_decimals<=18:raise ValueError("Invalid amount precision")
        for value in [self.minimum_quantity,self.minimum_notional]:
            if not D(value).is_finite() or D(value)<0:raise ValueError("Invalid market minimum")

    def round_quantity(self, amount):
        return amount.quantize(D(1).scaleb(-self.amount_decimals),rounding=ROUND_DOWN)


def features(rows, i):
    if i<24:raise ValueError("Feature warmup requires 24 prior bars")
    closes=[b.c for b in rows[i-24:i+1]]
    returns=[math.log(b/a) for a,b in zip(closes,closes[1:])]
    mean_volume=statistics.fmean(b.v for b in rows[i-23:i+1])
    current=rows[i]
    return [returns[-1],math.log(current.c/rows[i-3].c),math.log(current.c/rows[i-12].c),
        statistics.pstdev(returns[-12:]),statistics.fmean(closes[-6:])/statistics.fmean(closes[-24:])-1,
        current.v/mean_volume-1 if mean_volume else 0,(current.h-current.l)/current.c]


def fit_model(rows, train_end, settings):
    # Signal is formed at close(i), filled at open(i+1); label matures at open(i+1+horizon).
    indices=list(range(24,train_end-settings.horizon-1))
    if len(indices)<80:raise ValueError("Insufficient training examples after label embargo")
    xs=[features(rows,i) for i in indices]
    forward=[rows[i+1+settings.horizon].o/rows[i+1].o-1 for i in indices]
    ys=[int(r>0) for r in forward]
    if len(set(ys))<2:raise ValueError("Training window contains only one target class")
    means=[statistics.fmean(x[j] for x in xs) for j in range(len(FEATURE_NAMES))]
    scales=[max(1e-9,statistics.pstdev(x[j] for x in xs)) for j in range(len(FEATURE_NAMES))]
    standardized=[[1]+[max(-10,min(10,(x[j]-means[j])/scales[j])) for j in range(len(means))] for x in xs]
    weights=[0.0]*(len(means)+1)
    for epoch in range(250):
        gradient=[0.0]*len(weights)
        for x,y in zip(standardized,ys):
            z=max(-30,min(30,sum(a*b for a,b in zip(weights,x))))
            error=1/(1+math.exp(-z))-y
            for j in range(len(weights)):gradient[j]+=error*x[j]
        rate=0.15/(1+epoch/100)
        for j in range(len(weights)):
            weights[j]-=rate*(gradient[j]/len(xs)+(0.02*weights[j] if j else 0))
    return {"kind":"standardized_l2_logistic_regression","feature_names":FEATURE_NAMES,
        "means":means,"scales":scales,"weights":weights,"training_rows":len(xs),
        "last_label_time":rows[indices[-1]+1+settings.horizon].t,
        "train_end_exclusive":train_end,"training_data_sha256":digest([asdict(b) for b in rows[:train_end]]),
        "mean_positive_return":statistics.fmean(r for r in forward if r>0),
        "mean_nonpositive_return":statistics.fmean(r for r in forward if r<=0),
        "target":"open(i+1+horizon)/open(i+1)>1; embargo before holdout"}


def predict(model, values):
    z=model["weights"][0]+sum(w*max(-10,min(10,(x-m)/s)) for w,x,m,s in zip(model["weights"][1:],values,model["means"],model["scales"]))
    return 1/(1+math.exp(-max(-30,min(30,z))))


def decision(model, probability, settings):
    edge=probability*model["mean_positive_return"]+(1-probability)*model["mean_nonpositive_return"]
    costs=2*(float(settings.fee_rate)+float(settings.slippage))
    if probability>=settings.buy_probability and edge>costs:
        return "BUY",{"expected_gross_return":edge,"assumed_roundtrip_cost":costs,"reason":"learned_probability_and_expected_edge_pass"}
    if probability<=settings.sell_probability:
        return "SELL",{"expected_gross_return":edge,"assumed_roundtrip_cost":costs,"reason":"learned_probability_exit"}
    return "HOLD",{"expected_gross_return":edge,"assumed_roundtrip_cost":costs,"reason":"insufficient_probability_or_cost_adjusted_edge"}


class Audit:
    def __init__(self, path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.previous="0"*64;self.seq=0
        if self.path.exists():
            for event in verify_audit(self.path):self.previous=event["hash"];self.seq=event["seq"]+1

    def append(self, payload):
        event={"seq":self.seq,"previous":self.previous,"payload":payload}
        event["hash"]=digest(event)
        with self.path.open("a") as f:
            f.write(canonical(event)+"\n");f.flush()
            import os;os.fsync(f.fileno())
        self.previous=event["hash"];self.seq+=1
        return event


def verify_audit(path):
    previous="0"*64
    for seq,line in enumerate(Path(path).read_text().splitlines()):
        event=json.loads(line);copy=dict(event);h=copy.pop("hash")
        if event["seq"]!=seq or event["previous"]!=previous or digest(copy)!=h:raise ValueError("Audit chain mismatch")
        previous=h;yield event


class PaperBroker:
    def __init__(self, market, settings):
        self.market=market;self.s=settings;self.cash=D(settings.initial_cash);self.quantity=D(0)
        self.entry=D(0);self.entry_cost=D(0);self.entry_index=-1;self.peak=self.cash;self.halted=False
        self.fees=D(0);self.trades=[];self.max_dd=D(0)

    def equity(self, price):return self.cash+self.quantity*D(str(price))

    def mark(self, price):
        equity=self.equity(price);self.peak=max(self.peak,equity)
        dd=(self.peak-equity)/self.peak;self.max_dd=max(self.max_dd,dd)
        if dd>=D(self.s.max_drawdown):self.halted=True
        return equity

    def buy(self, reference, volume, index, timestamp):
        if self.quantity or self.halted:return {"blocked":"already_in_position_or_halted"}
        price=D(str(reference))*(1+D(self.s.slippage));equity=self.equity(reference)
        fee=D(self.s.fee_rate)
        risk_budget=equity*D(self.s.risk_fraction)/(D(self.s.stop_fraction)+2*(fee+D(self.s.slippage)))
        budget=min(self.cash/(1+fee),equity*D(self.s.max_position_fraction),risk_budget)
        quantity=self.market.round_quantity(min(budget/price,D(str(volume))*D(self.s.volume_participation)))
        notional=quantity*price
        if quantity<D(self.market.minimum_quantity) or notional<max(D(self.market.minimum_notional),D(self.s.min_order_tmn)):
            return {"blocked":"minimum_order_or_liquidity","quantity":str(quantity)}
        charge=notional*fee
        self.cash-=notional+charge;self.quantity=quantity;self.entry=price;self.entry_cost=notional+charge;self.entry_index=index
        self.fees+=charge
        fill={"side":"BUY","t":timestamp,"quantity":str(quantity),"price":str(price),"fee":str(charge),"kind":"simulated"}
        self.trades.append(fill);return fill

    def sell(self, reference, timestamp, reason, volume=None):
        if not self.quantity:return {"blocked":"no_position"}
        quantity=self.quantity if volume is None else self.market.round_quantity(min(self.quantity,D(str(volume))*D(self.s.volume_participation)))
        if quantity<=0:return {"blocked":"no_exit_liquidity"}
        price=D(str(reference))*(1-D(self.s.slippage));notional=quantity*price
        # A real venue can reject a dust exit. Report it instead of assuming settlement.
        if quantity<D(self.market.minimum_quantity) or notional<max(D(self.market.minimum_notional),D(self.s.min_order_tmn)):
            return {"blocked":"exit_below_minimum","remaining_quantity":str(self.quantity)}
        fee=notional*D(self.s.fee_rate);basis=self.entry_cost*quantity/self.quantity
        self.cash+=notional-fee;self.quantity-=quantity;self.entry_cost-=basis;self.fees+=fee
        fill={"side":"SELL","t":timestamp,"quantity":str(quantity),"price":str(price),"fee":str(fee),"realized_pnl":str(notional-fee-basis),"reason":reason,"kind":"simulated"}
        self.trades.append(fill)
        if not self.quantity:self.entry=D(0);self.entry_cost=D(0);self.entry_index=-1
        return fill


def simulate(rows, model, market, settings, evaluation_start):
    validate_candles(rows,settings.interval_seconds)
    if model["last_label_time"]>=rows[evaluation_start-1].t:raise ValueError("Training labels overlap evaluation signal")
    broker=PaperBroker(market,settings);events=[];curve=[]
    for i in range(evaluation_start,len(rows)):
        bar=rows[i];signal_time=rows[i-1].t+settings.interval_seconds
        x=features(rows,i-1);probability=predict(model,x);action,explanation=decision(model,probability,settings)
        fills=[]
        # Risk exits at open before any new entry. Intra-bar ambiguity resolves stop first.
        broker.mark(bar.o)
        if broker.quantity and (broker.halted or action=="SELL" or i-broker.entry_index>=settings.max_hold_bars):
            reason="drawdown_halt" if broker.halted else ("model_exit" if action=="SELL" else "holding_timeout")
            fills.append(broker.sell(bar.o,bar.t,reason,bar.v))
        elif not broker.quantity and action=="BUY" and i<len(rows)-1:
            # Prior closed volume sizes the order; current volume only caps the simulated fill.
            fills.append(broker.buy(bar.o,min(rows[i-1].v,bar.v),i,bar.t))
        if broker.quantity:
            stop=broker.entry*(1-D(settings.stop_fraction));take=broker.entry*(1+D(settings.take_fraction))
            if D(str(bar.l))<=stop:
                fills.append(broker.sell(min(bar.o,float(stop)),bar.t,"stop_loss",bar.v))
            elif D(str(bar.h))>=take:
                fills.append(broker.sell(float(take),bar.t,"take_profit",bar.v))
        if i==len(rows)-1 and broker.quantity:
            fills.append(broker.sell(bar.c,bar.t+settings.interval_seconds,"end_of_test_settlement",bar.v))
        equity=broker.mark(bar.c)
        curve.append({"t":bar.t,"equity_tmn":str(equity)})
        events.append({"kind":"backtest_decision","signal_time":signal_time,"execution_bar":bar.t,
            "feature_values":x,"probability_up":probability,"action":action,"explanation":explanation,
            "fills":fills,"cash_tmn":str(broker.cash),"quantity":str(broker.quantity),"equity_tmn":str(equity),"halted":broker.halted})
    initial=D(settings.initial_cash);final=broker.equity(rows[-1].c)
    sells=[f for f in broker.trades if f['side']=='SELL']
    bh=(D(str(rows[-1].c))/D(str(rows[evaluation_start].o))*(1-D(settings.slippage))/(1+D(settings.slippage))*(1-D(settings.fee_rate))/(1+D(settings.fee_rate))-1)*100
    report={"mode":"historical_simulation","final_equity_tmn":str(final),"cash_tmn":str(broker.cash),"remaining_quantity":str(broker.quantity),
        "net_return_percent":float((final/initial-1)*100),"max_drawdown_percent":float(broker.max_dd*100),"fees_tmn":str(broker.fees),
        "fill_count":len(broker.trades),"exit_fill_count":len(sells),"realized_pnl_tmn":str(sum((D(f['realized_pnl']) for f in sells),D(0))),
        "settled_to_tmn":not bool(broker.quantity),"buy_hold_net_return_percent":float(bh),"cash_baseline_return_percent":0,
        "evaluation_bars":len(curve),"start":rows[evaluation_start].t,"end":rows[-1].t+settings.interval_seconds,
        "limitations":["Fees and slippage are assumptions until account fee is available","OHLCV cannot reproduce queue priority or latency","No validated live or shadow profitability","Drawdown checked at bar open and close; gaps can exceed stops","No exchange order was submitted"]}
    return report,events,curve
