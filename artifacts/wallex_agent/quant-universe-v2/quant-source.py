"""Causal TMN forecasts and frozen, chronological development research.

No tuning on evaluation prices. Native JSON artifacts only: loading never unpickles code.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import math
import re
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

FEATURE_NAMES = ["return_1", "return_3", "return_8", "return_24", "return_48",
    "volatility_8", "volatility_24", "mean_gap_12_48", "relative_volume_24",
    "range_fraction", "atr_14_fraction", "rsi_14_scaled", "drawdown_48"]
CANDIDATES = ("ridge", "tree", "ensemble", "zero", "trend")
WARMUP = 48
INTERVAL = 3600
HORIZON = 8
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{1,24}TMN$")

def valid_symbol(symbol):
    return isinstance(symbol,str) and bool(SYMBOL_PATTERN.fullmatch(symbol))

TREE_SETTINGS = {"objective":"reg:squarederror","tree_method":"hist","device":"cpu",
    "n_estimators":300,"max_depth":3,"learning_rate":0.03,"min_child_weight":20,
    "subsample":0.8,"colsample_bytree":0.8,"reg_alpha":0.0,"reg_lambda":10.0,
    "random_state":42,"n_jobs":4}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)

def candle_dict(candle):
    keys=("t","o","h","l","c","v")
    raw={k:candle[k] for k in keys} if isinstance(candle,dict) else {k:getattr(candle,k) for k in keys}
    return {k:float(raw[k]) for k in keys}

def validate_rows(candles):
    rows = [candle_dict(c) for c in candles]
    previous = None
    for r in rows:
        values = [float(r[k]) for k in ("o","h","l","c","v")]
        if not all(math.isfinite(x) for x in values): raise ValueError("nonfinite OHLCV")
        if int(r["t"]) != r["t"] or int(r["t"]) % INTERVAL: raise ValueError("unaligned hourly candle")
        if previous is not None and int(r["t"]) <= previous: raise ValueError("unsorted/duplicate candles")
        if min(values[:4]) <= 0 or values[-1] < 0: raise ValueError("nonpositive price/negative volume")
        if r["h"] < max(r["o"],r["c"]) or r["l"] > min(r["o"],r["c"]): raise ValueError("OHLC bounds")
        previous = int(r["t"])
    return rows

def feature_vector(candles, index=None):
    """Use only the48 prior bars and closed current bar. Refuse a gap in this window."""
    i = len(candles)-1 if index is None else index
    if i < WARMUP: raise ValueError("48 contiguous prior hours required")
    rows = [candle_dict(r) for r in candles[i-WARMUP:i+1]]
    if len(rows) != WARMUP+1: raise ValueError("feature index out of range")
    if any(b["t"]-a["t"] != INTERVAL for a,b in zip(rows,rows[1:])):
        raise ValueError("feature warmup crosses a gap")
    c=np.asarray([r["c"] for r in rows],dtype=float)
    if not np.isfinite(c).all() or (c<=0).any(): raise ValueError("invalid price")
    returns=np.diff(np.log(c)); v=np.asarray([r["v"] for r in rows[-24:]],dtype=float)
    true_range=[max(r["h"]-r["l"],abs(r["h"]-rows[j-1]["c"]),abs(r["l"]-rows[j-1]["c"])) for j,r in enumerate(rows) if j>0]
    change=np.diff(c[-15:]); gain=np.maximum(change,0).mean(); loss=np.maximum(-change,0).mean()
    rsi=0.5 if gain+loss==0 else gain/(gain+loss)
    x=[float(c[-1]/c[-1-n]-1) for n in (1,3,8,24,48)]
    x += [float(returns[-8:].std()),float(returns[-24:].std()),float(c[-12:].mean()/c[-48:].mean()-1),
        float(v[-1]/v.mean()-1) if v.mean()>0 else 0.0,
        float((rows[-1]["h"]-rows[-1]["l"])/c[-1]),float(np.mean(true_range[-14:])/c[-1]),
        float(rsi-0.5),float(c[-1]/max(c[-48:])-1)]
    if not np.isfinite(x).all(): raise ValueError("nonfinite features")
    return x

def make_samples(candles):
    rows=validate_rows(candles); samples=[]
    for i in range(WARMUP,len(rows)-HORIZON-1):
        # Every label and feature must reside entirely within one uninterrupted segment.
        if any(rows[j+1]["t"]-rows[j]["t"]!=INTERVAL for j in range(i-WARMUP,i+HORIZON+1)): continue
        samples.append({"index":i,"feature_cutoff":int(rows[i]["t"])+INTERVAL,
            "entry_time":int(rows[i+1]["t"]),"label_end":int(rows[i+1+HORIZON]["t"]),
            "features":feature_vector(rows,i),"target":float(rows[i+1+HORIZON]["o"]/rows[i+1]["o"]-1),
            "opportunity":int(rows[i+1]["t"])//INTERVAL%HORIZON==0})
    return samples

def split_samples(samples, boundaries):
    start,train_end,cal_end,end=map(int,boundaries)
    if not start<train_end<cal_end<end: raise ValueError("invalid split boundaries")
    out={}
    for name,lo,hi in (("train",start,train_end),("calibration",train_end,cal_end),("evaluation",cal_end,end)):
        out[name]=[s for s in samples if lo<=s["feature_cutoff"]<hi and s["label_end"]<hi]
    return out

def cost_factor(fee, slippage):
    f,s=float(fee),float(slippage)
    if not all(math.isfinite(v) and 0<=v<1 for v in (f,s)): raise ValueError("invalid costs")
    return ((1-s)*(1-f))/((1+s)*(1+f))

def net_return(gross_return, fee, slippage):
    if not math.isfinite(float(gross_return)) or gross_return < -1: raise ValueError("invalid return")
    return float((1+gross_return)*cost_factor(fee,slippage)-1)

def fit_estimators(train):
    if len(train)<120: raise ValueError("insufficient training data")
    x=np.array([s["features"] for s in train]); y=np.array([s["target"] for s in train])
    ridge=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    tree=XGBRegressor(**TREE_SETTINGS)
    with threadpool_limits(limits=4):
        ridge.fit(x,y); tree.fit(x,y)
    scaler=ridge.steps[0][1]; linear=ridge.steps[1][1]
    params={"means":scaler.mean_.tolist(),"scales":scaler.scale_.tolist(),
        "coefficients":linear.coef_.tolist(),"intercept":float(linear.intercept_)}
    return params,tree

def predictions(params,tree,x):
    x=np.asarray(x,dtype=float)
    ridge=(x-np.array(params["means"]))/np.array(params["scales"])@np.array(params["coefficients"])+params["intercept"]
    with threadpool_limits(limits=4): predicted=tree.predict(x).astype(float)
    return {"ridge":ridge,"tree":predicted,"ensemble":(ridge+predicted)/2,
        "zero":np.zeros(len(x)),"trend":x[:,3]/3}  # trend return prediction is descriptive; entry is fixed rule.

def metrics(actual,predicted):
    a=np.asarray(actual); p=np.asarray(predicted); denom=float(np.sum(a*a))
    return {"count":len(a),"mae":float(np.mean(abs(p-a))) if len(a) else None,
        "mse":float(np.mean((p-a)**2)) if len(a) else None,
        "oos_r2_vs_zero":float(1-np.sum((p-a)**2)/denom) if denom else None,
        "mean_prediction":float(p.mean()) if len(p) else None,"mean_actual":float(a.mean()) if len(a) else None,
        "direction_agreement":float(np.mean((p>0)==(a>0))) if len(a) else None}

def entry_decision(candidate,sample,predicted,fee,slip,margin,residual_low=None):
    if candidate=="zero": return False
    if candidate=="trend":
        technical=sample["features"][3]>0 and sample["features"][7]>0
        return bool(technical and (not sample.get("trend_cost_gate",False) or net_return(float(predicted),fee,slip)>margin))
    accepted=net_return(float(predicted),fee,slip)>margin
    if residual_low is not None: accepted=accepted and net_return(float(predicted)+residual_low,fee,slip)>0
    return bool(accepted)

def simulate_sleeve(rows,samples,preds,candidate,fee,slip,margin,delay_hours=0,cost_multiplier=1,residual_low=None,evaluation_end_exclusive=None):
    """One fully funded independent sleeve; fixed signals, optional delayed fills/cost stress.

    No sizing, stop or live-book claim. Fees, slippage and mark-to-close drawdown are explicit.
    """
    opportunities=[(s,float(p)) for s,p in zip(samples,preds) if s["opportunity"]]
    equity=1.0; peak=1.0; maxdd=0.0; trades=[]; curve=[]; busy_until=-1; requested=0; unavailable=0
    f,s=fee*cost_multiplier,slip*cost_multiplier
    for sample,p in opportunities:
        if not entry_decision(candidate,sample,p,fee,slip,margin,residual_low):
            if sample["index"]+1>=busy_until:
                curve.append({"t":sample["entry_time"],"equity":equity})
            continue
        requested+=1; a=sample["index"]+1+delay_hours; b=a+HORIZON
        if b>=len(rows) or a<busy_until or (evaluation_end_exclusive is not None and rows[b]["t"]>=evaluation_end_exclusive) or any(rows[j+1]["t"]-rows[j]["t"]!=INTERVAL for j in range(a,b)):
            unavailable+=1;continue
        initial=equity; entry=rows[a]["o"];exit_price=rows[b]["o"]
        for j in range(a,b):
            marked=initial*rows[j]["c"]/entry*(1-s)/((1+s)*(1+f))
            peak=max(peak,marked); maxdd=max(maxdd,1-marked/peak)
        gross=float(exit_price/entry-1); net=net_return(gross,f,s);equity*=1+net
        peak=max(peak,equity);maxdd=max(maxdd,1-equity/peak);busy_until=b
        trades.append({"signal_cutoff":sample["feature_cutoff"],"entry_time":int(rows[a]["t"]),
            "exit_time":int(rows[b]["t"]),"gross_return":gross,"net_return":net,
            "predicted_gross_return":p,"sleeve_equity":equity,"net_pnl_initial_sleeve_units":initial*net})
        curve.append({"t":int(rows[b]["t"]),"equity":equity})
    positives=[t["net_pnl_initial_sleeve_units"] for t in trades if t["net_return"]>0]
    negatives=[t["net_pnl_initial_sleeve_units"] for t in trades if t["net_return"]<=0]
    abs_pnl=[abs(t["net_pnl_initial_sleeve_units"]) for t in trades]
    return {"opportunity_count":len(opportunities),"requested_entries":requested,"unavailable_entries":unavailable,
        "trade_count":len(trades),"net_return":equity-1,"max_drawdown_mark_to_hourly_close":maxdd,
        "win_rate":len(positives)/len(trades) if trades else None,
        "profit_factor":sum(positives)/-sum(negatives) if sum(negatives)<0 else None,
        "largest_absolute_trade_share":max(abs_pnl)/sum(abs_pnl) if sum(abs_pnl)>0 else None,
        "net_return_without_best_trade":float(np.prod([1+t["net_return"] for t in sorted(trades,key=lambda t:t["net_return"])[:-1]])-1) if trades else 0.0,
        "fees_per_side":f,"slippage_per_side":s,"delay_hours":delay_hours,"trades":trades,"curve":curve,
        "capital_assumption":"100% of an independent asset sleeve; aggregate allocation declared separately; not the actual capped paper portfolio"}

class QuantModel:
    def __init__(self,manifest,models,path): self.manifest=manifest;self.models=models;self.path=Path(path);self.model_hash=digest(manifest)

    @classmethod
    def load(cls,path):
        path=Path(path)
        if path.is_dir(): path=path/"model.json"
        manifest=json.loads(path.read_text())
        if manifest.get("schema_version")!=1 or manifest.get("feature_names")!=FEATURE_NAMES or manifest.get("quote_currency")!="TMN":
            raise ValueError("quant model schema mismatch")
        if manifest.get("horizon_hours")!=HORIZON or manifest.get("interval_seconds")!=INTERVAL: raise ValueError("quant target mismatch")
        models={}
        for symbol,meta in manifest["assets"].items():
            if not valid_symbol(symbol):raise ValueError("invalid registered TMN symbol")
            params=meta["ridge"]
            if any(len(params[k])!=len(FEATURE_NAMES) for k in ("means","scales","coefficients")):raise ValueError("ridge feature schema mismatch")
            if not np.isfinite([params[k] for k in ("means","scales","coefficients")]).all() or any(s<=0 for s in params["scales"]):raise ValueError("invalid ridge parameters")
            name=meta["tree_file"]
            if Path(name).name!=name: raise ValueError("tree path must be artifact-local")
            tree_path=path.parent/name
            if hashlib.sha256(tree_path.read_bytes()).hexdigest()!=meta["tree_sha256"]:raise ValueError("tree hash mismatch")
            tree=XGBRegressor(**TREE_SETTINGS); tree.load_model(str(tree_path));models[symbol]=tree
        return cls(manifest,models,path)

    def forecast(self,symbol,candles):
        if not valid_symbol(symbol) or symbol not in self.manifest["assets"]:raise ValueError("symbol/model mismatch; only declared TMN assets supported")
        rows=validate_rows(candles)
        x=feature_vector(rows); meta=self.manifest["assets"][symbol]
        feature_cutoff=int(rows[-1]["t"])+INTERVAL
        if feature_cutoff<=max(meta["training_cutoff"],meta["calibration_cutoff"]) or feature_cutoff<self.manifest.get("forecast_not_before",0):
            raise ValueError("forecast cutoff precedes model/calibration availability")
        values=predictions(meta["ridge"],self.models[symbol],[x])
        selected=meta.get("selected_candidate","ensemble")
        if selected not in CANDIDATES:raise ValueError("unknown selected candidate")
        gross=float(values[selected][0])
        # V2 trend is a calibrated forecast arm with a cost-aware technical eligibility gate.
        if selected=="trend" and not (x[3]>0 and x[7]>0):gross=min(gross,0.0)
        residual=meta["residuals"][selected]; fee=self.manifest["fee_rate_per_side"];slip=self.manifest["slippage_rate_per_side"]
        lower=gross+residual["q10"];upper=gross+residual["q90"]
        result={"symbol":symbol,"quote_currency":"TMN","predicted_gross_return":gross,
            "predicted_net_return":net_return(gross,fee,slip),"lower_bound":lower,"upper_bound":upper,
            "lower_bound_net_return":net_return(lower,fee,slip),
            "uncertainty":{"method":"empirical_10th_90th_calibration_residual_quantiles",
                "residual_std":residual["std"],"calibration_count":residual["count"],
                "coverage_guarantee":False,"warning":"overlapping dependent returns and regime shift; descriptive interval only"},
            "horizon_hours":HORIZON,"features":dict(zip(FEATURE_NAMES,x)),
            "technical_summary":{"return_24h":x[3],"return_48h":x[4],"volatility_24h":x[6],"rsi_14":(x[11]+.5)*100,"atr_14_fraction":x[10]},
            "components":{k:float(values[k][0]) for k in ("ridge","tree")},
            "model_hash":self.model_hash,"training_cutoff":meta["training_cutoff"],
            "calibration_cutoff":meta["calibration_cutoff"],"feature_cutoff":int(rows[-1]["t"])+INTERVAL,
            "candidate_status":"research_paper_only","selected_candidate":selected,
            "calibration_selection":meta.get("calibration_selection",{"mode":"fixed_ensemble"}),
            "assumed_fee_rate_per_side":fee,"assumed_slippage_rate_per_side":slip}
        if not math.isfinite(gross) or not math.isfinite(lower):raise ValueError("nonfinite model output")
        return result

def train_research(output_dir,config,datasets=None):
    """Fit frozen ensemble and evaluate once; calibration is for residuals only.

    config can be a dictionary or JSON path. datasets maps TMN symbol to dataset path/dict.
    All prices are TMN, no synthetic USDT conversion. Existing history is never mutated.
    """
    if not isinstance(config,dict):config=json.loads(Path(config).read_text())
    if config.get("interval_seconds",INTERVAL)!=INTERVAL or config.get("horizon_hours",HORIZON)!=HORIZON:raise ValueError("frozen hourly8h contract")
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[2]
    datasets=datasets or {symbol:project/"data/wallex_agent"/symbol/"history60d/dataset.json" for symbol in config["symbols"]}
    selection_mode=config.get("quant",{}).get("selection_mode","fixed_ensemble")
    if selection_mode not in ("fixed_ensemble","calibration_only"):raise ValueError("unknown selection mode")
    cost_aware_trend=selection_mode=="calibration_only"
    data={}; all_times=[]
    for symbol,source in datasets.items():
        d=source if isinstance(source,dict) else json.loads(Path(source).read_text())
        if not valid_symbol(symbol) or d["market"]["symbol"]!=symbol:raise ValueError("dataset TMN symbol mismatch")
        rows=validate_rows(d["candles"]);data[symbol]=(d,rows);all_times.extend(r["t"] for r in rows)
    timestamps=sorted(set(all_times));start=timestamps[0];end=timestamps[-1]+INTERVAL
    # Timestamp boundaries fixed by elapsed duration, shared across assets; no data-driven optimal split.
    train_end=(start+int((end-start)*.5))//INTERVAL*INTERVAL
    cal_end=(start+int((end-start)*.7))//INTERVAL*INTERVAL
    boundaries=[start,train_end,cal_end,end]
    protocol={"schema_version":1,"candidate_suite":list(CANDIDATES),"primary":("calibration_selected" if selection_mode=="calibration_only" else "ensemble"),
        "boundaries":boundaries,"horizon_hours":HORIZON,"warmup_hours":WARMUP,"ridge_alpha":10,
        "tree_settings":TREE_SETTINGS,"minimum_net_edge":float(config.get("minimum_net_edge",.002)),
        "fee_rate_per_side":float(config.get("fee_rate_per_side",.0035)),"slippage_rate_per_side":float(config.get("slippage_rate_per_side",.001)),
        "selection":("calibration-only >=3 trades, positive baseline and delay; max min(base,delay); cash otherwise" if selection_mode=="calibration_only" else "fixed ensemble; no evaluation tuning"),
        "selection_mode":selection_mode,"trend_cost_gate":cost_aware_trend,"dataset_hashes":{s:digest(d) for s,(d,r) in data.items()},
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"config_hash":digest(config)}
    # Persist the fully resolved protocol before fitting/predicting/evaluating any candidate.
    write_json(output/"protocol.json",protocol)
    manifest={"schema_version":1,"feature_names":FEATURE_NAMES,"quote_currency":"TMN","horizon_hours":HORIZON,
        "interval_seconds":INTERVAL,"primary_candidate":("calibration_selected" if selection_mode=="calibration_only" else "ensemble"),"protocol_hash":digest(protocol),
        "forecast_not_before":cal_end,
        "fee_rate_per_side":protocol["fee_rate_per_side"],"slippage_rate_per_side":protocol["slippage_rate_per_side"],"assets":{}}
    report={"schema_version":1,"protocol":protocol,"evaluation_status":"research_development_reused_period",
        "created_at":int(time.time()),"assets":{},"limitations":["Not an untouched test: final30d data were previously inspected",
        "Calibration and evaluation are short and autocorrelated; no statistical profitability guarantee",
        "TMN return includes changes in rial FX and crypto price; forecasts are not USDT forecasts",
        "Fee/slippage assumptions unverified; OHLC cannot establish fillability, spread, depth, queue or latency",
        f"Historical independent sleeves allocate1/{len(data)} initial capital perasset; actual paper risk caps20% perposition and40% total",
        "No historical news features or future backfilled news used; current news cannot validate old trades",
        "No live exchange order submitted; prospective frozen paper evaluation required"]}
    fee=protocol["fee_rate_per_side"];slip=protocol["slippage_rate_per_side"];margin=protocol["minimum_net_edge"]
    for symbol,(dataset,rows) in data.items():
        samples=make_samples(rows);split=split_samples(samples,boundaries)
        if len(split["calibration"])<40 or len(split["evaluation"])<40:raise ValueError("insufficient calibration/evaluation data")
        params,tree=fit_estimators(split["train"])
        tree_path=output/(symbol+".tree.json");tree.save_model(str(tree_path))
        cp=predictions(params,tree,[s["features"] for s in split["calibration"]]); ep=predictions(params,tree,[s["features"] for s in split["evaluation"]])
        ca=np.array([s["target"] for s in split["calibration"]]);ea=np.array([s["target"] for s in split["evaluation"]])
        residuals={name:{"q10":float(np.quantile(ca-p,.1)),"q90":float(np.quantile(ca-p,.9)),"std":float((ca-p).std()),"count":len(ca)} for name,p in cp.items()}
        manifest["assets"][symbol]={"ridge":params,"tree_file":tree_path.name,"tree_sha256":hashlib.sha256(tree_path.read_bytes()).hexdigest(),
            "training_cutoff":max(s["label_end"] for s in split["train"]),"calibration_cutoff":max(s["label_end"] for s in split["calibration"]),
            "residuals":residuals,"dataset_hash":digest(dataset)}
        selection={"mode":selection_mode,"selected_candidate":"ensemble","calibration_only":True}
        if selection_mode=="calibration_only":
            comparisons={}
            selection_samples=[s for s in split["calibration"] if s["label_end"]+INTERVAL<cal_end]
            selection_indices=[i for i,s in enumerate(split["calibration"]) if s["label_end"]+INTERVAL<cal_end]
            for name in ("ensemble","ridge","tree","trend"):
                if name=="trend":
                    # Same technical rule, now also apply the fixed forecast-cost gate for paper consistency.
                    cal_samples=[dict(s,trend_cost_gate=True) for s in selection_samples]
                else:cal_samples=selection_samples
                base=simulate_sleeve(rows,cal_samples,cp[name][selection_indices],name,fee,slip,margin,evaluation_end_exclusive=cal_end)
                delayed=simulate_sleeve(rows,cal_samples,cp[name][selection_indices],name,fee,slip,margin,delay_hours=1,evaluation_end_exclusive=cal_end)
                eligible=base["trade_count"]>=3 and delayed["trade_count"]>=3 and base["net_return"]>0 and delayed["net_return"]>0
                comparisons[name]={"trade_count":base["trade_count"],"net_return":base["net_return"],
                    "delay_trade_count":delayed["trade_count"],"delay_net_return":delayed["net_return"],
                    "eligible":eligible,"score":min(base["net_return"],delayed["net_return"])}
            eligible=[name for name,v in comparisons.items() if v["eligible"]]
            # Deterministic suite order breaks ties. No evaluation outcome is read here.
            chosen=max(eligible,key=lambda name:comparisons[name]["score"]) if eligible else "zero"
            selection.update(selected_candidate=chosen,minimum_trades=3,comparison=comparisons,
                stress_adjusted_selection_samples=len(selection_samples),stress_label_end_exclusive=cal_end,
                rule="positive base and delay with >=3 trades; max min(base,delay), deterministic suite-order ties; otherwise zero")
            manifest["assets"][symbol]["selected_candidate"]=chosen
            manifest["assets"][symbol]["calibration_selection"]=selection
        candidates={}
        for candidate in CANDIDATES:
            p=ep[candidate]
            eval_samples=([dict(s,trend_cost_gate=True) for s in split["evaluation"]] if candidate=="trend" and cost_aware_trend else split["evaluation"])
            candidates[candidate]={"forecast_metrics":metrics(ea,p),
                "base":simulate_sleeve(rows,eval_samples,p,candidate,fee,slip,margin),
                "double_cost":simulate_sleeve(rows,eval_samples,p,candidate,fee,slip,margin,cost_multiplier=2),
                "delay_1h":simulate_sleeve(rows,eval_samples,p,candidate,fee,slip,margin,delay_hours=1)}
            if candidate not in ("zero","trend"):
                candidates[candidate]["empirical_lower_bound_gate"]=simulate_sleeve(rows,eval_samples,p,candidate,fee,slip,margin,residual_low=residuals[candidate]["q10"])
        report["assets"][symbol]={"candle_count":len(rows),"gap_count":sum(b["t"]-a["t"]!=INTERVAL for a,b in zip(rows,rows[1:])),
            "samples":{k:len(v) for k,v in split.items()},"split_ranges":{k:{"first_feature_cutoff":v[0]["feature_cutoff"],"last_label_end":v[-1]["label_end"]} for k,v in split.items()},
            "residuals":residuals,"candidates":candidates,"calibration_selection":selection}
    report["equal_initial_capital_sleeves"]={}
    for candidate in CANDIDATES:
        report["equal_initial_capital_sleeves"][candidate]={}
        for stress in ("base","double_cost","delay_1h"):
            parts=[v["candidates"][candidate][stress] for v in report["assets"].values()]
            report["equal_initial_capital_sleeves"][candidate][stress]={"net_return":float(np.mean([r["net_return"] for r in parts])),
                "trade_count":sum(r["trade_count"] for r in parts),"capital_assumption":"equal initial capital, independent sleeves; not shared execution portfolio"}
    if selection_mode=="calibration_only":
        report["calibration_selected_suite"]={}
        for stress in ("base","double_cost","delay_1h"):
            parts=[asset["candidates"][asset["calibration_selection"]["selected_candidate"]][stress] for asset in report["assets"].values()]
            report["calibration_selected_suite"][stress]={"net_return":float(np.mean([part["net_return"] for part in parts])),
                "trade_count":sum(part["trade_count"] for part in parts),"sleeve_count":len(parts),
                "selected_nonzero_assets":sum(asset["calibration_selection"]["selected_candidate"]!="zero" for asset in report["assets"].values()),
                "capital_assumption":"equal initial capital across all registered research assets, including cash-selected sleeves; not actual20%/40% capped portfolio"}
    write_json(output/"model.json",manifest);report["model_hash"]=digest(manifest)
    write_json(output/"report.json",report)
    return {"model_path":str(output/"model.json"),"manifest_path":str(output/"model.json"),"report_path":str(output/"report.json"),
        "protocol_path":str(output/"protocol.json"),"model_hash":digest(manifest),"candidate_status":"research_paper_only"}


def download_research_history(symbols,output_dir,days=60,max_workers=4):
    """Bounded public Wallex-only history fetch; no account writes. Raw pages stay beside datasets.

    Symbol list must already be chosen from a volume/spread ranking independent of returns.
    Existing datasets are reused, and individual unavailable assets are recorded without backfilling.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from experiments.wallex_challenge.api import PublicAPI
    symbols=list(symbols)
    if len(set(symbols))!=len(symbols) or not symbols or any(not valid_symbol(s) for s in symbols):raise ValueError("invalid/duplicate research universe")
    if not 1<=max_workers<=4 or not 1<=days<=60:raise ValueError("bounded history limits")
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    api=PublicAPI();response=api.get("/v1/markets")
    if not response.get("success"):raise ValueError("market metadata unavailable")
    markets=response["result"]["symbols"]
    def fetch(symbol):
        dest=root/symbol/("history%dd"%days);file=dest/"dataset.json"
        if file.exists():
            d=json.loads(file.read_text())
            if d["market"]["symbol"]!=symbol:raise ValueError("cached market mismatch")
            validate_rows(d["candles"])
            return {"status":"cached","dataset_path":str(file),"candles":len(d["candles"])}
        info=markets[symbol]
        _,quality=api.history(symbol,days,INTERVAL,dest)
        candles=[]
        for page in sorted(dest.glob("raw-history-*.json")):
            part=json.loads(page.read_text())["response"]
            if part.get("s")!="ok":raise ValueError("history page unavailable")
            for values in zip(*(part[k] for k in ("t","o","h","l","c","v"))):
                candles.append(dict(zip(("t","o","h","l","c","v"),values)))
        if quality["source_interval_seconds"]!=INTERVAL:raise ValueError("source did not return hourly bars")
        rows=validate_rows(candles)
        quality["all_hourly_candles_preserved"]=len(rows)
        quality["gaps"]=[{"after":int(a["t"]),"before":int(b["t"]),"missing_hours":int((b["t"]-a["t"])/INTERVAL)-1}
            for a,b in zip(rows,rows[1:]) if b["t"]-a["t"]!=INTERVAL]
        market={"symbol":symbol,"amount_decimals":int(info["stepSize"]),"minimum_quantity":str(info["minQty"]),"minimum_notional":str(info["minNotional"])}
        write_json(file,{"candles":rows,"market":market,"quality":quality})
        return {"status":"downloaded","dataset_path":str(file),"candles":len(rows),"gaps":len(quality["gaps"])}
    results={}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures={pool.submit(fetch,symbol):symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol=futures[future]
            try:results[symbol]=future.result()
            except Exception as error:results[symbol]={"status":"unavailable","error_type":type(error).__name__,"detail":str(error)[:400]}
    write_json(root/"download-manifest.json",{"requested_symbols":symbols,"requested_days":days,"max_workers":max_workers,"results":results})
    return results


def simulate_shared_research(model_path,report_path,datasets,output_path,initial_tmn=5000000.0,max_position_fraction=.2,max_positions=2):
    """Separate QUANT-only shared-capital replay, with frozen entry ranking and stresses.

    No historical LLM/news, exchange minimum-size, depth or credited-market claim is made.
    Assets qualify only from the calibration selector; current liquidity universe remains biased.
    """
    if not 0<max_position_fraction<=.2 or not 1<=max_positions<=2 or max_position_fraction*max_positions>.4+1e-12:raise ValueError("shared exposure caps")
    model=QuantModel.load(model_path);report=json.loads(Path(report_path).read_text())
    if report.get("model_hash")!=model.model_hash:raise ValueError("shared report/model hash mismatch")
    if digest(report["protocol"])!=model.manifest["protocol_hash"]:raise ValueError("shared protocol hash mismatch")
    if set(datasets)!=set(model.manifest["assets"]):raise ValueError("shared dataset universe mismatch")
    fee=model.manifest["fee_rate_per_side"];slip=model.manifest["slippage_rate_per_side"]
    margin=report["protocol"]["minimum_net_edge"];groups={};rows_by_symbol={}
    for symbol,source in datasets.items():
        d=source if isinstance(source,dict) else json.loads(Path(source).read_text())
        if d["market"]["symbol"]!=symbol:raise ValueError("shared dataset symbol mismatch")
        expected=model.manifest["assets"][symbol]["dataset_hash"]
        if digest(d)!=expected or report["protocol"]["dataset_hashes"].get(symbol)!=expected:raise ValueError("shared dataset hash mismatch")
        rows=validate_rows(d["candles"]);rows_by_symbol[symbol]=rows
        samples=split_samples(make_samples(rows),report["protocol"]["boundaries"])["evaluation"]
        if not samples:continue
        chosen=model.manifest["assets"][symbol].get("selected_candidate","ensemble")
        values=predictions(model.manifest["assets"][symbol]["ridge"],model.models[symbol],[s["features"] for s in samples])[chosen]
        for sample,predicted in zip(samples,values):
            if not sample["opportunity"]:continue
            s=dict(sample,trend_cost_gate=True)
            if entry_decision(chosen,s,float(predicted),fee,slip,margin):
                groups.setdefault(s["entry_time"],[]).append({"symbol":symbol,"index":s["index"],"candidate":chosen,
                    "predicted_gross_return":float(predicted),"predicted_net_return":net_return(float(predicted),fee,slip)})
    decisions=[]
    for timestamp,candidates in sorted(groups.items()):
        chosen=sorted(candidates,key=lambda d:(-d["predicted_net_return"],d["symbol"]))[:max_positions]
        decisions.append({"entry_time":timestamp,"candidate_count":len(candidates),"selected":chosen})
    result={"schema_version":1,"model_hash":model.model_hash,"policy":{
        "initial_tmn":initial_tmn,"max_position_fraction":max_position_fraction,"max_positions":max_positions,
        "max_total_fraction":max_position_fraction*max_positions,"selection":"calibration-qualified strategy perasset, then highest net forecast among current signals; symbol breaks ties",
        "holding_hours":HORIZON,"rebalance":"exit fixed horizon before next8h entry; budget each entry at20% of pre-entry shared equity",
        "stress":"same frozen selected entries; doubled fees/slippage or1h delay; unavailable delayed fills stay cash",
        "declared_before_v2_outcome_inspection":True,"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "limitations":["Quant-only replay; historical LLM/news decisions are absent", "Current-volume/current-exclusion universe has survivorship/selection bias",
            "OHLC cannot establish fills, queue priority, live spreads/depth, actual eligibility or credited-market approval",
            "No intrahorizon stops modeled; actual paper risk engine may act differently", "Development prices, not independent evidence of live profitability"],
        "decisions":decisions,"scenarios":{}}
    for name,multiplier,delay in (("base",1,0),("double_cost",2,0),("delay_1h",1,1)):
        f=fee*multiplier;s=slip*multiplier;equity=float(initial_tmn);peak=equity;maxdd=0;trades=[];curve=[];blocked=[]
        for decision in decisions:
            opening=equity;filled=[]
            for entry in decision["selected"]:
                rows=rows_by_symbol[entry["symbol"]];a=entry["index"]+1+delay;b=a+HORIZON
                if b>=len(rows) or any(rows[j+1]["t"]-rows[j]["t"]!=INTERVAL for j in range(a,b)):
                    blocked.append({"symbol":entry["symbol"],"signal_entry_time":decision["entry_time"],"reason":"delayed_fill_or_exit_crosses_gap_or_dataset_end"});continue
                gross=rows[b]["o"]/rows[a]["o"]-1;net=net_return(gross,f,s)
                filled.append((entry,rows,a,b,net))
            cash_fraction=1-max_position_fraction*len(filled)
            for h in range(HORIZON):
                mark=cash_fraction
                for entry,rows,a,b,net in filled:
                    mark+=max_position_fraction*rows[a+h]["c"]/rows[a]["o"]*(1-s)/((1+s)*(1+f))
                current=opening*mark;peak=max(peak,current);maxdd=max(maxdd,1-current/peak)
            equity=opening*(1+sum(max_position_fraction*net for entry,rows,a,b,net in filled))
            for entry,rows,a,b,net in filled:
                budget=opening*max_position_fraction
                entry_price=rows[a]["o"]*(1+s);exit_price=rows[b]["o"]*(1-s)
                quantity=budget/(entry_price*(1+f));buy_fee=quantity*entry_price*f;sell_fee=quantity*exit_price*f
                trades.append(dict(entry,entry_time=int(rows[a]["t"]),exit_time=int(rows[b]["t"]),net_return=float(net),
                    budget_tmn=budget,net_pnl_tmn=budget*net,quantity=quantity,
                    entry_price_tmn=entry_price,exit_price_tmn=exit_price,buy_fee_tmn=buy_fee,sell_fee_tmn=sell_fee))
            peak=max(peak,equity);maxdd=max(maxdd,1-equity/peak)
            curve.append({"t":decision["entry_time"]+(delay+HORIZON)*INTERVAL,"equity_tmn":equity})
        result["scenarios"][name]={"final_equity_tmn":equity,"net_return":equity/initial_tmn-1,"trade_count":len(trades),
            "blocked_count":len(blocked),"max_drawdown_mark_to_hourly_close":maxdd,"trades":trades,"blocked":blocked,"curve":curve,
            "fee_rate_per_side":f,"slippage_rate_per_side":s,"delay_hours":delay,
            "fees_tmn":sum(t["buy_fee_tmn"]+t["sell_fee_tmn"] for t in trades),"settled_to_tmn":True,"remaining_positions":[],
            "cash_tmn":equity,"maximum_configured_exposure_fraction":max_position_fraction*max_positions}
    write_json(output_path,result)
    return result
