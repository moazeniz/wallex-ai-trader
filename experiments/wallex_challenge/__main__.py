from dataclasses import asdict
from pathlib import Path
import argparse, fcntl, hashlib, json, math, os, sys, time
from .core import Settings, Market, Candle, Audit, atomic_json, digest, validate_candles, fit_model, features, predict, decision, simulate, verify_audit
from .api import PublicAPI
from .policy import ChallengePolicy

ROOT=Path(__file__).resolve().parents[2]

def source_hash():
    return digest({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"experiments/wallex_challenge").glob("*.py"))})

def load_settings(path):
    return Settings(**(json.loads(Path(path).read_text()) if path else {}))

def fetch(args):
    root=Path(args.output)
    if root.exists() and any(root.iterdir()):raise ValueError("Output directory must be empty to preserve provenance")
    api=PublicAPI();settings=load_settings(args.settings)
    market,info=api.market(args.symbol)
    rows,quality=api.history(args.symbol,args.days,settings.interval_seconds,root)
    atomic_json(root/"dataset.json",{"candles":[asdict(b) for b in rows],"market":asdict(market),"quality":quality})
    atomic_json(root/"market-observed.json",info)
    print(json.dumps(quality,indent=2))

def backtest(args):
    dataset=json.loads(Path(args.dataset).read_text());rows=[Candle(**b) for b in dataset["candles"]]
    market=Market(**dataset["market"]);settings=load_settings(args.settings);validate_candles(rows,settings.interval_seconds)
    if len(rows)<300:raise ValueError("Need at least 300 contiguous complete candles")
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):raise ValueError("Run output must be empty; use a new run directory")
    output.mkdir(parents=True,exist_ok=True)
    split=int(len(rows)*0.7);evaluation_start=split+1
    model=fit_model(rows,split,settings)
    report,events,curve=simulate(rows,model,market,settings,evaluation_start)
    valid_indices=range(evaluation_start-1,len(rows)-settings.horizon-1)
    forecasts=[(predict(model,features(rows,i)),int(rows[i+1+settings.horizon].o>rows[i+1].o)) for i in valid_indices]
    loss=-sum(y*math.log(max(p,1e-12))+(1-y)*math.log(max(1-p,1e-12)) for p,y in forecasts)/len(forecasts)
    base=sum(int(rows[i+1+settings.horizon].o>rows[i+1].o) for i in range(24,split-settings.horizon-1))/(split-settings.horizon-25)
    baseline=-sum(y*math.log(max(base,1e-12))+(1-y)*math.log(max(1-base,1e-12)) for p,y in forecasts)/len(forecasts)
    report.update({"prediction_log_loss":loss,"constant_probability_log_loss":baseline,"forecast_examples":len(forecasts),
        "model_sha256":digest(model),"settings":asdict(settings),"market":asdict(market),"dataset_sha256":digest(dataset),
        "algorithm_sha256":source_hash(),"data_quality":dataset["quality"],"training_bars":split,"evaluation_start_index":evaluation_start})
    audit=Audit(output/"decisions.jsonl")
    audit.append({"kind":"run_manifest","algorithm_sha256":source_hash(),"dataset_sha256":digest(dataset),"model_sha256":digest(model),"settings":asdict(settings)})
    for event in events:audit.append(event)
    report["audit_head_sha256"]=audit.previous;report["decision_count"]=len(events)
    atomic_json(output/"model.json",model);atomic_json(output/"report.json",report);atomic_json(output/"equity.json",curve)
    # Keep an exact input copy on the server for replay.
    atomic_json(output/"dataset.json",dataset)
    print(json.dumps(report,indent=2))

def replay(args):
    root=Path(args.run);report=json.loads((root/"report.json").read_text());dataset=json.loads((root/"dataset.json").read_text());model=json.loads((root/"model.json").read_text())
    if source_hash()!=report["algorithm_sha256"] or digest(dataset)!=report["dataset_sha256"] or digest(model)!=report["model_sha256"]:raise ValueError("Run input or algorithm hash differs")
    rows=[Candle(**b) for b in dataset["candles"]];settings=Settings(**report["settings"]);market=Market(**report["market"])
    rerun,events,curve=simulate(rows,model,market,settings,report["evaluation_start_index"])
    stored=list(verify_audit(root/"decisions.jsonl"))
    if stored[-1]["hash"]!=report["audit_head_sha256"]:raise ValueError("Audit head differs from manifest")
    if [e["payload"] for e in stored[1:]]!=events:raise ValueError("Replayed decisions differ")
    for key,value in rerun.items():
        if report[key]!=value:raise ValueError("Replayed report differs: "+key)
    if json.loads((root/"equity.json").read_text())!=curve:raise ValueError("Replayed equity differs")
    print(json.dumps({"replay":"identical","decisions":len(events),"audit_head_sha256":stored[-1]["hash"]},indent=2))

def shadow(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    lock=(root/"collector.lock").open("w")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError("Another collector owns this output directory")
    model=json.loads(Path(args.model).read_text());settings=load_settings(args.settings)
    report=json.loads(Path(args.model).with_name("report.json").read_text())
    if args.symbol!=report["market"]["symbol"]:raise ValueError("Model symbol does not match")
    if digest(model)!=report["model_sha256"] or source_hash()!=report["algorithm_sha256"]:raise ValueError("Shadow model or source hash differs from backtest")
    if asdict(settings)!=report["settings"]:raise ValueError("Shadow settings differ from backtest")
    policy=ChallengePolicy(**(json.loads(Path(args.policy).read_text()) if args.policy else {}))
    audit=Audit(root/"observations.jsonl");last_signal=-1
    # Derive checkpoint from verified audit, preventing duplicate signals after a crash.
    for e in verify_audit(audit.path) if audit.path.exists() else []:
        if e["payload"].get("kind")=="shadow_decision":last_signal=max(last_signal,e["payload"]["signal_bar"])
    api=PublicAPI()
    while True:
        try:
            rows,quality=api.history(args.symbol,2,settings.interval_seconds)
            validate_candles(rows,settings.interval_seconds)
            if len(rows)<25:raise ValueError("Insufficient contiguous recent data")
            if rows[-1].t+settings.interval_seconds<int(time.time())-settings.interval_seconds:raise ValueError("Latest completed bar is stale")
            if rows[-1].t>last_signal:
                if model["last_label_time"]>=rows[-1].t:raise ValueError("Model training time reaches shadow signal")
                x=features(rows,len(rows)-1);p=predict(model,x);action,why=decision(model,p,settings);book=api.book(args.symbol)
                payload={"kind":"shadow_decision","mode":"observe_only","symbol":args.symbol,"signal_bar":rows[-1].t,
                    "observed_at":int(time.time()),"feature_values":x,"probability_up":p,"proposed_action":action,"explanation":why,
                    "book":book,"quality":quality,"algorithm_sha256":source_hash(),"model_sha256":digest(model),
                    "execution":"disabled","policy_gate":policy.gate(args.symbol,int(time.time())),"blocked_by":policy.gate(args.symbol,int(time.time()))["reasons"]+["live_adapter_not_implemented"]}
                audit.append(payload);last_signal=rows[-1].t
                atomic_json(root/"status.json",{"status":"observing","last_signal":last_signal,"event_hash":audit.previous,"mode":"observe_only","symbol":args.symbol})
                print(json.dumps(payload),flush=True)
            if args.once:return
        except Exception as exc:
            audit.append({"kind":"collector_error","observed_at":int(time.time()),"error_type":type(exc).__name__,"message":str(exc)[:500]})
            atomic_json(root/"status.json",{"status":"error","error_type":type(exc).__name__,"message":str(exc)[:500],"mode":"observe_only"})
            if args.once:raise
        time.sleep(max(30,args.poll_seconds))

def main():
    parser=argparse.ArgumentParser(description="Wallex research only: public data, historical simulations, read-only shadow logging")
    sub=parser.add_subparsers(dest="command",required=True)
    f=sub.add_parser("fetch");f.add_argument("--symbol",default="BTCTMN");f.add_argument("--days",type=int,default=30);f.add_argument("--settings");f.add_argument("--output",required=True);f.set_defaults(func=fetch)
    b=sub.add_parser("backtest");b.add_argument("--dataset",required=True);b.add_argument("--settings");b.add_argument("--output",required=True);b.set_defaults(func=backtest)
    r=sub.add_parser("replay");r.add_argument("--run",required=True);r.set_defaults(func=replay)
    s=sub.add_parser("shadow");s.add_argument("--symbol",default="BTCTMN");s.add_argument("--model",required=True);s.add_argument("--policy");s.add_argument("--settings");s.add_argument("--output",required=True);s.add_argument("--once",action="store_true");s.add_argument("--poll-seconds",type=int,default=300);s.set_defaults(func=shadow)
    args=parser.parse_args()
    try:args.func(args)
    except Exception as exc:print(type(exc).__name__+": "+str(exc),file=sys.stderr);sys.exit(1)

if __name__=="__main__":main()
