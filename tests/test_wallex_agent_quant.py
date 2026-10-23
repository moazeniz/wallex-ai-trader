"""Behavioral checks for time causality, quote schema and conservative costs."""
import copy
import hashlib
import json
from pathlib import Path
import math
import numpy as np
import pytest
from experiments.wallex_agent import quant as q


def candles(n=1000,offset=0):
    rows=[]
    for i in range(n):
        price=1000*math.exp(.0001*i+.015*math.sin(i/14)+.006*math.sin(i/3.1))
        close=price*(1+.001*math.sin(i))
        rows.append({"t":(480000+i+offset)*3600,"o":price,"h":max(price,close)*1.002,
            "l":min(price,close)*.998,"c":close,"v":10+3*math.cos(i/6)})
    return rows


def config():
    return {"symbols":["BTCTMN","ETHTMN"],"interval_seconds":3600,"horizon_hours":8,
        "fee_rate_per_side":.0035,"slippage_rate_per_side":.001,"minimum_net_edge":.002}


def dataset(rows,symbol): return {"market":{"symbol":symbol},"candles":rows,"quality":{"test":True}}

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root=tmp_path_factory.mktemp("quant")
    rows=candles(); data={s:dataset(copy.deepcopy(rows),s) for s in config()["symbols"]}
    paths=q.train_research(root,config(),data)
    return root,rows,data,paths


def test_closed_features_do_not_read_future():
    rows=candles(140); before=q.feature_vector(rows,90)
    future=copy.deepcopy(rows)
    for r in future[91:]:
        for k in ["o","h","l","c"]:r[k]*=100
        r["v"]*=1000
    assert q.feature_vector(future,90)==before
    assert q.feature_vector(rows[:91])==before


def test_feature_gap_requires_new_warmup():
    rows=candles(150); del rows[65]
    with pytest.raises(ValueError,match="gap"):q.feature_vector(rows,90)
    assert np.isfinite(q.feature_vector(rows,120)).all()
    samples=q.make_samples(rows)
    gap_after=rows[64]["t"]
    assert all(s["label_end"]<=gap_after or s["feature_cutoff"]>=rows[65]["t"]+49*3600 for s in samples)


def test_label_is_next_open_eight_hours_later():
    rows=candles(100);s=q.make_samples(rows)[0];i=s["index"]
    assert s["target"]==pytest.approx(rows[i+9]["o"]/rows[i+1]["o"]-1)
    assert s["feature_cutoff"]==rows[i]["t"]+3600
    assert s["entry_time"]==s["feature_cutoff"]


def test_split_purges_maturing_labels_across_each_boundary():
    rows=candles(300);times=[r["t"] for r in rows]
    bounds=[times[0],times[150],times[220],times[-1]+3600]
    split=q.split_samples(q.make_samples(rows),bounds)
    for name,lo,hi in zip(("train","calibration","evaluation"),bounds[:-1],bounds[1:]):
        assert split[name]
        assert all(lo<=s["feature_cutoff"]<s["label_end"]<hi for s in split[name])
    assert max(s["label_end"] for s in split["train"])<min(s["feature_cutoff"] for s in split["calibration"])
    assert max(s["label_end"] for s in split["calibration"])<min(s["feature_cutoff"] for s in split["evaluation"])


def test_gap_label_never_bridges_missing_hour():
    rows=candles(150);del rows[80]
    samples=q.make_samples(rows)
    assert all(not(rows[79]["t"]<s["label_end"] and s["entry_time"]<=rows[79]["t"]) for s in samples)


def test_costs_are_exact_two_sided_and_cannot_improve_return():
    expected=(1-.001)*(1-.0035)/((1+.001)*(1+.0035))-1
    assert q.net_return(0,.0035,.001)==pytest.approx(expected)
    assert q.net_return(.02,.007,.002)<q.net_return(.02,.0035,.001)<.02
    for bad in [-.1,float("nan"),1.0]:
        with pytest.raises(ValueError):q.cost_factor(bad,.001)


def test_expensive_flat_forecast_is_rejected_and_zero_never_trades():
    sample={"features":[0.0]*len(q.FEATURE_NAMES)}
    assert not q.entry_decision("ensemble",sample,.008,.0035,.001,.002)
    assert not q.entry_decision("zero",sample,1,.0035,.001,.002)
    assert q.entry_decision("ensemble",sample,.02,.0035,.001,.002)
    assert not q.entry_decision("ensemble",sample,.02,.0035,.001,.002,residual_low=-.03)


def test_forecast_roundtrip_is_finite_and_names_quote_units(trained):
    root,rows,data,paths=trained
    a=q.QuantModel.load(paths["model_path"]);b=q.QuantModel.load(root)
    one=a.forecast("BTCTMN",rows);two=b.forecast("BTCTMN",rows)
    assert one==two
    assert one["quote_currency"]=="TMN" and one["horizon_hours"]==8
    assert len(one["model_hash"])==64
    assert one["model_hash"]==paths["model_hash"]
    assert one["candidate_status"]=="research_paper_only"
    assert one["uncertainty"]["coverage_guarantee"] is False
    assert all(math.isfinite(one[k]) for k in ("predicted_gross_return","predicted_net_return","lower_bound"))
    assert one["training_cutoff"]<one["calibration_cutoff"]<one["feature_cutoff"]
    with pytest.raises(ValueError,match="symbol/model"):a.forecast("BTCUSDT",rows)


def test_tampered_native_model_hash_is_rejected(trained,tmp_path):
    root,rows,data,paths=trained
    manifest=json.loads(Path(paths["model_path"]).read_text());meta=manifest["assets"]["BTCTMN"]
    for s,m in manifest["assets"].items():(tmp_path/m["tree_file"]).write_bytes((root/m["tree_file"]).read_bytes())
    (tmp_path/meta["tree_file"]).write_bytes(b"tampered")
    (tmp_path/"model.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="hash"):q.QuantModel.load(tmp_path)


def test_schema_and_dataset_symbol_rejection(trained,tmp_path):
    root,rows,data,paths=trained
    manifest=json.loads(Path(paths["model_path"]).read_text());manifest["quote_currency"]="USDT"
    (tmp_path/"model.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="schema"):q.QuantModel.load(tmp_path)
    bad=copy.deepcopy(data);bad["BTCTMN"]["market"]["symbol"]="BTCUSDT"
    with pytest.raises(ValueError,match="symbol mismatch"):q.train_research(tmp_path/"bad",config(),bad)


def test_evaluation_prices_cannot_change_fit_or_calibration(trained,tmp_path):
    root,rows,data,paths=trained
    original=json.loads(Path(paths["model_path"]).read_text())
    cutoff=json.loads((root/"protocol.json").read_text())["boundaries"][2]
    altered=copy.deepcopy(data)
    for d in altered.values():
        for r in d["candles"]:
            if r["t"]>=cutoff:
                factor=1+.5*math.sin(r["t"]/3600)
                for k in ["o","h","l","c"]:r[k]*=factor
    changed=q.train_research(tmp_path/"perturbed",config(),altered)
    after=json.loads(Path(changed["model_path"]).read_text())
    for symbol in original["assets"]:
        a,b=original["assets"][symbol],after["assets"][symbol]
        assert a["ridge"]==b["ridge"] and a["tree_sha256"]==b["tree_sha256"]
        assert a["residuals"]==b["residuals"]


def test_candidates_share_opportunities_and_stress_uses_same_entries(trained):
    root,rows,data,paths=trained
    report=json.loads(Path(paths["report_path"]).read_text())
    for asset in report["assets"].values():
        candidates=asset["candidates"]
        assert len(set(c["base"]["opportunity_count"] for c in candidates.values()))==1
        assert candidates["zero"]["base"]["net_return"]==0
        for c in candidates.values():
            assert c["base"]["requested_entries"]==c["double_cost"]["requested_entries"]==c["delay_1h"]["requested_entries"]
            assert c["base"]["trade_count"]==c["double_cost"]["trade_count"]
            assert c["double_cost"]["net_return"]<=c["base"]["net_return"]+1e-12
            assert c["base"]["fees_per_side"]==config()["fee_rate_per_side"]
            assert c["base"]["slippage_per_side"]==config()["slippage_rate_per_side"]


def test_frozen_protocol_no_selection_and_unchanged_model_family(trained):
    root,rows,data,paths=trained
    protocol=json.loads((root/"protocol.json").read_text())
    assert protocol["primary"]=="ensemble" and protocol["ridge_alpha"]==10
    assert protocol["candidate_suite"]==list(q.CANDIDATES)
    assert protocol["selection"]=="fixed ensemble; no evaluation tuning"
    assert protocol["tree_settings"]==q.TREE_SETTINGS
    assert protocol["code_sha256"]==hashlib.sha256(Path(q.__file__).read_bytes()).hexdigest()


def test_invalid_candles_fail_closed():
    rows=candles(100);rows[1]["v"]=-1
    with pytest.raises(ValueError,match="volume"):q.validate_rows(rows)
    rows=candles(100);rows[1]["t"]=rows[0]["t"]
    with pytest.raises(ValueError,match="duplicate"):q.validate_rows(rows)


def test_public_history_numeric_strings_are_normalized():
    rows=candles(100)
    raw=[{k:(str(v) if k!="t" else v) for k,v in r.items()} for r in rows]
    actual=q.make_samples(raw);expected=q.make_samples(rows)
    assert actual==expected


def test_delayed_equity_curve_does_not_stamp_future_cash_before_exit():
    rows=candles(140); samples=[s for s in q.make_samples(rows) if s["opportunity"]]
    predicted=[.02]+[-.02]*(len(samples)-1)
    result=q.simulate_sleeve(rows,samples,predicted,"ensemble",.0035,.001,.002,delay_hours=1)
    assert result["trade_count"]==1
    times=[r["t"] for r in result["curve"]]
    assert times==sorted(times)
    assert result["curve"][0]["t"]==result["trades"][0]["exit_time"]


@pytest.mark.parametrize("symbol",["DOGETMN","SOLTMN","1INCHTMN","1000PEPETMN"])
def test_registered_research_tmn_symbols_are_generic(symbol):
    assert q.valid_symbol(symbol)

@pytest.mark.parametrize("symbol",["BTCUSDT","dogeTMN","DOGE/TMN","DOGE-TMN","../BTCTMN","TMN",""])
def test_non_tmn_or_ambiguous_symbols_rejected(symbol):
    assert not q.valid_symbol(symbol)


def test_v2_selection_uses_calibration_only_and_preserves_v1_artifact(trained,tmp_path):
    root,rows,data,paths=trained
    before={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir() if f.is_file()}
    c=config();c["symbols"]=["DOGETMN"];c["quant"]={"selection_mode":"calibration_only"}
    d={"DOGETMN":dataset(copy.deepcopy(rows),"DOGETMN")}
    first=q.train_research(tmp_path/"universe-v2",c,d)
    first_model=json.loads(Path(first["model_path"]).read_text())
    first_report=json.loads(Path(first["report_path"]).read_text())
    cut=first_report["protocol"]["boundaries"][2]
    altered=copy.deepcopy(d)
    for r in altered["DOGETMN"]["candles"]:
        if r["t"]>=cut:
            scale=1+.35*math.sin(r["t"]/3600)
            for k in ["o","h","l","c"]:r[k]*=scale
    second=q.train_research(tmp_path/"universe-v2-perturbed",c,altered)
    second_model=json.loads(Path(second["model_path"]).read_text())
    a,b=first_model["assets"]["DOGETMN"],second_model["assets"]["DOGETMN"]
    assert a["calibration_selection"]==b["calibration_selection"]
    assert a["selected_candidate"]==b["selected_candidate"]
    selection=a["calibration_selection"];eligible=[k for k,v in selection["comparison"].items() if v["eligible"]]
    if eligible:
        chosen=max(eligible,key=lambda name:selection["comparison"][name]["score"])
        assert a["selected_candidate"]==chosen
        assert selection["comparison"][chosen]["trade_count"]>=3
        assert selection["comparison"][chosen]["net_return"]>0 and selection["comparison"][chosen]["delay_net_return"]>0
    else:assert a["selected_candidate"]=="zero"
    forecast=q.QuantModel.load(first["model_path"]).forecast("DOGETMN",rows)
    assert forecast["selected_candidate"]==a["selected_candidate"]
    if a["selected_candidate"]=="zero":assert forecast["predicted_gross_return"]==0
    assert first_report["calibration_selected_suite"]["base"]["net_return"]==pytest.approx(first_report["assets"]["DOGETMN"]["candidates"][a["selected_candidate"]]["base"]["net_return"])
    assert before=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir() if f.is_file()}


def test_v2_trend_paper_and_research_share_cost_gate():
    sample={"features":[0.0]*len(q.FEATURE_NAMES),"trend_cost_gate":True}
    sample["features"][3]=.03;sample["features"][7]=.01
    assert not q.entry_decision("trend",sample,.005,.0035,.001,.002)
    assert q.entry_decision("trend",sample,.03,.0035,.001,.002)


def test_shared_replay_has_cash_conservation_caps_settlement_and_fixed_stresses(trained,tmp_path,monkeypatch):
    root,rows,data,paths=trained
    def positive_forecasts(params,tree,x):
        return {name:np.full(len(x),0.02 if name!="zero" else 0.0) for name in q.CANDIDATES}
    monkeypatch.setattr(q,"predictions",positive_forecasts)
    result=q.simulate_shared_research(paths["model_path"],paths["report_path"],data,tmp_path/"shared.json")
    base=result["scenarios"]["base"];double=result["scenarios"]["double_cost"];delay=result["scenarios"]["delay_1h"]
    assert base["trade_count"]>0 and base["fees_tmn"]>0
    assert all(len(d["selected"])<=2 for d in result["decisions"])
    assert base["settled_to_tmn"] and base["remaining_positions"]==[]
    assert base["cash_tmn"]==base["final_equity_tmn"]
    f=base["fee_rate_per_side"]
    cash=5000000+sum(t["quantity"]*(t["exit_price_tmn"]*(1-f)-t["entry_price_tmn"]*(1+f)) for t in base["trades"])
    assert base["final_equity_tmn"]==pytest.approx(cash)
    assert base["fees_tmn"]==pytest.approx(sum(t["buy_fee_tmn"]+t["sell_fee_tmn"] for t in base["trades"]))
    for decision in result["decisions"]:
        matches=[t for t in base["trades"] if t["entry_time"]==decision["entry_time"]]
        if matches:
            assert sum(t["budget_tmn"] for t in matches)/(matches[0]["budget_tmn"]/.2)<=.4+1e-12
    assert {(t["symbol"],t["entry_time"]) for t in base["trades"]}=={(t["symbol"],t["entry_time"]) for t in double["trades"]}
    assert double["net_return"]<=base["net_return"]
    assert all(t["exit_time"]-t["entry_time"]==8*3600 for t in base["trades"])
    assert {(t["symbol"],t["entry_time"]+3600) for t in base["trades"]}=={(t["symbol"],t["entry_time"]) for t in delay["trades"]}
    with pytest.raises(ValueError,match="caps"):
        q.simulate_shared_research(paths["model_path"],paths["report_path"],data,tmp_path/"bad.json",max_position_fraction=.3)
