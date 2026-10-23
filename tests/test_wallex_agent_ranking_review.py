"""Independent regression for capacity-aware opportunity ranking after risk exits."""
from decimal import Decimal
import pytest
from experiments.wallex_agent import runner as runner_module
from experiments.wallex_agent.risk import entry_plan as actual_entry_plan
# The repository uses pytest importlib mode; load the existing offline helper module by path.
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys
_spec=spec_from_file_location("wallex_ranking_integration_helpers",Path(__file__).with_name("test_wallex_agent_integration_review.py"))
_helpers=module_from_spec(_spec);sys.modules[_spec.name]=_helpers;_spec.loader.exec_module(_helpers)
NOW=_helpers.NOW
build_runner=_helpers.build_runner
forecast=_helpers.forecast
market_context=_helpers.market_context
seed_long=_helpers.seed_long


def test_forced_exits_release_capacity_before_ranking_new_profit_opportunities(build_runner,monkeypatch):
    symbols=("SOLTMN","XRPTMN","DOGETMN","SUITMN")
    contexts={s:market_context(s,bid=98000 if s in ("SOLTMN","XRPTMN") else 100000) for s in symbols}
    # DOGE has the higher return forecast, but a wider stop makes SUI the larger TMN-profit opportunity.
    contexts["DOGETMN"]["technical"]["one_hour"]["atr_fraction"]=.02
    contexts["SUITMN"]["technical"]["one_hour"]["atr_fraction"]=.005
    predictions={"SOLTMN":forecast(0,selected="zero"),"XRPTMN":forecast(0,selected="zero"),
        "DOGETMN":forecast(.05),"SUITMN":forecast(.045)}
    agent=build_runner(symbols=symbols,contexts=contexts,predictions=predictions,action="BUY",research_limit=1)
    seed_long(agent,"SOLTMN");seed_long(agent,"XRPTMN")
    initial=agent.portfolio.snapshot()
    exposure=sum(Decimal(p["quantity"])*Decimal(p["mark_price"]) for p in initial["positions"].values())
    assert len(initial["positions"])==2
    assert exposure/Decimal(initial["equity_tmn"])>.39
    capacity_observations=[]
    def capacity_plan(context,state,risk):
        # Isolate the full-capacity branch: ranking sees zero spendable budget until both old slots exit.
        # Portfolio sizing/fills retain the real entry_plan implementation and actual ledger.
        capacity_observations.append(tuple(sorted(state["positions"])))
        if state["positions"]:
            plan=actual_entry_plan(context,state,risk);plan["maximum_notional_tmn"]="0";return plan
        return actual_entry_plan(context,state,risk)
    monkeypatch.setattr(runner_module,"entry_plan",capacity_plan)
    result=agent.cycle()
    assert any(len(p)==2 for p in capacity_observations) and () in capacity_observations
    decisions={d["symbol"]:d for d in result["decisions"]}
    for symbol in ("SOLTMN","XRPTMN"):
        assert decisions[symbol]["proposal"]["forced_risk_exit"]
        assert decisions[symbol]["proposal"]["action"]=="SELL"
    assert {c["symbol"] for c in agent.client.calls}=={"SUITMN"}
    assert decisions["SUITMN"]["proposal"]["action"]=="BUY"
    assert decisions["DOGETMN"]["proposal"]["action"]=="HOLD"
    assert "SUITMN" in result["portfolio"]["positions"]
    assert result["audit"]["verified"]
    ranking={r["symbol"]:r for r in result["opportunity_ranking"]}
    assert ranking["SUITMN"]["expected_profit_tmn"]>ranking["DOGETMN"]["expected_profit_tmn"]>0
