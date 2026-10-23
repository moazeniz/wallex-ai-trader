"""Restart, no-provider, and concurrency checks for the paper-only stop monitor."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from experiments.wallex_agent.market import MarketContext,MarketConfig
from experiments.wallex_agent.monitor import ProtectiveMonitor
from experiments.wallex_agent.portfolio import PaperPortfolio,checksum
from experiments.wallex_agent.risk import RiskConfig
from experiments.wallex_challenge.core import atomic_json
from tests.test_wallex_agent_integration_review import NOW,market_context


class DepthOnlyAPI:
    def __init__(self,bid=95000,quantity=100):self.bid=bid;self.quantity=quantity;self.calls=[];self.error=None
    def get(self,path,params):
        assert path=='/v1/depth';assert params=={'symbol':'SOLTMN'}
        self.calls.append((path,params))
        if self.error:raise self.error
        return {'success':True,'result':{'bid':[{'price':self.bid,'quantity':self.quantity}],
                                        'ask':[{'price':self.bid*1.001,'quantity':self.quantity}]}}
    def market(self,*args):pytest.fail('Protective monitor must not request metadata/history')
    def history(self,*args):pytest.fail('Protective monitor must not request history')


def setup(tmp_path,bid=95000,quantity=100,seed=True):
    state=tmp_path/'state';artifacts=tmp_path/'artifacts';state.mkdir()
    risk=RiskConfig(research_universe=('SOLTMN',))
    portfolio=PaperPortfolio(state/'paper-portfolio.sqlite',risk)
    source=market_context('SOLTMN')
    source['market_metadata']={'source':'wallex','quote_currency':'TMN','base_currency':'SOL','venue':{}}
    source['provenance']={'primary':'https://api.wallex.ir','snapshot_cutoff':NOW}
    sha=checksum(source)
    if seed:
        atomic_json(artifacts/'snapshots'/(sha+'.json'),source)
        portfolio.apply('seed',source,{'action':'BUY','horizon_hours':8},NOW)
    api=DepthOnlyAPI(bid,quantity)
    market=MarketContext(api=api,config=MarketConfig(tradingview_enabled=False))
    clock={'now':NOW+65}
    agent=SimpleNamespace(state=state,artifacts=artifacts,risk=risk,portfolio=portfolio,
                          fresh_execution_context=lambda context:market.refresh_execution_context(context,clock['now']))
    return agent,api,clock,source,sha


def test_restart_before_first_cycle_uses_durable_entry_and_depth_only(tmp_path):
    agent,api,clock,source,sha=setup(tmp_path)
    # No status/council cache exists, and the account instance is reopened.
    agent.portfolio=PaperPortfolio(agent.portfolio.path,agent.risk)
    result=ProtectiveMonitor(agent,clock=lambda:clock['now']).tick()
    assert result['status']=='ok' and result['positions_checked']==1
    assert result['portfolio']['positions']=={}
    assert result['events'][0]['action']=='SELL'
    assert result['events'][0]['fill']['side']=='SELL'
    assert result['audit']['verified'] and len(api.calls)==1
    assert result['live_execution_enabled'] is False
    json.dumps(result,allow_nan=False)
    event=result['events'][0];record=json.loads((agent.artifacts/'decisions'/(event['decision_id']+'.json')).read_text())
    assert record['council'] is None and record['news']==[]
    assert (agent.artifacts/'snapshots'/(record['context_sha256']+'.json')).exists()


def test_no_positions_performs_no_network_or_audit_hold(tmp_path):
    agent,api,clock,*_=setup(tmp_path,seed=False)
    result=ProtectiveMonitor(agent,clock=lambda:clock['now']).tick()
    assert result['positions_checked']==0 and result['events']==[] and api.calls==[]
    assert result['audit']['decision_count']==0


def test_mark_only_tick_creates_no_hold_event_and_preserves_cash(tmp_path):
    agent,api,clock,*_=setup(tmp_path,bid=100000)
    before=agent.portfolio.snapshot()
    monitor=ProtectiveMonitor(agent,clock=lambda:clock['now'])
    result=monitor.tick();clock['now']+=30;second=monitor.tick()
    assert result['events']==second['events']==[]
    assert result['marks_refreshed']==1 and result['portfolio']['cash_tmn']==before['cash_tmn']
    assert second['audit']['decision_count']==1


def test_partial_exit_continues_on_next_fresh_tick(tmp_path):
    agent,api,clock,*_=setup(tmp_path,quantity=2)
    monitor=ProtectiveMonitor(agent,clock=lambda:clock['now'])
    first=monitor.tick();left=float(first['portfolio']['positions']['SOLTMN']['quantity'])
    assert first['events'][0]['fill']['partial']
    clock['now']+=30;second=monitor.tick()
    assert float(second['portfolio']['positions']['SOLTMN']['quantity'])<left
    assert first['events'][0]['decision_id']!=second['events'][0]['decision_id']
    assert second['audit']['verified']


def test_changed_position_race_blocks_monitor_fill_atomically(tmp_path):
    agent,api,clock,*_=setup(tmp_path,quantity=2)
    original=agent.portfolio.apply;intervened=[]
    def race(decision_id,context,decision,now):
        if decision.get('protective_monitor') and not intervened:
            other=original('other-paper-worker',context,{'action':'SELL'},now)
            intervened.append(other)
        return original(decision_id,context,decision,now)
    agent.portfolio.apply=race
    result=ProtectiveMonitor(agent,clock=lambda:clock['now']).tick()
    assert intervened[0]['event']['fill']['quantity']=='2.00000000'
    assert result['events'][0]['fill'] is None
    assert result['events'][0]['blocked']=='position_changed_since_protection'
    assert result['portfolio']['fill_count']==2  # One entry and only the other worker's exit.
    assert result['audit']['verified']


def test_quote_failure_is_redacted_and_preserves_held_position(tmp_path):
    agent,api,clock,*_=setup(tmp_path)
    api.error=TimeoutError('secret from a transport must never be recorded')
    result=ProtectiveMonitor(agent,clock=lambda:clock['now']).tick()
    assert result['status']=='degraded' and result['events']==[]
    assert result['errors']==[{'symbol':'SOLTMN','stage':'public_depth_refresh','error_type':'TimeoutError'}]
    assert 'secret' not in json.dumps(result)
    assert 'SOLTMN' in result['portfolio']['positions']


def test_missing_or_corrupt_snapshot_blocks_without_fabricated_context(tmp_path):
    agent,api,clock,source,sha=setup(tmp_path)
    path=agent.artifacts/'snapshots'/(sha+'.json')
    source['book']['bid']=1;path.write_text(json.dumps(source))
    result=ProtectiveMonitor(agent,clock=lambda:clock['now']).tick()
    assert result['status']=='degraded' and result['events']==[] and api.calls==[]
    assert result['errors'][0]['stage']=='entry_context'


def test_status_can_reference_a_newer_immutable_fallback_snapshot(tmp_path):
    agent,api,clock,source,sha=setup(tmp_path)
    (agent.artifacts/'snapshots'/(sha+'.json')).unlink()
    newer=deepcopy(source);newer['analysis_observed_at']=NOW;newer['observed_at']=NOW+1
    newer_sha=checksum(newer);atomic_json(agent.artifacts/'snapshots'/(newer_sha+'.json'),newer)
    atomic_json(agent.state/'status.json',{'decisions':[{'symbol':'SOLTMN','context_sha256':newer_sha}]})
    result=ProtectiveMonitor(agent,clock=lambda:clock['now']).tick()
    assert result['status']=='ok' and result['portfolio']['positions']=={}


def test_overlapping_tick_is_busy_and_never_enters_position(tmp_path):
    agent,api,clock,*_=setup(tmp_path,seed=False)
    monitor=ProtectiveMonitor(agent,clock=lambda:clock['now'])
    monitor._lock.acquire()
    try:result=monitor.tick()
    finally:monitor._lock.release()
    assert result['status']=='busy' and result['events']==[] and api.calls==[]
