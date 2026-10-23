"""Offline regression checks for long-running paper cycles and slow evidence sources."""
from copy import deepcopy
from types import SimpleNamespace
from experiments.wallex_agent import runner as runner_module
from tests.test_wallex_agent_integration_review import (NOW,FakeQuant,FakeNews,FakeClient,market_context,seed_long)


def build_slow_runner(tmp_path,monkeypatch,*,news_delay=0,council_delay=0,held=False):
    clock={'now':NOW}
    monkeypatch.setattr(runner_module.time,'time',lambda:clock['now'])
    monkeypatch.setattr(runner_module.QuantModel,'load',lambda path:FakeQuant(('SOLTMN',)))
    calls=[]
    class Market:
        def snapshot(self,symbol):
            calls.append(symbol)
            result=market_context(symbol,bid=95000 if held else 100000)
            result['observed_at']=clock['now'];result['book']['observed_at']=clock['now']
            # Keep completed-bar evidence fixed across the controlled65-second delay.
            return result
    class News(FakeNews):
        def refresh(self):
            clock['now']+=news_delay
            return {'status':'ok','feeds':[]}
    client=FakeClient(action='BUY')
    agent=runner_module.AgentRunner('fake-model.json',tmp_path/'state',tmp_path/'artifacts',
                                  market=Market(),news=News(),client=client)
    if held:seed_long(agent)
    if council_delay:
        def analyze(context,forecast,evidence,portfolio,now):
            clock['now']+=council_delay
            assessment={'status':'ok','proposed_action':'BUY','confidence':.9,'risk_flags':[]}
            return {'key':'slow-council','status':'ok','assessments':[deepcopy(assessment) for _ in range(3)]}
        agent.council=SimpleNamespace(analyze=analyze)
    return agent,calls


def test_slow_news_cannot_prevent_stop_monitor_refresh(tmp_path,monkeypatch):
    agent,calls=build_slow_runner(tmp_path,monkeypatch,news_delay=65,held=True)
    result=agent.cycle()
    assert result['portfolio']['positions']=={}
    assert result['decisions'][0]['proposal']['action']=='SELL'
    assert len(calls)>=2
    assert agent.client.calls==[]
    assert result['audit']['verified']


def test_slow_council_rechecks_fresh_quote_before_final_buy_gate(tmp_path,monkeypatch):
    agent,calls=build_slow_runner(tmp_path,monkeypatch,council_delay=65)
    result=agent.cycle()
    assert 'SOLTMN' in result['portfolio']['positions']
    assert result['decisions'][0]['proposal']['action']=='BUY'
    assert len(calls)>=2
    assert result['audit']['verified']
