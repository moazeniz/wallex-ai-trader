from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
import json,sqlite3
import pytest
from experiments.wallex_agent.portfolio import PaperPortfolio
from experiments.wallex_agent.risk import RiskConfig,resolve_action,net_expected_return


def context(symbol='BTCTMN',now=100000,price=100000,qty=1000):
    return {'symbol':symbol,'observed_at':now,'market':{'symbol':symbol,'amount_decimals':5,'minimum_quantity':'0.00001','minimum_notional':'50000'},
        'book':{'bid':price*.999,'ask':price*1.001,'spread_fraction':.002,'observed_at':now,
                'bids':[{'price':price*.999,'quantity':qty}], 'asks':[{'price':price*1.001,'quantity':qty}]},
        'technical':{'last_closed_at':int(now)//3600*3600,'one_hour':{'atr_fraction':.006}}}


def assessments(action='BUY',flags=None):
    return [{'status':'ok','proposed_action':action,'confidence':.9,'risk_flags':flags or []} for _ in range(3)]


def forecast(gross=.03):return {'predicted_gross_return':gross,'horizon_hours':8,'training_cutoff':90000}


def buy():return {'action':'BUY','horizon_hours':8,'reason':['fixture_only']}


def test_shared_cash_position_caps_restart_and_exact_reconciliation(tmp_path):
    path=tmp_path/'portfolio.sqlite';p=PaperPortfolio(path)
    for i,symbol in enumerate(['BTCTMN','ETHTMN']):
        c=context(symbol);state=p.snapshot();proposal=resolve_action(c,forecast(),assessments(),state,100000,RiskConfig());proposal['horizon_hours']=8
        assert proposal['action']=='BUY'
        fill=p.apply(str(i),c,proposal,100000)['event']['fill'];assert fill['kind']=='simulated_orderbook_fill'
        assert D(fill['notional_tmn'])<=D('1000000')
    after=PaperPortfolio(path).snapshot()
    assert len(after['positions'])==2
    assert D(after['cash_tmn'])>=0
    assert p.verify()['accounting_reconciled']
    assert after['initial_cash_tmn']=='5000000'
    assert sum(D(x['notional_tmn']) for x in after['fills'])<=D('2000000')


def test_idempotent_execution_and_collision_rejection(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');c=context();d=buy()
    p.apply('same',c,d,100000);cash=p.snapshot()['cash_tmn']
    assert p.apply('same',c,d,100000)['duplicate']
    assert p.snapshot()['cash_tmn']==cash
    changed=dict(d,action='SELL')
    with pytest.raises(ValueError,match='reused'):p.apply('same',c,changed,100000)


def test_stop_exit_needs_no_llm_and_fees_reconcile(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');p.apply('entry',context(),buy(),100000)
    low=context(now=100060,price=95000)
    decision=resolve_action(low,forecast(),[],p.snapshot(),100060,RiskConfig())
    assert decision['action']=='SELL' and decision['forced_risk_exit']
    event=p.apply('exit',low,decision,100060)['event'];assert event['fill']['side']=='SELL'
    state=p.snapshot();assert not state['positions']
    assert D(state['cash_tmn'])==D('5000000')+D(state['realized_pnl_tmn'])
    assert D(state['realized_pnl_tmn'])<0
    assert p.verify()['accounting_reconciled']


def test_limit_partial_exit_preserves_basis_and_quantity(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');p.apply('entry',context(),buy(),100000)
    state=p.snapshot();original=D(state['positions']['BTCTMN']['quantity']);c=context(now=100060,price=110000,qty=2)
    p.apply('partial',c,{'action':'SELL'},100060);remaining=p.snapshot()['positions']['BTCTMN']
    assert D(remaining['quantity'])==original-D('2')
    assert p.verify()['verified']


def test_minimum_precision_and_absent_liquidity_do_not_fabricate_fills(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');c=context(qty=.000001)
    result=p.apply('tiny',c,buy(),100000)
    assert not result['event']['fill'];assert p.snapshot()['cash_tmn']=='5000000'
    assert p.verify()['verified']


def test_adverse_depth_prices_cannot_overspend_budget(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');c=context();c['book']['asks']=[{'price':100100,'quantity':.1},{'price':140000,'quantity':100}]
    result=p.apply('depth',c,buy(),100000)['event']
    assert result['fill'] is None
    assert result['blocked']=='minimum_order_or_insufficient_depth'
    assert p.snapshot()['cash_tmn']=='5000000'
    assert p.verify()['accounting_reconciled']


def test_cost_edge_news_veto_and_council_failure_hold(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');c=context();state=p.snapshot();cfg=RiskConfig()
    assert resolve_action(c,forecast(.003),assessments(),state,100000,cfg)['action']=='HOLD'
    assert resolve_action(c,forecast(.05),assessments(flags=['exchange_incident']),state,100000,cfg)['action']=='HOLD'
    assert resolve_action(c,forecast(.05),[],state,100000,cfg)['action']=='HOLD'
    assert resolve_action(c,forecast(.05),assessments(),state,100000,cfg)['action']=='BUY'


def test_stale_snapshot_cannot_trade_and_training_overlap_blocks(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');c=context()
    with pytest.raises(ValueError,match='stale'):p.apply('stale',c,buy(),100061)
    f=forecast();f['training_cutoff']=99000
    assert resolve_action(c,f,assessments(),p.snapshot(),100000,RiskConfig())['action']=='HOLD'


def test_drawdown_halts_portfolio_and_forces_exit(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');p.apply('entry',context(),buy(),100000)
    c=context(now=100060,price=70000);p.mark({'BTCTMN':c},100060)
    state=p.snapshot();assert state['halted']
    d=resolve_action(c,forecast(),[],state,100060,RiskConfig())
    assert d['action']=='SELL' and d['reason']==['portfolio_drawdown_halt']


def test_cash_tampering_detected_and_risk_config_change_refused(tmp_path):
    path=tmp_path/'p.sqlite';p=PaperPortfolio(path);p.apply('entry',context(),buy(),100000)
    with sqlite3.connect(path) as db:db.execute("UPDATE meta SET value=? WHERE key='cash'",(json.dumps('9000000'),))
    with pytest.raises(ValueError,match='reconcile'):p.verify()
    with pytest.raises(ValueError,match='differs'):PaperPortfolio(path,replace(RiskConfig(),fee_rate='0.001'))


def test_fee_stress_always_reduces_expected_net_return():
    low=RiskConfig();high=replace(low,fee_rate='0.007')
    assert net_expected_return(.04,.002,high)<net_expected_return(.04,.002,low)<.04


@pytest.mark.parametrize('outer,inner',[(100000,1),(1,100000),(100000,100006),(100006,100000),(100000,float('nan')),(100000,float('inf'))])
def test_both_snapshot_and_book_timestamps_must_be_fresh(tmp_path,outer,inner):
    p=PaperPortfolio(tmp_path/'p.sqlite');c=context();c['observed_at']=outer;c['book']['observed_at']=inner
    with pytest.raises(ValueError,match='stale/future'):p.apply('bad-clock',c,buy(),100000)
    decision=resolve_action(c,forecast(),assessments(),p.snapshot(),100000,RiskConfig())
    assert decision['action']=='HOLD'
    assert decision['reason']==['stale_future_or_invalid_market_data']
    assert p.snapshot()['fill_count']==0


def test_bounded_book_reception_delay_is_allowed(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');c=context(now=99997);c['book']['observed_at']=100000
    assert p.apply('newer-book',c,buy(),100000)['event']['fill']


def test_stale_stop_quote_returns_hold_and_cannot_fill(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');p.apply('entry',context(),buy(),100000)
    stale=context(price=90000)
    decision=resolve_action(stale,forecast(),[],p.snapshot(),100500,RiskConfig())
    assert decision['action']=='HOLD' and not decision['forced_risk_exit']
    assert decision['reason']==['stale_future_or_invalid_market_data']
    fresh=context(now=100500,price=90000)
    assert resolve_action(fresh,forecast(),[],p.snapshot(),100500,RiskConfig())['action']=='SELL'


def test_safe_depth_partial_buy_reports_quantity_and_impact_cap(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');c=context();c['book']['asks']=[{'price':100100,'quantity':2},{'price':140000,'quantity':100}]
    fill=p.apply('limited-entry',c,buy(),100000)['event']['fill']
    assert D(fill['quantity'])==D('2') and fill['partial']
    assert D(fill['average_price'])<=D(str(c['book']['ask']))*(1+D(RiskConfig().max_book_impact_fraction))
    assert p.verify()['verified']


def test_sell_ignores_depth_beyond_impact_limit_and_preserves_basis(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');p.apply('entry',context(),buy(),100000)
    original=p.snapshot()['positions']['BTCTMN'];c=context(now=100060,price=110000)
    c['book']['bids']=[{'price':109890,'quantity':2},{'price':70000,'quantity':100}]
    fill=p.apply('limited-exit',c,{'action':'SELL'},100060)['event']['fill']
    assert D(fill['quantity'])==D('2') and fill['partial']
    remaining=p.snapshot()['positions']['BTCTMN']
    assert D(remaining['quantity'])==D(original['quantity'])-D('2')
    assert D(remaining['basis_tmn'])<D(original['basis_tmn'])
    assert p.verify()['verified']


def test_fill_loss_updates_halt_atomically_and_blocks_next_entry(tmp_path):
    # A deliberately small drawdown cap makes entry fees/impact cross it.
    cfg=replace(RiskConfig(),max_drawdown='0.0001');p=PaperPortfolio(tmp_path/'p.sqlite',cfg)
    p.apply('entry',context(),buy(),100000)
    state=p.snapshot();assert state['halted']
    assert D(state['equity_tmn'])<D(state['peak_equity_tmn'])
    result=p.apply('second-entry',context('ETHTMN'),buy(),100000)['event']
    assert result['fill'] is None and result['blocked']=='portfolio_limit_or_existing_position'
    assert PaperPortfolio(p.path,cfg).snapshot()['halted']
    assert p.verify()['verified']


@pytest.mark.parametrize('field,value',[('basis_tmn','1'),('entry_price','1'),('quantity','1'),('stop_price','1'),('take_price','1'),('expires_at',1)])
def test_position_basis_entry_quantity_and_terms_tampering_detected(tmp_path,field,value):
    p=PaperPortfolio(tmp_path/'p.sqlite');p.apply('entry',context(),buy(),100000)
    with sqlite3.connect(p.path) as db:
        position=json.loads(db.execute("SELECT value FROM positions WHERE symbol='BTCTMN'").fetchone()[0])
        position[field]=value
        db.execute("UPDATE positions SET value=? WHERE symbol='BTCTMN'",(json.dumps(position),))
    with pytest.raises(ValueError,match='Position'):p.verify()


def test_forged_realized_pnl_detected_even_when_audit_hashes_recomputed(tmp_path):
    from experiments.wallex_agent.portfolio import packed,checksum
    p=PaperPortfolio(tmp_path/'p.sqlite');p.apply('entry',context(),buy(),100000)
    p.apply('exit',context(now=100060,price=110000),{'action':'SELL'},100060)
    with sqlite3.connect(p.path) as db:
        row=db.execute("SELECT payload,previous_hash FROM decisions WHERE id='exit'").fetchone()
        event=json.loads(row[0]);event['fill']['realized_pnl_tmn']='1234567'
        db.execute("UPDATE decisions SET payload=?,event_hash=? WHERE id='exit'",(packed(event),checksum({'previous':row[1],'event':event})))
        db.execute("UPDATE fills SET payload=? WHERE decision_id='exit'",(packed(event['fill']),))
        db.execute("UPDATE meta SET value=? WHERE key='realized_pnl'",(packed('1234567'),))
    with pytest.raises(ValueError,match='reconstructed basis'):p.verify()


def test_partial_rounding_multiple_exits_reconcile_to_flat_cash(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');c=context()
    # Individual depth sizes need not be divisible by the order quantity step.
    c['book']['asks']=[{'price':100100,'quantity':1.1234567},{'price':100101,'quantity':100}]
    entry=p.apply('entry',c,buy(),100000)['event']['fill']
    assert D(entry['quantity'])==D(entry['quantity']).quantize(D('.00001'))
    for i in range(3):
        exit_context=context(now=100060+i,price=110000,qty=1.2345678)
        fill=p.apply('partial-'+str(i),exit_context,{'action':'SELL'},100060+i)['event']['fill']
        assert D(fill['quantity'])==D(fill['quantity']).quantize(D('.00001'))
        assert p.verify()['verified']
    p.apply('final-exit',context(now=100070,price=110000),{'action':'SELL'},100070)
    state=p.snapshot();assert not state['positions']
    assert abs((D(state['cash_tmn'])-D(state['initial_cash_tmn']))-D(state['realized_pnl_tmn']))<=D('1e-18')
    assert p.verify()['accounting_reconciled']


def test_fill_table_tampering_detected(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');p.apply('entry',context(),buy(),100000)
    with sqlite3.connect(p.path) as db:
        fill=json.loads(db.execute('SELECT payload FROM fills').fetchone()[0]);fill['quantity']='1'
        db.execute('UPDATE fills SET payload=?',(json.dumps(fill),))
    with pytest.raises(ValueError,match='Fill table'):p.verify()


def test_execution_impact_configuration_cannot_hide_configured_slippage():
    with pytest.raises(ValueError,match='impact'):RiskConfig(max_book_impact_fraction='0.0005')



def test_non_btc_tmn_asset_trades_and_reconciles(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');c=context('ADATMN')
    p.apply('ada-entry',c,buy(),100000)
    assert 'ADATMN' in p.snapshot()['positions']
    p.apply('ada-exit',context('ADATMN',now=100060,price=110000),{'action':'SELL'},100060)
    assert p.verify()['accounting_reconciled']


def test_configured_research_universe_and_identity_are_enforced(tmp_path):
    cfg=replace(RiskConfig(),research_universe=('ADATMN','SOLTMN'))
    p=PaperPortfolio(tmp_path/'p.sqlite',cfg)
    with pytest.raises(ValueError,match='Unsupported'):p.apply('btc',context(),buy(),100000)
    p.apply('ada',context('ADATMN'),buy(),100000)
    assert PaperPortfolio(p.path,cfg).verify()['verified']
    c=context('SOLTMN');c['market']['symbol']='ADATMN'
    with pytest.raises(ValueError,match='mismatched'):p.apply('wrong',c,buy(),100000)
    for symbol in ('BTCUSDT','ADATMN;cmd','../ADATMN','adaTMN','TMN'):
        with pytest.raises(ValueError):RiskConfig(research_universe=(symbol,))



def test_stale_hold_is_audited_without_mark_or_fill(tmp_path):
    p=PaperPortfolio(tmp_path/'p.sqlite');p.apply('entry',context(),buy(),100000)
    before=p.snapshot();stale=context(price=1000)
    result=p.apply('stale-hold',stale,{'action':'HOLD','reason':['stale_future_or_invalid_market_data']},100500)['event']
    after=p.snapshot()
    assert result['fill'] is None and result['market_data_status']=='invalid_for_execution'
    assert 'stale/future' in result['market_data_reason']
    assert after['positions']==before['positions']
    assert after['cash_tmn']==before['cash_tmn']
    assert p.verify()['decision_count']==2
    with pytest.raises(ValueError,match='stale/future'):
        p.apply('stale-sell',stale,{'action':'SELL'},100500)


def test_final_toman_settlement_window_blocks_entry_and_liquidates(tmp_path):
    cfg=replace(RiskConfig(),trading_cutoff_epoch=101000,settlement_buffer_seconds=1800)
    p=PaperPortfolio(tmp_path/'cutoff.sqlite',cfg)
    c=context(now=98000);p.apply('entry',c,buy(),98000)
    at_window=context(now=99200)
    decision=resolve_action(at_window,forecast(),[],p.snapshot(),99200,cfg)
    assert decision['action']=='SELL' and decision['forced_risk_exit']
    assert decision['reason']==['final_toman_settlement_window']
    p.apply('settle',at_window,decision,99200)
    assert not p.snapshot()['positions']
    hold=resolve_action(at_window,forecast(.1),assessments(),p.snapshot(),99200,cfg)
    assert hold['action']=='HOLD' and 'settlement_window_no_new_entries' in hold['reason']
    assert p.verify()['accounting_reconciled']
