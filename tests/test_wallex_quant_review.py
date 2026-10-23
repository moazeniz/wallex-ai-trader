"""Independent adversarial checks for causal selection and research provenance.

All fixtures are generated in memory; no network, paid model, or market downloads.
"""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.wallex_agent import quant as q


class FakeTree:
    def predict(self,x):return np.zeros(len(x))
    def save_model(self,path):
        from pathlib import Path
        Path(path).write_text('{"offline_fixture":true}')


def rows(n=120):
    return [{'t':i*3600,'o':100*(1.001**i),'h':101*(1.001**i),'l':99*(1.001**i),
             'c':100*(1.001**i),'v':10+i%5} for i in range(n)]


def parameters():
    n=len(q.FEATURE_NAMES)
    return {'means':[0.0]*n,'scales':[1.0]*n,'coefficients':[0.0]*n,'intercept':0.05}


def fake_model(dataset,protocol):
    residual={name:{'q10':-.01,'q90':.01,'std':.01,'count':100} for name in q.CANDIDATES}
    manifest={'schema_version':1,'feature_names':q.FEATURE_NAMES,'quote_currency':'TMN',
              'horizon_hours':8,'interval_seconds':3600,'fee_rate_per_side':.0035,
              'slippage_rate_per_side':.001,'protocol_hash':q.digest(protocol),
              'assets':{'BTCTMN':{'ridge':parameters(),'selected_candidate':'ridge','residuals':residual,
                        'training_cutoff':20*3600,'calibration_cutoff':60*3600,
                        'dataset_hash':q.digest(dataset)}}}
    return q.QuantModel(manifest,{'BTCTMN':FakeTree()},'.')


def test_forecast_rejects_history_before_calibration_matures():
    data={'market':{'symbol':'BTCTMN'},'candles':rows()}
    model=fake_model(data,{})
    # Features end at hour 55, while calibration selection/residuals use hour 60.
    with pytest.raises(ValueError,match='cutoff|calibration|future|matur'):
        model.forecast('BTCTMN',data['candles'][:55])
    forecast=model.forecast('BTCTMN',data['candles'])
    assert forecast['quote_currency']=='TMN'
    assert forecast['predicted_net_return']<forecast['predicted_gross_return']
    assert forecast['calibration_cutoff']<forecast['feature_cutoff']


@pytest.mark.parametrize('mismatch',['protocol','model','dataset'])
def test_shared_replay_rejects_unrelated_artifacts(tmp_path,monkeypatch,mismatch):
    dataset={'market':{'symbol':'BTCTMN'},'candles':rows()}
    protocol={'boundaries':[0,40*3600,70*3600,120*3600],'minimum_net_edge':.002,
              'dataset_hashes':{'BTCTMN':q.digest(dataset)}}
    model=fake_model(dataset,protocol)
    report={'protocol':deepcopy(protocol),'model_hash':model.model_hash}
    source=deepcopy(dataset)
    if mismatch=='protocol':report['protocol']['minimum_net_edge']=.99
    elif mismatch=='model':report['model_hash']='0'*64
    else:
        source['candles'][-1]['o']*=1.5;source['candles'][-1]['h']*=1.5
        source['candles'][-1]['l']*=1.5;source['candles'][-1]['c']*=1.5
    monkeypatch.setattr(q.QuantModel,'load',classmethod(lambda cls,path:model))
    report_path=tmp_path/'report.json';report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='hash|protocol|dataset|model'):
        q.simulate_shared_research('unused',report_path,{'BTCTMN':source},tmp_path/'shared.json')


def test_calibration_selector_invariant_to_first_evaluation_open(tmp_path,monkeypatch):
    # 608 hourly bars gives cal_end=425h, one hour after the final eligible8h label.
    # A+1h stress must not inspect that first evaluation open during selection.
    original={'market':{'symbol':'BTCTMN'},'candles':rows(608)}
    perturbed=deepcopy(original)
    perturbed['candles'][425].update({'o':.0001,'l':.0001})
    monkeypatch.setattr(q,'fit_estimators',lambda train:(parameters(),FakeTree()))
    def predict(params,tree,x):
        return {name:(np.zeros(len(x)) if name=='zero' else np.full(len(x),.05)) for name in q.CANDIDATES}
    monkeypatch.setattr(q,'predictions',predict)
    config={'symbols':['BTCTMN'],'fee_rate_per_side':0,'slippage_rate_per_side':0,
            'minimum_net_edge':.002,'quant':{'selection_mode':'calibration_only'}}
    first=q.train_research(tmp_path/'first',config,datasets={'BTCTMN':original})
    second=q.train_research(tmp_path/'second',config,datasets={'BTCTMN':perturbed})
    first_model=json.loads(__import__('pathlib').Path(first['model_path']).read_text())
    second_model=json.loads(__import__('pathlib').Path(second['model_path']).read_text())
    assert first_model['assets']['BTCTMN']['selected_candidate']==second_model['assets']['BTCTMN']['selected_candidate']
    assert first_model['assets']['BTCTMN']['calibration_selection']==second_model['assets']['BTCTMN']['calibration_selection']


def test_small_positive_gross_is_a_net_loss_after_roundtrip_costs():
    assert q.net_return(.0035,.0035,.001)<0
    assert q.net_return(0,.0035,.001)<0
    sample={'features':[0.0]*len(q.FEATURE_NAMES),'opportunity':True}
    assert not q.entry_decision('ridge',sample,.0035,.0035,.001,.002)
