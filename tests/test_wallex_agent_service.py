"""Persistent paper process recovers without replacing the last completed portfolio report."""
import json
from types import SimpleNamespace
from experiments.wallex_agent import __main__ as cli


def test_service_failure_retries_without_destroying_last_good_status(tmp_path,monkeypatch):
    last_good={'portfolio':{'cash_tmn':4991234},'marker':'prior-completed-cycle'}
    (tmp_path/'status.json').write_text(json.dumps(last_good))
    observed=[]
    class Stop:
        cycles=0
        def is_set(self):return self.cycles>=2
        def wait(self,delay):
            observed.append(json.loads((tmp_path/'daemon.json').read_text()))
            self.cycles+=1
    calls=[]
    def cycle():
        calls.append(1)
        if len(calls)==1:raise TimeoutError('private provider request must never be logged')
        return {'successful':True}
    monkeypatch.setattr(cli,'summarize',lambda report:report)
    cli.serve(SimpleNamespace(state=tmp_path,cycle=cycle),30,Stop())
    assert observed[0]['last_error_type']=='TimeoutError'
    assert observed[0]['consecutive_errors']==1
    assert observed[1]['cycles_completed']==1
    assert observed[1]['consecutive_errors']==0
    assert observed[1]['last_completed_at'] is not None
    assert json.loads((tmp_path/'status.json').read_text())==last_good
    assert 'stopped_at' in json.loads((tmp_path/'daemon.json').read_text())


def test_concurrent_atomic_reports_remain_valid_and_leave_no_temporary_files(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from experiments.wallex_agent.storage import atomic_json
    path=tmp_path/'health.json'
    values=[{'worker':i,'values':[i]*100} for i in range(24)]
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(lambda value:atomic_json(path,value),values))
    assert json.loads(path.read_text()) in values
    assert list(tmp_path.iterdir())==[path]
