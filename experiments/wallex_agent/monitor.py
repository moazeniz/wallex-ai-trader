"""Fast paper-only protective exits using fresh public depth and durable entry evidence.

This module imports no news, LLM, TradingView or quantitative-model clients. It never
enters positions and cannot access an authenticated exchange or submit real orders.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import threading
import time

from .storage import atomic_json
from .portfolio import checksum
from .risk import resolve_action, validate_context

POSITION_GUARD_FIELDS=('symbol','quantity','basis_tmn','entry_price','opened_at','stop_price','take_price','expires_at')


class ProtectiveMonitor:
    def __init__(self,agent,clock=None):
        self.agent=agent
        self.clock=clock
        self._lock=threading.Lock()
        self._contexts={}

    def _now(self):
        return self.clock() if self.clock is not None else time.time()

    def _snapshot_file(self,sha,symbol):
        if not isinstance(sha,str) or not re.fullmatch(r'[0-9a-f]{64}',sha):
            raise ValueError('Invalid immutable context identity')
        path=Path(self.agent.artifacts)/'snapshots'/(sha+'.json')
        context=json.loads(path.read_text())
        if checksum(context)!=sha or context.get('symbol')!=symbol:
            raise ValueError('Immutable context identity mismatch')
        return context

    def _entry_context(self,symbol,position):
        identity=(symbol,position['opened_at'],position['entry_price'])
        if identity in self._contexts:
            return deepcopy(self._contexts[identity])
        found=None
        # The durable account audit identifies the entry even before a restarted
        # process has completed its first research cycle or written status.json.
        with self.agent.portfolio.connect() as db:
            for (raw,) in db.execute('SELECT payload FROM decisions ORDER BY rowid DESC'):
                event=json.loads(raw);fill=event.get('fill') or {}
                if event.get('symbol')!=symbol or fill.get('side')!='BUY':continue
                terms=fill.get('position_terms',{})
                if terms.get('opened_at')==position['opened_at'] and terms.get('entry_price')==position['entry_price']:
                    found=event.get('context_sha256');break
        if found:
            try:context=self._snapshot_file(found,symbol)
            except (OSError,ValueError,KeyError):context=None
        else:context=None
        if context is None:
            # A recent status may reference a newer immutable context at the same
            # artifact destination. Never fabricate metadata if all evidence is lost.
            status_path=Path(self.agent.state)/'status.json'
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            for record in reversed(status.get('decisions',[])):
                if record.get('symbol')!=symbol:continue
                try:context=self._snapshot_file(record.get('context_sha256'),symbol);break
                except (OSError,ValueError,KeyError):continue
        if context is None:raise ValueError('No durable market context for held paper position')
        self._contexts[identity]=deepcopy(context)
        return context

    def tick(self):
        if not self._lock.acquire(blocking=False):
            return {'mode':'paper','live_execution_enabled':False,'status':'busy','exits':[],'events':[],'errors':[]}
        result={'mode':'paper','live_execution_enabled':False,'status':'ok','observed_at':self._now(),
                'positions_checked':0,'marks_refreshed':0,'exits':[],'events':[],'errors':[]}
        try:
            # Fail closed on any broken ledger before protective decisions.
            self.agent.portfolio.verify()
            initial=self.agent.portfolio.snapshot()
            for symbol,position in initial['positions'].items():
                result['positions_checked']+=1;stage='entry_context'
                try:
                    context=self._entry_context(symbol,position)
                    stage='public_depth_refresh'
                    context=self.agent.fresh_execution_context(context)
                    stage='mark'
                    validate_context(context,self._now(),self.agent.risk)
                    state=self.agent.portfolio.mark({symbol:context},self._now())
                    result['marks_refreshed']+=1
                    current=state['positions'].get(symbol)
                    if current is None:continue  # Another paper worker already closed it.
                    neutral={'predicted_gross_return':0,'horizon_hours':8,'training_cutoff':0}
                    stage='protective_decision'
                    proposal=resolve_action(context,neutral,[],state,self._now(),self.agent.risk)
                    if proposal.get('action')!='SELL' or not proposal.get('forced_risk_exit'):continue
                    expected={key:current[key] for key in POSITION_GUARD_FIELDS}
                    proposal=deepcopy(proposal)
                    proposal.update({'expected_position':expected,'protective_monitor':True,
                                     'authority':'paper_only_forced_exit'})
                    context_sha=checksum(context)
                    key=checksum({'protective_monitor_version':1,'expected_position':expected,
                                  'context_sha256':context_sha,'reason':proposal['reason']})
                    stage='snapshot_persistence'
                    snapshot_path=Path(self.agent.artifacts)/'snapshots'/(context_sha+'.json')
                    if snapshot_path.exists():
                        if checksum(json.loads(snapshot_path.read_text()))!=context_sha:
                            raise ValueError('Existing immutable snapshot is corrupted')
                    else:atomic_json(snapshot_path,context)
                    stage='paper_exit'
                    execution=self.agent.portfolio.apply(key,context,proposal,self._now())
                    record={'symbol':symbol,'decision_id':key,'context_sha256':context_sha,
                            'news':[],'quantitative_forecast':None,'council':None,
                            'proposal':proposal,'execution':execution,'protective_monitor':True}
                    stage='decision_persistence'
                    decision_path=Path(self.agent.artifacts)/'decisions'/(key+'.json')
                    if not decision_path.exists():atomic_json(decision_path,record)
                    result['exits'].append(record)
                    event=execution.get('event',{})
                    result['events'].append({'decision_id':key,'symbol':symbol,'action':'SELL','fill':event.get('fill'),
                                             'blocked':event.get('blocked'),'duplicate':execution.get('duplicate',False)})
                except Exception as error:
                    # Arbitrary transport exceptions can contain sensitive request data.
                    result['errors'].append({'symbol':symbol,'stage':stage,'error_type':type(error).__name__})
            result['portfolio']=self.agent.portfolio.snapshot()
            result['audit']=self.agent.portfolio.verify()
            if result['errors']:result['status']='degraded'
        except Exception as error:
            result['status']='error'
            result['errors'].append({'symbol':'account','stage':'account_verification','error_type':type(error).__name__})
        finally:
            result['finished_at']=self._now()
            self._lock.release()
        return result
