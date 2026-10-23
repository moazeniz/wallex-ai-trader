"""Profit-ranked multi-asset research, specialist AI council, shared-capital paper execution."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import asdict
import hashlib,json,time,os,sys
from .portfolio import PaperPortfolio,checksum,packed
from .risk import RiskConfig,resolve_action,net_expected_return,entry_plan
from .council import ResearchCouncil
from .market import MarketContext
from .news import NewsStore
from .llm import HaikuClient
from .quant import QuantModel
from experiments.wallex_challenge.core import Candle
from .storage import atomic_json

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT.parent


def code_manifest():
    files=list((ROOT/'experiments/wallex_agent').glob('*.py'))
    files += [ROOT/'experiments/wallex_agent/tradingview'/n for n in ['fetch.mjs','package.json','package-lock.json']]
    files += [ROOT/'experiments/wallex_challenge'/n for n in ['api.py','core.py','policy.py']]
    result={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.is_file()}
    return {'files':result,'sha256':checksum(result)}

def rank_opportunities(contexts,forecasts,portfolio,risk):
    ranking=[]
    for symbol,forecast in forecasts.items():
        context=contexts[symbol]
        expected=net_expected_return(float(forecast['predicted_gross_return']),float(context['book']['spread_fraction']),risk)
        budget=float(entry_plan(context,portfolio,risk)['maximum_notional_tmn'])
        ranking.append({'symbol':symbol,'expected_net_return_including_spread':expected,'maximum_notional_tmn':budget,
                        'expected_profit_tmn':expected*budget,'gross_prediction':forecast['predicted_gross_return'],
                        'selected_candidate':forecast.get('selected_candidate','ensemble'),
                        'calibration_selection':forecast.get('calibration_selection')})
    return sorted(ranking,key=lambda row:row['expected_profit_tmn'],reverse=True)


class AgentRunner:
    def __init__(self,model_path,state_dir,artifacts_dir,risk=None,market=None,news=None,client=None,research_limit=3,trading_cutoff_epoch=None):
        self.state=Path(state_dir);self.state.mkdir(parents=True,exist_ok=True)
        self.artifacts=Path(artifacts_dir);self.artifacts.mkdir(parents=True,exist_ok=True)
        self.quant=QuantModel.load(model_path)
        self.symbols=tuple(self.quant.manifest['assets'])
        self.risk=risk or RiskConfig(research_universe=tuple(sorted(self.symbols)),trading_cutoff_epoch=trading_cutoff_epoch)
        if float(self.risk.fee_rate)!=self.quant.manifest['fee_rate_per_side'] or float(self.risk.slippage_rate)!=self.quant.manifest['slippage_rate_per_side']:
            raise ValueError('Risk costs differ from the frozen quantitative model')
        self.market=market or MarketContext();self.news=news or NewsStore(self.state/'news.sqlite')
        self.portfolio=PaperPortfolio(self.state/'paper-portfolio.sqlite',self.risk)
        self.client=client or HaikuClient(Path(os.getenv('WALLEX_ANTHROPIC_SECRET_FILE',str(BASE/'private/anthropic.env'))),Path(os.getenv('WALLEX_ANTHROPIC_BUDGET_DB',str(BASE/'private/haiku-budget.sqlite'))),max_usd=2,max_calls=1200)
        self.council=ResearchCouncil(self.client,self.state/'council-cache.sqlite')
        if not 1<=research_limit<=8:raise ValueError('Research limit1..8')
        self.research_limit=research_limit

    def fresh_execution_context(self,context):
        method=getattr(self.market,'refresh_execution_context',None)
        return method(context) if method else self.market.snapshot(context['symbol'])

    def cycle(self,symbols=None,refresh_news=True):
        symbols=tuple(symbols or self.symbols)
        if not symbols or len(set(symbols))!=len(symbols):raise ValueError('Nonempty unique symbols required')
        if any(s not in self.symbols for s in symbols):raise ValueError('Symbol not registered in frozen research model')
        self.portfolio.verify()
        held=tuple(self.portfolio.snapshot()['positions'])
        symbols=tuple(dict.fromkeys((*symbols,*held)))
        started=time.time();contexts={};errors={};forecasts={};ranking=[]
        def collect(symbol):
            try:return symbol,self.market.snapshot(symbol),None
            except Exception as exc:return symbol,None,type(exc).__name__
        with ThreadPoolExecutor(max_workers=4) as pool:
            for symbol,value,error in pool.map(collect,symbols):
                if value:contexts[symbol]=value
                else:errors[symbol]=error
        feed_status=self.news.refresh() if refresh_news else {'status':'cached_only'}
        now=time.time();self.portfolio.mark(contexts,now)
        result={'mode':'paper','started_at':started,'observed_at':now,'news_refresh':feed_status,'source_errors':errors,
                'source_manifest':code_manifest(),'decisions':[],'live_execution_enabled':False,'research_universe':list(symbols)}
        for symbol,context in contexts.items():
            try:forecasts[symbol]=self.quant.forecast(symbol,[Candle(**b) for b in context['candles']])
            except Exception as exc:errors[symbol]='quant_'+type(exc).__name__
        ranking=rank_opportunities(contexts,forecasts,self.portfolio.snapshot(),self.risk)
        state=self.portfolio.snapshot()
        viable=[x['symbol'] for x in ranking if x['expected_net_return_including_spread']>self.risk.minimum_net_edge and x['selected_candidate']!='zero']
        # Research the best current net opportunities; when none pass, inspect the top few diagnostically.
        targets=set(viable[:self.research_limit] if viable else [x['symbol'] for x in ranking[:min(2,self.research_limit)]])
        targets.update(state['positions'])
        result['opportunity_ranking']=ranking;result['specialist_research_targets']=sorted(targets)
        order=sorted(contexts,key=lambda s:(s not in state['positions'],next((i for i,x in enumerate(ranking) if x['symbol']==s),999)))
        held_at_cycle_start=set(state['positions'])
        pending=list(order);entries_ranked=False
        while pending:
            if not entries_ranked and not any(s in held_at_cycle_start for s in pending):
                # Risk exits can release cash/exposure. Reprice opportunity budgets before selecting any new entry.
                ranking=rank_opportunities(contexts,forecasts,self.portfolio.snapshot(),self.risk)
                viable=[x['symbol'] for x in ranking if x['expected_net_return_including_spread']>self.risk.minimum_net_edge and x['selected_candidate']!='zero' and x['maximum_notional_tmn']>=float(self.risk.min_order_tmn)]
                targets=set(viable[:self.research_limit] if viable else [x['symbol'] for x in ranking[:min(2,self.research_limit)]])
                targets.update(self.portfolio.snapshot()['positions'])
                result['opportunity_ranking']=ranking;result['specialist_research_targets']=sorted(targets)
                priority={x['symbol']:i for i,x in enumerate(ranking)}
                pending.sort(key=lambda s:priority.get(s,999));entries_ranked=True
            symbol=pending.pop(0)
            context=contexts[symbol];now=time.time();state=self.portfolio.snapshot();forecast=forecasts.get(symbol);council=None
            if symbol in state['positions']:
                try:
                    context=self.fresh_execution_context(context);self.portfolio.mark({symbol:context},time.time())
                    state=self.portfolio.snapshot();now=time.time()
                except Exception as exc:errors[symbol]='risk_quote_'+type(exc).__name__
            neutral={'predicted_gross_return':0,'horizon_hours':8,'training_cutoff':0}
            emergency=resolve_action(context,neutral,[],state,now,self.risk)
            evidence=self.news.snapshot(symbol,as_of=now,limit=15)
            if emergency.get('forced_risk_exit'):
                proposal=emergency;key=checksum({'risk_exit':symbol,'opened_at':state['positions'][symbol]['opened_at'],'reason':proposal['reason']})
            elif forecast is None:
                proposal={'action':'HOLD','reason':['quantitative_model_unavailable'],'forced_risk_exit':False}
                key=checksum({'quant_error':symbol,'bar':context['technical']['last_closed_at'],'source':result['source_manifest']['sha256']})
            elif symbol not in targets:
                proposal={'action':'HOLD','reason':['not_among_top_current_net_opportunities'],'horizon_hours':forecast['horizon_hours'],'forced_risk_exit':False}
                key=checksum({'ranked_out':symbol,'bar':context['technical']['last_closed_at'],'model':forecast['model_hash']})
            else:
                council=self.council.analyze(context,forecast,evidence,state,now)
                try:
                    fresh=self.fresh_execution_context(context)
                    if fresh['technical']['last_closed_at']!=context['technical']['last_closed_at']:
                        proposal={'action':'HOLD','reason':['completed_bar_changed_during_research'],'forced_risk_exit':False}
                    else:
                        context=fresh;self.portfolio.mark({symbol:context},time.time())
                        proposal=resolve_action(context,forecast,council['assessments'],self.portfolio.snapshot(),time.time(),self.risk)
                except Exception as exc:
                    proposal={'action':'HOLD','reason':['execution_snapshot_unavailable'],'error_type':type(exc).__name__,'forced_risk_exit':False}
                proposal['horizon_hours']=forecast['horizon_hours']
                key=checksum({'research':council['key'],'symbol':symbol,'position_open':symbol in state['positions']})
            if proposal['action'] in ['BUY','SELL']:
                try:
                    execution_context=self.fresh_execution_context(context)
                    if forecast and not proposal.get('forced_risk_exit') and execution_context['technical']['last_closed_at']!=context['technical']['last_closed_at']:
                        proposal={'action':'HOLD','reason':['completed_bar_changed_during_research'],'forced_risk_exit':False}
                    context=execution_context;self.portfolio.mark({symbol:context},time.time())
                    if proposal.get('forced_risk_exit'):
                        proposal=resolve_action(context,neutral,[],self.portfolio.snapshot(),time.time(),self.risk)
                    if forecast and proposal['action']=='BUY':
                        proposal=resolve_action(context,forecast,council['assessments'],self.portfolio.snapshot(),time.time(),self.risk)
                        proposal['horizon_hours']=forecast['horizon_hours']
                except Exception as exc:
                    proposal={'action':'HOLD','reason':['execution_snapshot_unavailable'],'error_type':type(exc).__name__,'forced_risk_exit':False}
            if proposal['action']=='SELL':
                proposal['expected_position']=self.portfolio.snapshot()['positions'].get(symbol,{})
            context_sha=checksum(context)
            if proposal.get('forced_risk_exit'):
                current_position=self.portfolio.snapshot()['positions'].get(symbol,{})
                key=checksum({'risk_exit':symbol,'position':current_position,'reason':proposal['reason'],'execution_snapshot':context_sha})
            path=self.artifacts/'snapshots'/f'{context_sha}.json'
            if not path.exists():atomic_json(path,context)
            proposal.update({'quant_model_hash':forecast.get('model_hash') if forecast else None,
                             'council_key':council.get('key') if council else None,'news_evidence_ids':[x['id'] for x in evidence],
                             'source_sha256':result['source_manifest']['sha256']})
            if self.portfolio.seen(key):execution={'duplicate':True,'reason':'decision_already_applied'}
            else:execution=self.portfolio.apply(key,context,proposal,time.time())
            record={'symbol':symbol,'decision_id':key,'context_sha256':context_sha,'news':evidence,
                    'quantitative_forecast':forecast,'council':council,'proposal':proposal,'execution':execution}
            decision_path=self.artifacts/'decisions'/f'{key}.json'
            if not decision_path.exists():atomic_json(decision_path,record)
            result['decisions'].append(record)
            self.portfolio.mark({symbol:context},time.time())
        result['portfolio']=self.portfolio.snapshot();result['audit']=self.portfolio.verify()
        result['finished_at']=time.time();result['llm_budget']=self.client.budget_status()
        atomic_json(self.state/'status.json',result)
        atomic_json(self.artifacts/'cycles'/f'{int(started*1000)}-{checksum(result)[:12]}.json',result)
        return result
