"""Three specialist Haiku perspectives; bounded requests, explicit evidence and no order tools."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,sqlite3,time
from .portfolio import checksum,packed

ROLES={
    'crypto_analyst':'Assess asset-specific crypto events, adoption, flows and market structure. Separate repeated headlines from new facts; abstain without actionable evidence.',
    'macro_analyst':'Assess rates, central banks, currencies, economics and geopolitical transmission to this asset and its TMN quote. Do not treat every world headline as financially relevant.',
    'skeptical_risk_analyst':'Try to invalidate the trade thesis: stale information, already-priced events, unreliable sources, liquidity, uncertainty, forecast limitations and adverse scenarios. HOLD is valid.'}

class ResearchCouncil:
    def __init__(self,client,cache_path):
        self.client=client;self.path=Path(cache_path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as db:db.execute('CREATE TABLE IF NOT EXISTS council_cache(key TEXT PRIMARY KEY,created_at INTEGER NOT NULL,payload TEXT NOT NULL)')
        self.path.chmod(0o600)

    def analyze(self,context,forecast,news,portfolio,as_of):
        cross=context.get('cross_market',{})
        closed_price=context['technical']['one_hour'].get('last_price')
        if closed_price is None:
            bars=context.get('candles') or []
            closed_price=bars[-1].get('c') if bars else None
        base={'symbol':context['symbol'],'as_of':float(as_of),'mode':'paper_research',
              'quantitative_forecast':forecast,'wallex_technical':context['technical'],
              'closed_bar_market':{'currency':'TMN','close_price':closed_price},
              'execution_cost_assumptions':{'fee_rate_per_side':forecast.get('assumed_fee_rate_per_side',.0035),'slippage_rate_per_side':forecast.get('assumed_slippage_rate_per_side',.001),'account_fee_verified':False},
              'global_market':{k:cross.get(k) for k in ['source','status','symbol','quote_currency','conversion_to_tmn_applied','technical']},
              'position_open':context['symbol'] in portfolio['positions'],'initial_balance_tmn':portfolio['initial_cash_tmn'],
              'instruction':'Give an evidence-based BUY/SELL/HOLD research proposal. Model confidence is not a calibrated win probability. Quote actual IDs; do not invent data, expected profits, or market prices. No real orders are authorized by this research call.'}
        from .llm import SYSTEM,SCHEMA,MODEL
        # Every semantic prompt input and model/schema/system version participates in the cache key.
        # as_of is excluded only from reuse when evidence and market inputs are otherwise identical;
        # a cached assessment is never used after1hour, and risk rechecks fresh quotes before any fill.
        identity=dict(base);identity.pop('as_of')
        research={'context':identity,'news':news,'roles':ROLES,'model':MODEL,'system':SYSTEM,'schema':SCHEMA,'council_version':3}
        key=checksum(research)
        with sqlite3.connect(self.path) as db:cached=db.execute('SELECT created_at,payload FROM council_cache WHERE key=?',(key,)).fetchone()
        if cached:
            value=json.loads(cached[1])
            ttl=3600 if value.get('status')=='ok' else 600
            if 0<=as_of-cached[0]<ttl:
                value['cached']=True;return value
        def one(item):
            role,instruction=item;prompt=dict(base,analyst_role=role,specialist_instruction=instruction)
            try:
                result=self.client.assess(prompt,news)
                result=dict(result);result['analyst_role']=role
                return result
            except Exception as exc:
                # Do not copy arbitrary exceptions/HTTP request objects containing credentials into logs.
                return {'status':'error','analyst_role':role,'error_type':type(exc).__name__,'proposed_action':'HOLD','confidence':0,
                        'risk_flags':['analyst_unavailable'],'evidence_ids':[],'thesis':'No usable model assessment; new entry disabled.',
                        'invalidation':'Retry only with a new bounded research run or a recovered provider.','horizon_hours':forecast['horizon_hours']}
        with ThreadPoolExecutor(max_workers=3) as pool:assessments=list(pool.map(one,ROLES.items()))
        result={'key':key,'created_at':float(as_of),'cached':False,'assessments':assessments,
                'status':'ok' if all(x['status']=='ok' for x in assessments) else 'incomplete',
                'news_evidence_count':len(news),'research_sha256':checksum(research)}
        # An incomplete opinion also pauses automatic retries of unchanged research.
        # Ambiguous provider attempts retain their charges/reservations in the ledger.
        # Successful councils reuse for 1h; incomplete councils cool down for 10min.
        with sqlite3.connect(self.path) as db:db.execute('INSERT OR REPLACE INTO council_cache VALUES(?,?,?)',(key,float(as_of),packed(result)))
        return result
