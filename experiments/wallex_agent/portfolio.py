"""Persistent, shared-capital paper fills against captured Wallex order books. No live transport."""
from contextlib import contextmanager
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
import hashlib,json,sqlite3,math,time
from .risk import D, RiskConfig, entry_plan, validate_context, valid_tmn_symbol


def packed(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def checksum(value):return hashlib.sha256(packed(value).encode()).hexdigest()

class PaperPortfolio:
    def __init__(self,path,config=None):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.config=config or RiskConfig()
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS positions(symbol TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions(id TEXT PRIMARY KEY,created_at INTEGER NOT NULL,payload TEXT NOT NULL,previous_hash TEXT NOT NULL,event_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS fills(id INTEGER PRIMARY KEY AUTOINCREMENT,decision_id TEXT NOT NULL,payload TEXT NOT NULL);
            ''')
            initial={'cash':self.config.initial_cash_tmn,'peak':self.config.initial_cash_tmn,'halted':False,'fees':'0','realized_pnl':'0',
                     'initial_cash':self.config.initial_cash_tmn,'config':self.config.__dict__}
            for key,value in initial.items():db.execute('INSERT OR IGNORE INTO meta VALUES(?,?)',(key,packed(value)))
            stored=json.loads(db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
            if stored!=json.loads(packed(self.config.__dict__)):raise ValueError('Paper account risk config differs; use a new account DB')
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=15);db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA busy_timeout=15000')
        try:
            db.execute('BEGIN IMMEDIATE');yield db;db.commit()
        except Exception:db.rollback();raise
        finally:db.close()

    def _read(self,db):
        meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM meta')}
        positions={k:json.loads(v) for k,v in db.execute('SELECT symbol,value FROM positions')}
        equity=D(meta['cash'])+sum((D(p['quantity'])*D(p['mark_price']) for p in positions.values()),D(0))
        return meta,positions,equity

    def _update_risk(self,db):
        meta,positions,equity=self._read(db)
        old_peak=D(meta['peak'])
        if not equity.is_finite() or not old_peak.is_finite() or old_peak<=0 or type(meta['halted']) is not bool:
            raise ValueError('Invalid persisted equity/risk state')
        peak=max(old_peak,equity)
        halted=meta['halted'] or (peak-equity)/peak>=D(self.config.max_drawdown)
        for key,value in [('peak',str(peak)),('halted',halted)]:
            db.execute('UPDATE meta SET value=? WHERE key=?',(packed(value),key))
        return self._read(db)

    def snapshot(self):
        with self.connect() as db:
            meta,positions,equity=self._read(db)
            fills=[json.loads(v[0]) for v in db.execute('SELECT payload FROM fills ORDER BY id')]
            return {'mode':'paper','cash_tmn':meta['cash'],'equity_tmn':str(equity),'peak_equity_tmn':meta['peak'],
                    'halted':meta['halted'],'positions':positions,'fees_tmn':meta['fees'],'realized_pnl_tmn':meta['realized_pnl'],
                    'initial_cash_tmn':meta['initial_cash'],'fill_count':len(fills),'fills':fills}

    def mark(self,contexts,now):
        with self.connect() as db:
            meta,positions,equity=self._read(db)
            for symbol,position in positions.items():
                context=contexts.get(symbol)
                if context:
                    try:validate_context(context,now,self.config)
                    except (ValueError,KeyError,TypeError):continue
                    if context['symbol']!=symbol:raise ValueError('Mark symbol mismatch')
                    position['mark_price']=str(context['book']['bid'])
                    db.execute('UPDATE positions SET value=? WHERE symbol=?',(packed(position),symbol))
            self._update_risk(db)
        return self.snapshot()

    def seen(self,decision_id):
        with self.connect() as db:return db.execute('SELECT 1 FROM decisions WHERE id=?',(decision_id,)).fetchone() is not None

    def apply(self,decision_id,context,decision,now):
        data_error=None
        try:validate_context(context,now,self.config)
        except ValueError as error:
            # A no-order observation remains auditable after a long research cycle.
            # Invalid structure/quotes still fail, and no stale data may update marks.
            if decision.get('action')!='HOLD' or not any(word in str(error) for word in ('stale/future','acquisition delay')):
                raise
            data_error=str(error)
        with self.connect() as db:
            prior=db.execute('SELECT payload FROM decisions WHERE id=?',(decision_id,)).fetchone()
            if prior:
                previous_event=json.loads(prior[0])
                if previous_event['context_sha256']!=checksum(context) or previous_event['decision']!=decision:
                    raise ValueError('Decision ID reused for different input')
                return {'duplicate':True,'event':previous_event}
            meta,positions,equity=self._read(db);symbol=context['symbol'];action=decision['action'];fill=None;blocked=None
            if symbol in positions and data_error is None:
                positions[symbol]['mark_price']=str(context['book']['bid'])
                db.execute('UPDATE positions SET value=? WHERE symbol=?',(packed(positions[symbol]),symbol))
            meta,positions,equity=self._update_risk(db)
            if action=='BUY':
                if meta['halted'] or symbol in positions or len(positions)>=self.config.max_positions:blocked='portfolio_limit_or_existing_position'
                else:
                    state={'cash_tmn':meta['cash'],'equity_tmn':str(equity),'positions':positions}
                    plan=entry_plan(context,state,self.config)
                    fill=self._fill(context,'BUY',D(plan['maximum_notional_tmn']),None)
                    if fill:
                        fee=D(fill['fee_tmn']);cost=D(fill['notional_tmn'])+fee
                        if cost>D(meta['cash']):raise ValueError('Paper cash would be negative')
                        meta['cash']=str(D(meta['cash'])-cost);meta['fees']=str(D(meta['fees'])+fee)
                        avg=D(fill['average_price']);qty=D(fill['quantity'])
                        position={'symbol':symbol,'quantity':str(qty),'basis_tmn':str(cost),'entry_price':str(avg),'mark_price':str(context['book']['bid']),
                                  'stop_price':str(avg*(1-D(plan['stop_fraction']))),'take_price':str(avg*(1+D(plan['take_fraction']))),
                                  'opened_at':int(now),'expires_at':int(now)+int(decision['horizon_hours'])*3600}
                        fill['position_terms']={key:position[key] for key in ('entry_price','stop_price','take_price','opened_at','expires_at')}
                        db.execute('INSERT INTO positions VALUES(?,?)',(symbol,packed(position)))
                    else:blocked='minimum_order_or_insufficient_depth'
            elif action=='SELL':
                expected=decision.get('expected_position')
                identity_keys=('symbol','quantity','basis_tmn','entry_price','opened_at','stop_price','take_price','expires_at')
                if expected is not None and not isinstance(expected,dict):raise ValueError('Expected position must be a snapshot')
                if symbol not in positions:blocked='no_long_position'
                elif expected is not None and any(expected.get(k)!=positions[symbol].get(k) for k in identity_keys):
                    blocked='position_changed_since_protection'
                else:
                    p=positions[symbol];fill=self._fill(context,'SELL',None,D(p['quantity']))
                    if fill:
                        qty=D(fill['quantity']);fee=D(fill['fee_tmn']);basis=D(p['basis_tmn'])*qty/D(p['quantity'])
                        proceeds=D(fill['notional_tmn'])-fee;pnl=proceeds-basis
                        fill['realized_pnl_tmn']=str(pnl);meta['cash']=str(D(meta['cash'])+proceeds);meta['fees']=str(D(meta['fees'])+fee);meta['realized_pnl']=str(D(meta['realized_pnl'])+pnl)
                        left=D(p['quantity'])-qty
                        if left:
                            p.update({'quantity':str(left),'basis_tmn':str(D(p['basis_tmn'])-basis),'mark_price':str(context['book']['bid'])})
                            db.execute('UPDATE positions SET value=? WHERE symbol=?',(packed(p),symbol))
                        else:db.execute('DELETE FROM positions WHERE symbol=?',(symbol,))
                    else:blocked='exit_minimum_or_insufficient_depth'
            elif action!='HOLD':raise ValueError('Unsupported action')
            for key in ['cash','fees','realized_pnl']:db.execute('UPDATE meta SET value=? WHERE key=?',(packed(meta[key]),key))
            meta,positions,equity=self._update_risk(db)
            event={'decision_id':decision_id,'at':int(now),'symbol':symbol,'mode':'paper','decision':decision,'fill':fill,'blocked':blocked,'context_sha256':checksum(context),'cash_tmn':meta['cash'],'market_data_status':'invalid_for_execution' if data_error else 'valid','market_data_reason':data_error}
            previous=db.execute('SELECT event_hash FROM decisions ORDER BY rowid DESC LIMIT 1').fetchone();previous=previous[0] if previous else '0'*64
            event_hash=checksum({'previous':previous,'event':event})
            db.execute('INSERT INTO decisions VALUES(?,?,?,?,?)',(decision_id,int(now),packed(event),previous,event_hash))
            if fill:db.execute('INSERT INTO fills(decision_id,payload) VALUES(?,?)',(decision_id,packed(fill)))
            return {'duplicate':False,'event':event,'event_hash':event_hash}

    def _fill(self,context,side,budget,quantity):
        rules=context['market'];decimals=int(rules['amount_decimals'])
        if not 0<=decimals<=18:raise ValueError('Invalid quantity precision')
        increment=D(1).scaleb(-decimals);fee=D(self.config.fee_rate);impact=D(self.config.slippage_rate)
        source=context['book']['asks' if side=='BUY' else 'bids']
        best=D(str(context['book']['ask' if side=='BUY' else 'bid']))
        bound=best*(1+D(self.config.max_book_impact_fraction) if side=='BUY' else 1-D(self.config.max_book_impact_fraction))
        levels=[]
        for level in source:
            price=D(str(level['price']));size=D(str(level['quantity']))
            if not price.is_finite() or not size.is_finite() or price<=0 or size<=0:raise ValueError('Invalid book depth')
            if (side=='BUY' and price<best) or (side=='SELL' and price>best):raise ValueError('Depth contradicts best quote')
            executed=price*(1+impact if side=='BUY' else 1-impact)
            if (side=='BUY' and executed<=bound) or (side=='SELL' and executed>=bound):
                levels.append((executed,size))
        levels.sort(key=lambda x:x[0],reverse=side=='SELL')
        if not levels:return None
        wanted=budget/levels[0][0] if side=='BUY' else quantity
        wanted=wanted.quantize(increment,rounding=ROUND_DOWN)
        original_wanted=wanted
        # Round the aggregate down, then walk actual levels without inventing liquidity.
        available=sum((size for _,size in levels),D(0));wanted=min(wanted,available).quantize(increment,rounding=ROUND_DOWN)
        if wanted<=0:return None
        remaining=wanted;total=D(0);filled=D(0)
        for price,size in levels:
            amount=min(remaining,size)
            if side=='BUY':amount=min(amount,max(D(0),(budget-total)/price))
            total+=amount*price;filled+=amount;remaining-=amount
            if remaining<=0 or (side=='BUY' and total>=budget):break
        rounded=filled.quantize(increment,rounding=ROUND_DOWN)
        if rounded<filled:
            # Recalculate the exact aggregate cost for the executable rounded quantity.
            remaining=rounded;total=D(0)
            for price,size in levels:
                amount=min(remaining,size);total+=amount*price;remaining-=amount
                if remaining<=0:break
        filled=rounded
        minimum=max(D(str(rules['minimum_notional'])),D(self.config.min_order_tmn))
        if filled<=0 or filled<D(str(rules['minimum_quantity'])) or total<minimum:return None
        return {'kind':'simulated_orderbook_fill','side':side,'symbol':context['symbol'],'quantity':str(filled),'average_price':str(total/filled),
                'notional_tmn':str(total),'fee_tmn':str(total*fee),'partial':filled<original_wanted,
                'observed_book_at':context['book']['observed_at'],'maximum_impact_fraction':self.config.max_book_impact_fraction,'assumptions':'captured depth + configured impact, quote fees; no queue/latency model'}

    def verify(self):
        with self.connect() as db:
            previous='0'*64;count=0;expected_fills=[];events=[]
            for raw,prev,event_hash in db.execute('SELECT payload,previous_hash,event_hash FROM decisions ORDER BY rowid'):
                event=json.loads(raw)
                if prev!=previous or checksum({'previous':prev,'event':event})!=event_hash:raise ValueError('Decision audit mismatch')
                if event['fill']:expected_fills.append((event['decision_id'],event['fill']))
                events.append(event);previous=event_hash;count+=1
            actual_fills=[(id,json.loads(raw)) for id,raw in db.execute('SELECT decision_id,payload FROM fills ORDER BY id')]
            if actual_fills!=expected_fills:raise ValueError('Fill table differs from decision audit')
            meta,positions,equity=self._read(db);cash=D(self.config.initial_cash_tmn);fees=D(0);pnl=D(0);ledger={}
            if meta['initial_cash']!=self.config.initial_cash_tmn or meta['config']!=json.loads(packed(self.config.__dict__)):raise ValueError('Initial capital or config differs')
            for event in events:
                fill=event['fill']
                if fill:
                    cost=D(fill['notional_tmn']);fee=D(fill['fee_tmn']);qty=D(fill['quantity']);avg=D(fill['average_price']);symbol=fill['symbol']
                    if symbol!=event['symbol'] or not valid_tmn_symbol(symbol) or (self.config.research_universe and symbol not in self.config.research_universe):raise ValueError('Fill symbol differs from decision')
                    if not all(x.is_finite() for x in (cost,fee,qty,avg)) or min(cost,qty,avg)<=0 or fee<0:raise ValueError('Invalid fill amounts')
                    if avg!=cost/qty or fee!=cost*D(self.config.fee_rate):raise ValueError('Fill price/fee does not reconcile')
                    fees+=fee
                    if fill['side']=='BUY':
                        if symbol in ledger:raise ValueError('Overlapping entry in ledger')
                        cash-=cost+fee
                        terms=fill.get('position_terms',{})
                        if D(terms.get('entry_price','0'))!=avg:raise ValueError('Entry price differs from fill')
                        ledger[symbol]={'quantity':qty,'basis':cost+fee,'entry_price':avg,'terms':terms}
                    elif fill['side']=='SELL':
                        if symbol not in ledger or qty>ledger[symbol]['quantity']:raise ValueError('Exit exceeds ledger position')
                        held=ledger[symbol];basis=held['basis']*qty/held['quantity'];realized=cost-fee-basis
                        if realized!=D(fill['realized_pnl_tmn']):raise ValueError('Realized PnL differs from reconstructed basis')
                        cash+=cost-fee;pnl+=realized;held['quantity']-=qty;held['basis']-=basis
                        if not held['quantity']:
                            if held['basis']!=0:raise ValueError('Closed position retains basis')
                            del ledger[symbol]
                    else:raise ValueError('Invalid ledger side')
                    if cash<0:raise ValueError('Negative paper cash')
                if cash!=D(event['cash_tmn']):raise ValueError('Decision cash differs from reconstructed ledger')
            if cash!=D(meta['cash']) or fees!=D(meta['fees']) or pnl!=D(meta['realized_pnl']):raise ValueError('Paper accounting does not reconcile')
            if set(ledger)!=set(positions):raise ValueError('Position set differs from fills')
            for symbol,held in ledger.items():
                position=positions[symbol]
                if position.get('symbol')!=symbol:raise ValueError('Position symbol differs from fills')
                if held['quantity']!=D(position['quantity']):raise ValueError('Position quantity differs from fills')
                if held['basis']!=D(position['basis_tmn']):raise ValueError('Position basis differs from fills')
                if held['entry_price']!=D(position['entry_price']):raise ValueError('Position entry price differs from fills')
                if any(position.get(key)!=value for key,value in held['terms'].items()):raise ValueError('Position terms differ from entry audit')
                mark=D(position['mark_price'])
                if not mark.is_finite() or mark<=0:raise ValueError('Invalid position mark')
            # Decimal division can leave sub-attotoman rounding after multiple partial exits.
            # All per-fill and persisted-ledger checks above remain exact.
            if not ledger and abs((cash-D(meta['initial_cash']))-pnl)>D('1e-18'):
                raise ValueError('Flat cash profit differs from realized PnL')
            peak=D(meta['peak'])
            if not peak.is_finite() or peak<max(D(meta['initial_cash']),equity) or type(meta['halted']) is not bool:
                raise ValueError('Invalid persisted risk state')
            if (peak-equity)/peak>=D(self.config.max_drawdown) and not meta['halted']:
                raise ValueError('Drawdown limit exceeded without halt')
        return {'decision_count':count,'audit_head':previous,'verified':True,'accounting_reconciled':True}
