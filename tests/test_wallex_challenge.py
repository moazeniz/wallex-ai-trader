"""Offline risk, causality, provenance and data-contract checks; no network calls."""
from dataclasses import asdict, replace
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
import json, math

from experiments.wallex_challenge.core import (Candle, Settings, Market, PaperBroker, Audit,
    normalize_history, validate_candles, features, fit_model, predict, decision, simulate, verify_audit)
from experiments.wallex_challenge.api import PublicAPI
from experiments.wallex_challenge.policy import ChallengePolicy


def candles(n=400):
    rows=[]
    for i in range(n):
        price=100000*(1+0.015*math.sin(i/8)+i/20000)
        rows.append(Candle(i*900,price,price*1.001,price*.999,price,10000+i%19))
    return rows


def forced_buy_model():
    return {"last_label_time":0,"weights":[math.log(4)]+[0]*7,"means":[0]*7,"scales":[1]*7,
        "mean_positive_return":0.05,"mean_nonpositive_return":-0.01}


class DataChecks(TestCase):
    def payload(self,n=45):
        return {"s":"ok","t":[i*60 for i in range(n)],"o":[100]*n,"h":[102]*n,"l":[99]*n,"c":[101]*n,"v":[2]*n}

    def test_minute_response_aggregates_to_requested_interval(self):
        rows,q=normalize_history(self.payload(),900,2700)
        self.assertEqual([b.t for b in rows],[0,900,1800]);self.assertEqual(rows[0].v,30)
        self.assertEqual(q['source_interval_seconds'],60)

    def test_unfinished_bucket_never_enters_model(self):
        rows,q=normalize_history(self.payload(),900,1801)
        self.assertEqual(len(rows),2)

    def test_missing_minutes_are_not_imputed(self):
        d=self.payload()
        for key in ['t','o','h','l','c','v']:d[key].pop(20)
        rows,q=normalize_history(d,900,2700)
        self.assertEqual([b.t for b in rows],[1800]);self.assertEqual(q['incomplete_buckets_dropped'],1)
        self.assertEqual(q['older_candles_before_last_gap'],1)

    def test_duplicate_and_bad_ohlcv_rejected(self):
        d=self.payload();d['t'][3]=d['t'][2]
        with self.assertRaises(ValueError):normalize_history(d,900,2700)
        d=self.payload();d['h'][2]=98
        with self.assertRaises(ValueError):normalize_history(d,900,2700)
        d=self.payload();d['c'][2]=float('nan')
        with self.assertRaises(ValueError):normalize_history(d,900,2700)

    def test_misaligned_or_gapped_normalized_data_rejected(self):
        rows=candles(50);rows.pop(30)
        with self.assertRaises(ValueError):validate_candles(rows,900)

    def test_wrong_granularity_cannot_be_silently_upsampled(self):
        d=self.payload();d['t']=[i*3600 for i in range(45)]
        with self.assertRaises(ValueError):normalize_history(d,900,200000)


class CausalityChecks(TestCase):
    def test_future_price_changes_do_not_change_past_features(self):
        rows=candles();before=features(rows,200)
        modified=rows[:201]+[replace(b,o=b.o*10,h=b.h*10,l=b.l*10,c=b.c*10,v=b.v*10) for b in rows[201:]]
        self.assertEqual(before,features(modified,200))

    def test_training_is_identical_after_future_data_is_changed(self):
        rows=candles();first=fit_model(rows,280,Settings())
        modified=rows[:280]+[replace(b,o=b.o*10,h=b.h*10,l=b.l*10,c=b.c*10) for b in rows[280:]]
        self.assertEqual(first,fit_model(modified,280,Settings()))
        self.assertLess(first['last_label_time'],rows[280].t)

    def test_training_label_overlap_refuses_backtest(self):
        rows=candles();m=forced_buy_model();m['last_label_time']=rows[30].t
        with self.assertRaises(ValueError):simulate(rows,m,Market('BTCTMN',8,'0.00000001','50000'),Settings(),30)

    def test_ai_predictions_change_with_learned_weights(self):
        m=forced_buy_model();p=predict(m,[0]*7);m['weights'][1]=1
        self.assertGreater(predict(m,[2]+[0]*6),p)

    def test_expected_edge_must_cover_costs(self):
        m=forced_buy_model();m['mean_positive_return']=0.0001;m['mean_nonpositive_return']=-0.0001
        self.assertEqual(decision(m,.99,Settings())[0],'HOLD')


class RiskChecks(TestCase):
    def broker(self,settings=None):return PaperBroker(Market('BTCTMN',8,'0.00000001','50000'),settings or Settings())

    def test_capital_cap_and_positive_cash(self):
        b=self.broker();fill=b.buy(100000,1e9,30,27000)
        self.assertEqual(fill['side'],'BUY');self.assertGreaterEqual(b.cash,0)
        self.assertLessEqual(b.quantity*D(fill['price']),D('1000000'))
        self.assertIn('blocked',b.buy(100000,1e9,31,27900))

    def test_cannot_short_or_buy_below_minimum(self):
        b=self.broker();self.assertEqual(b.sell(100000,0,'test')['blocked'],'no_position')
        self.assertEqual(b.buy(100000,0.001,0,0)['blocked'],'minimum_order_or_liquidity')

    def test_quantity_rounds_down_to_market_increment(self):
        m=Market('ETHTMN',5,'0.00001','50000')
        self.assertEqual(m.round_quantity(D('1.12345678')),D('1.12345'))

    def test_fees_reconcile_exactly_with_realized_cash(self):
        b=self.broker();b.buy(100000,1e9,0,0);sell=b.sell(110000,900,'test')
        self.assertEqual(b.cash,D('5000000')+D(sell['realized_pnl']))
        self.assertEqual(b.quantity,0);self.assertGreater(b.fees,0)

    def test_exit_liquidity_can_leave_unsettled_position(self):
        b=self.broker();b.buy(100000,1e9,0,0);q=b.quantity
        result=b.sell(100000,900,'test',volume=.00001)
        self.assertIn('blocked',result);self.assertEqual(b.quantity,q)

    def test_drawdown_halts_new_entries(self):
        b=self.broker();b.buy(100000,1e9,0,0);b.mark(50000)
        self.assertTrue(b.halted);b.sell(50000,900,'halt')
        self.assertEqual(b.buy(100000,1e9,2,1800)['blocked'],'already_in_position_or_halted')

    def test_both_stop_and_take_in_same_bar_use_stop_first(self):
        rows=[Candle(i*900,100000,100100,99900,100000,1e9) for i in range(40)]
        rows[30]=Candle(27000,100000,110000,90000,100000,1e9)
        report,events,curve=simulate(rows,forced_buy_model(),Market('BTCTMN',8,'0.00000001','50000'),Settings(),30)
        self.assertEqual(events[0]['fills'][1]['reason'],'stop_loss')
        self.assertLess(D(events[0]['fills'][1]['realized_pnl']),0)

    def test_end_of_test_settlement_is_explicit(self):
        rows=[Candle(i*900,100000,100100,99900,100000,1e9) for i in range(36)]
        report,events,curve=simulate(rows,forced_buy_model(),Market('BTCTMN',8,'0.00000001','50000'),Settings(),30)
        self.assertTrue(report['settled_to_tmn']);self.assertEqual(events[-1]['fills'][-1]['reason'],'end_of_test_settlement')
        self.assertLess(report['net_return_percent'],0)

    def test_invalid_risk_configuration_is_rejected(self):
        for kwargs in [{'fee_rate':'nan'},{'max_position_fraction':'1'},{'initial_cash':'-1'},{'sell_probability':.9}]:
            with self.assertRaises(ValueError):Settings(**kwargs)

    def test_usdt_cannot_be_misreported_as_toman(self):
        with self.assertRaises(ValueError):Market('BTCUSDT',8,'0.0001','1')


class ProvenanceChecks(TestCase):
    def test_audit_survives_restart_and_detects_tampering(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'log.jsonl';Audit(p).append({'reason':'AI BUY'})
            Audit(p).append({'reason':'AI SELL'});self.assertEqual(len(list(verify_audit(p))),2)
            p.write_text(p.read_text().replace('AI BUY','manual BUY'))
            with self.assertRaises(ValueError):list(verify_audit(p))

    def test_public_client_cannot_reach_order_or_wallet_route(self):
        for path in ['/v1/account/orders','/v1/account/balances','https://evil.example']:
            with self.assertRaises(ValueError):PublicAPI().get(path)

    def test_challenge_gate_fails_closed_until_account_is_verified(self):
        gate=ChallengePolicy().gate('BTCTMN',0)
        self.assertFalse(gate['new_entry_allowed']);self.assertEqual(len(gate['reasons']),4)

    def test_entry_gate_stops_before_cutoff_for_settlement(self):
        policy=ChallengePolicy(('BTCTMN',),'2026-10-23T20:00:00Z',1800,True,True)
        from datetime import datetime
        cutoff=datetime.fromisoformat('2026-10-23T20:00:00+00:00').timestamp()
        self.assertTrue(policy.gate('BTCTMN',cutoff-1801)['new_entry_allowed'])
        self.assertIn('settlement_window_no_new_entries',policy.gate('BTCTMN',cutoff-1800)['reasons'])
        self.assertIn('official_cutoff_passed',policy.gate('BTCTMN',cutoff)['reasons'])
        self.assertFalse(policy.gate('ETHTMN',cutoff-5000)['new_entry_allowed'])

    def test_cutoff_requires_timezone(self):
        with self.assertRaises(ValueError):ChallengePolicy(official_cutoff_utc='2026-10-23T20:00:00')


class ShadowRuntimeChecks(TestCase):
    def setup_run(self,root):
        from experiments.wallex_challenge.core import digest
        from experiments.wallex_challenge.__main__ import source_hash
        m=forced_buy_model();(root/'model.json').write_text(json.dumps(m))
        (root/'report.json').write_text(json.dumps({'market':{'symbol':'BTCTMN'},'model_sha256':digest(m),'algorithm_sha256':source_hash(),'settings':asdict(Settings())}))
        from types import SimpleNamespace
        return SimpleNamespace(output=str(root/'state'),model=str(root/'model.json'),symbol='BTCTMN',settings=None,policy=None,once=True,poll_seconds=300)

    def test_shadow_restart_does_not_duplicate_closed_bar(self):
        from experiments.wallex_challenge.__main__ import shadow
        import contextlib,io
        class Fake:
            def history(self,*a):return candles(48),{'source':'fixture'}
            def book(self,*a):return {'bid':99999,'ask':100001}
        with TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.setup_run(root)
            with patch('experiments.wallex_challenge.__main__.PublicAPI',return_value=Fake()),patch('time.time',return_value=48*900),contextlib.redirect_stdout(io.StringIO()):
                shadow(args);shadow(args)
            events=list(verify_audit(root/'state/observations.jsonl'))
            self.assertEqual(len(events),1);self.assertEqual(events[0]['payload']['execution'],'disabled')
            self.assertFalse(events[0]['payload']['policy_gate']['new_entry_allowed'])

    def test_stale_bar_and_changed_model_cannot_produce_decision(self):
        from experiments.wallex_challenge.__main__ import shadow
        class Fake:
            def history(self,*a):return candles(48),{}
        with TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.setup_run(root)
            with patch('experiments.wallex_challenge.__main__.PublicAPI',return_value=Fake()),patch('time.time',return_value=1000000):
                with self.assertRaisesRegex(ValueError,'stale'):shadow(args)
            events=list(verify_audit(root/'state/observations.jsonl'))
            self.assertTrue(all(x['payload']['kind']=='collector_error' for x in events))
            d=json.loads((root/'model.json').read_text());d['weights'][0]=0;(root/'model.json').write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'hash'):shadow(args)

    def test_another_collector_cannot_use_same_output(self):
        from experiments.wallex_challenge.__main__ import shadow
        import fcntl
        with TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.setup_run(root);(root/'state').mkdir()
            with (root/'state/collector.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError,'Another collector'):shadow(args)

    def test_invalid_verification_flag_cannot_enable_account(self):
        with self.assertRaises(ValueError):ChallengePolicy(account_verified='false')
        with self.assertRaises(ValueError):ChallengePolicy(approved_markets=('BTCUSDT',))
        with self.assertRaises(ValueError):ChallengePolicy().gate('BTCTMN',float('nan'))

if __name__=='__main__':main()

