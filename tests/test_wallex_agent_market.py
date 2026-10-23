"""Offline data-contract, time-causality and optional-adapter checks; no network."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
import subprocess

from experiments.wallex_challenge.core import Candle, Market
from experiments.wallex_agent.market import (MarketConfig, MarketContext, TradingViewSource,
    _closed_rows, aggregate_four_hour, order_book, technical_summary, tradingview_symbol, NODE, BRIDGE)

NOW = 720000.0


def candles(n=120, now=NOW):
    end = int(now) // 3600 * 3600
    return [Candle(end - (n - i) * 3600, 100 + i / 10, 102 + i / 10, 99 + i / 10, 101 + i / 10, 5 + i % 7)
            for i in range(n)]


def depth():
    return {"success": True, "result": {"bid": [{"price": "100", "quantity": "2"}, {"price": "99", "quantity": "3"}],
                                        "ask": [{"price": "102", "quantity": "4"}, {"price": "101", "quantity": "5"}]}}


class FakeAPI:
    def market(self, symbol):
        return Market(symbol, 8, "0.00000001", "50000"), {"symbol": symbol, "stepSize": 8}

    def history(self, symbol, days, interval):
        return candles(), {"requested_days": days, "requested_resolution": "60", "source_interval_seconds": 3600, "fetched_at": NOW}

    def get(self, path, params):
        assert path == "/v1/depth"
        return depth()


class FakeTV:
    def fetch(self, symbol, now):
        return {"status": "ok", "source": "tradingview", "symbol": tradingview_symbol(symbol,MarketConfig()), "quote_currency": "USDT",
                "conversion_to_tmn_applied": False, "candles": [asdict(row) for row in candles()]}


class SnapshotChecks(TestCase):
    def context(self, api=None, tv=None):
        return MarketContext(api=api or FakeAPI(), tradingview=tv or FakeTV())

    def test_snapshot_serializes_with_rules_full_book_and_provenance(self):
        result = self.context().snapshot("BTCTMN", NOW)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["market"], {"symbol": "BTCTMN", "amount_decimals": 8, "minimum_quantity": "0.00000001", "minimum_notional": "50000"})
        self.assertEqual(result["book"]["bids"][1]["quantity"], 3)
        self.assertEqual(result["book"]["asks"][0]["price"], 101)
        self.assertFalse(result["book"]["source_timestamp_available"])
        self.assertEqual(result["provenance"]["history_route"], "/v1/udf/history")
        self.assertEqual(len(result["provenance"]["candles_sha256"]), 64)
        self.assertFalse(result["provenance"]["exchange_order_submission"])

    def test_toman_candles_and_usdt_candles_are_distinct(self):
        result = self.context().snapshot("ETHTMN", NOW)
        self.assertEqual(result["candle_quality"]["quote_currency"], "TMN")
        self.assertEqual(result["market_metadata"]["base_currency"], "ETH")
        self.assertEqual(result["cross_market"]["quote_currency"], "USDT")
        self.assertFalse(result["cross_market"]["conversion_to_tmn_applied"])

    def test_malformed_research_symbols_rejected_before_network(self):
        api = FakeAPI()
        with patch.object(api, "market") as market:
            for symbol in ("BTCUSDT", "BTCTMN;echo hi", "../DOGETMN", "binance:BTCUSDT", "DOGETMN\\n", "دوژTMN", "TMN"):
                with self.assertRaises(ValueError): self.context(api).snapshot(symbol, NOW)
            market.assert_not_called()

    def test_market_symbol_mismatch_rejected(self):
        api = FakeAPI()
        with patch.object(api, "market", return_value=(Market("ETHTMN", 8, "0", "1"), {})):
            with self.assertRaisesRegex(ValueError, "symbol mismatch"): self.context(api).snapshot("BTCTMN", NOW)

    def test_stale_primary_fails(self):
        with self.assertRaisesRegex(ValueError, "stale"): self.context().snapshot("BTCTMN", NOW + 10000)

    def test_gap_and_insufficient_primary_fail(self):
        for rows in (candles(50), candles()[:40] + candles()[41:]):
            api = FakeAPI()
            with patch.object(api, "history", return_value=(rows, {})):
                with self.assertRaises(ValueError): self.context(api).snapshot("BTCTMN", NOW)

    def test_partial_and_future_bars_dropped(self):
        api = FakeAPI()
        future = [Candle(int(NOW), 100, 999999, 1, 888888, 1e9)]
        with patch.object(api, "history", return_value=(candles() + future, {})):
            result = self.context(api).snapshot("BTCTMN", NOW + 120)
        self.assertEqual(len(result["candles"]), 120)
        self.assertEqual(result["candle_quality"]["unfinished_candles_dropped_at_snapshot"], 1)
        self.assertLessEqual(result["technical"]["last_closed_at"], NOW + 120)

    def test_optional_failure_does_not_mask_primary(self):
        tv = FakeTV()
        with patch.object(tv, "fetch", side_effect=RuntimeError("sensitive private message")):
            result = self.context(tv=tv).snapshot("BTCTMN", NOW)
        self.assertEqual(result["cross_market"]["status"], "unavailable")
        self.assertNotIn("sensitive", json.dumps(result))
        self.assertEqual(len(result["candles"]), 120)

    def test_invalid_cutoffs_fail(self):
        for value in (float("nan"), float("inf"), -1, 0):
            with self.assertRaises(ValueError): self.context().snapshot("BTCTMN", value)


class BookChecks(TestCase):
    def test_empty_crossed_locked_nan_negative_and_zero_books_fail(self):
        variants = []
        for bad in ("nan", "inf", "0", "-3"):
            for field in ("price", "quantity"):
                payload = depth(); payload["result"]["bid"][0][field] = bad; variants.append(payload)
        payload = depth(); payload["result"]["bid"] = []; variants.append(payload)
        payload = depth(); payload["result"]["ask"] = []; variants.append(payload)
        payload = depth(); payload["result"]["bid"][0]["price"] = "102"; variants.append(payload)
        payload = depth(); payload["result"]["bid"][0]["price"] = "101"; variants.append(payload)
        payload = depth(); payload["result"]["bid"][1]["price"] = "100"; variants.append(payload)
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises(ValueError): order_book(payload, NOW)

    def test_best_quote_sorting_and_spread(self):
        result = order_book(depth(), NOW)
        self.assertAlmostEqual(result["spread_fraction"], 1 / 100.5)
        self.assertEqual(result["bid"], 100)
        self.assertEqual(result["ask"], 101)
        self.assertEqual(result["observed_at"], NOW)


class IndicatorChecks(TestCase):
    def test_future_perturbation_cannot_change_features(self):
        rows = candles()
        with_future = rows + [Candle(int(NOW), 1, 1e15, .000001, 1e10, 1e20)]
        self.assertEqual(technical_summary(rows, NOW), technical_summary(with_future, NOW))

    def test_four_hour_buckets_complete_and_utc_aligned(self):
        rows = candles(119)
        grouped = aggregate_four_hour(rows)
        self.assertTrue(all(row.t % 14400 == 0 for row in grouped))
        self.assertTrue(all(row.t + 14400 <= NOW for row in grouped))
        self.assertEqual(grouped[-1].v, sum(row.v for row in rows[-4:]))
        self.assertEqual(len(grouped), 29)

    def test_monotonic_price_has_rsi_100_positive_returns_and_trend(self):
        result = technical_summary(candles(), NOW)["one_hour"]
        self.assertEqual(result["rsi14"], 100)
        self.assertGreater(result["return_24h"], 0)
        self.assertEqual(result["ema_trend"], "up")
        self.assertGreater(result["atr14"], 0)
        self.assertGreaterEqual(result["realized_volatility_24h"], 0)

    def test_flat_price_rsi_50_and_zero_volume_explicitly_missing_ratio(self):
        rows = [replace(row, o=100, h=100, l=100, c=100, v=0) for row in candles()]
        result = technical_summary(rows, NOW)["one_hour"]
        self.assertEqual(result["rsi14"], 50)
        self.assertEqual(result["atr14"], 0)
        self.assertIsNone(result["relative_volume_24h"])

    def test_unsorted_duplicate_nonaligned_and_nonfinite_closed_bars_fail(self):
        variants = [candles()[::-1], candles() + [candles()[-1]]]
        rows = candles(); rows[1] = replace(rows[1], t=rows[1].t + 1); variants.append(rows)
        rows = candles(); rows[-1] = replace(rows[-1], c=float("nan")); variants.append(rows)
        rows = candles(); rows[-1] = replace(rows[-1], h=0); variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValueError): _closed_rows(rows, NOW)


class TradingViewChecks(TestCase):
    def payload(self, rows=None):
        return {"symbol": "BINANCE:BTCUSDT", "quote_currency": "USDT", "package_version": "4.0.0", "fetched_at": NOW,
                "runtime": {"node": "v24"}, "candles": [asdict(row) for row in (rows or candles())]}

    def completed(self, payload=None):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload or self.payload()), stderr="")

    def test_fixed_process_arguments_timeout_and_env_no_credentials(self):
        with patch("experiments.wallex_agent.market.subprocess.run", return_value=self.completed()) as run:
            result = TradingViewSource().fetch("BTCTMN", NOW)
        self.assertEqual(result["status"], "ok")
        args, kwargs = run.call_args
        self.assertEqual(args[0], [str(NODE), str(BRIDGE)])
        self.assertLessEqual(kwargs["timeout"], 30)
        self.assertNotIn("shell", kwargs)
        self.assertEqual(set(kwargs["env"]), {"PATH", "LANG"})
        self.assertEqual(json.loads(kwargs["input"])["symbol"], "BINANCE:BTCUSDT")

    def test_timeout_child_failure_missing_node_malformed_json_degrade(self):
        exceptions = [subprocess.TimeoutExpired(["node"], 25), FileNotFoundError("node")]
        for error in exceptions:
            with patch("experiments.wallex_agent.market.subprocess.run", side_effect=error):
                self.assertEqual(TradingViewSource().fetch("BTCTMN", NOW)["status"], "unavailable")
        for completed in (SimpleNamespace(returncode=1, stdout="", stderr="secret"), SimpleNamespace(returncode=0, stdout="not json", stderr="")):
            with patch("experiments.wallex_agent.market.subprocess.run", return_value=completed):
                result = TradingViewSource().fetch("BTCTMN", NOW)
                self.assertEqual(result["status"], "unavailable")
                self.assertNotIn("secret", json.dumps(result))

    def test_source_symbol_currency_package_mismatch_rejected(self):
        for key, value in (("symbol", "BINANCE:ETHUSDT"), ("quote_currency", "TMN"), ("package_version", "3.5")):
            payload = self.payload(); payload[key] = value
            with patch("experiments.wallex_agent.market.subprocess.run", return_value=self.completed(payload)):
                self.assertEqual(TradingViewSource().fetch("BTCTMN", NOW)["status"], "unavailable")

    def test_partial_tv_bar_dropped_and_stale_tv_degrades(self):
        payload = self.payload(candles() + [Candle(int(NOW), 100, 102, 99, 101, 5)])
        with patch("experiments.wallex_agent.market.subprocess.run", return_value=self.completed(payload)):
            result = TradingViewSource().fetch("BTCTMN", NOW + 120)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["quality"]["unfinished_candles_dropped"], 1)
        with patch("experiments.wallex_agent.market.subprocess.run", return_value=self.completed()):
            self.assertEqual(TradingViewSource().fetch("BTCTMN", NOW + 10000)["status"], "unavailable")

    def test_gap_in_tv_data_rejected(self):
        payload = self.payload(candles()[:10] + candles()[11:])
        with patch("experiments.wallex_agent.market.subprocess.run", return_value=self.completed(payload)):
            self.assertEqual(TradingViewSource().fetch("BTCTMN", NOW)["status"], "unavailable")

    def test_cache_immutable_and_invalidated_at_hour_boundary(self):
        source = TradingViewSource()
        with patch("experiments.wallex_agent.market.subprocess.run", return_value=self.completed()) as run:
            result = source.fetch("BTCTMN", NOW + 1)
            result["candles"][0]["c"] = -123
            second = source.fetch("BTCTMN", NOW + 2)
            self.assertTrue(second["cache_hit"])
            self.assertGreater(second["candles"][0]["c"], 0)
            source.fetch("BTCTMN", NOW + 3601)
            self.assertEqual(run.call_count, 2)

    def test_disabled_never_starts_node(self):
        source = TradingViewSource(MarketConfig(tradingview_enabled=False))
        with patch("experiments.wallex_agent.market.subprocess.run") as run:
            self.assertEqual(source.fetch("BTCTMN", NOW)["status"], "disabled")
            run.assert_not_called()

    def test_configuration_bounds(self):
        for kwargs in ({"history_days": 1}, {"minimum_hourly_bars": 0}, {"tradingview_count": 501},
                       {"tradingview_timeout_seconds": 31}, {"tradingview_timeout_seconds": float("nan")},
                       {"tradingview_enabled": "false"}, {"tradingview_cache_seconds": 7200}, {"history_days": 8.5}, {"tradingview_count": 200.5}):
            with self.assertRaises(ValueError): MarketConfig(**kwargs)


if __name__ == "__main__": main()


class GeneralMarketChecks(TestCase):
    def test_actual_metadata_gate_and_non_btc_asset(self):
        api=FakeAPI()
        tv=TradingViewSource(MarketConfig(tradingview_enabled=False))
        result=MarketContext(api=api,tradingview=tv).snapshot('ADATMN',NOW)
        self.assertEqual(result['market']['symbol'],'ADATMN')
        self.assertEqual(result['market_metadata']['base_currency'],'ADA')
        self.assertEqual(result['market_metadata']['quote_currency'],'TMN')
        self.assertEqual(result['cross_market']['symbol'],'BINANCE:ADAUSDT')
        self.assertEqual(result['cross_market']['quote_currency'],'USDT')
        self.assertFalse(result['cross_market']['conversion_to_tmn_applied'])
        with patch.object(api,'market',side_effect=KeyError('not listed')):
            with self.assertRaisesRegex(ValueError,'not listed'):
                MarketContext(api=api,tradingview=tv).snapshot('FAKETMN',NOW)

    def test_arbitrary_valid_binance_target_and_optional_missing_feed(self):
        source=TradingViewSource()
        payload={'symbol':'BINANCE:LUNCUSDT','quote_currency':'USDT','package_version':'4.0.0',
                 'fetched_at':NOW,'candles':[asdict(row) for row in candles()]}
        completed=SimpleNamespace(returncode=0,stdout=json.dumps(payload),stderr='')
        with patch('experiments.wallex_agent.market.subprocess.run',return_value=completed) as run:
            result=source.fetch('LUNCTMN',NOW)
            self.assertEqual(result['status'],'ok')
            self.assertEqual(json.loads(run.call_args.kwargs['input'])['symbol'],'BINANCE:LUNCUSDT')
        with patch('experiments.wallex_agent.market.subprocess.run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='SYMBOL_ERROR')):
            self.assertEqual(TradingViewSource().fetch('LUNCTMN',NOW)['status'],'unavailable')

    def test_explicit_valid_symbol_alias_and_invalid_mapping(self):
        config=MarketConfig(tradingview_symbols={'RENDERTMN':'BINANCE:RNDRUSDT'})
        self.assertEqual(tradingview_symbol('RENDERTMN',config),'BINANCE:RNDRUSDT')
        self.assertEqual(tradingview_symbol('WTMN',config),'BINANCE:WUSDT')
        for mapping in ({'BTCTMN;cmd':'BINANCE:BTCUSDT'}, {'BTCTMN':'BINANCE:BTCUSDT;cmd'}, {'BTCTMN':'BTCUSDT'}, {'BTCTMN':'BINANCE:BTCTMN'}):
            with self.assertRaises(ValueError):MarketConfig(tradingview_symbols=mapping)
        config.tradingview_symbols['RENDERTMN']='BINANCE:RNDRUSDT;cmd'
        with self.assertRaises(ValueError):tradingview_symbol('RENDERTMN',config)



class ExecutionRefreshChecks(TestCase):
    def test_refresh_only_public_depth_and_preserves_closed_analysis(self):
        api=FakeAPI();tv=FakeTV();market=MarketContext(api=api,tradingview=tv)
        original=market.snapshot('BTCTMN',NOW)
        with patch.object(api,'get',return_value=depth()) as get, patch.object(api,'history') as history, patch.object(api,'market') as metadata, patch.object(tv,'fetch') as global_fetch:
            result=market.refresh_execution_context(original,NOW+65)
            get.assert_called_once_with('/v1/depth',{'symbol':'BTCTMN'})
            history.assert_not_called();metadata.assert_not_called();global_fetch.assert_not_called()
        self.assertEqual(result['observed_at'],NOW+65)
        self.assertEqual(result['book']['observed_at'],NOW+65)
        self.assertEqual(result['analysis_observed_at'],NOW)
        for key in ('candles','technical','candle_quality','cross_market','market','market_metadata'):
            self.assertEqual(result[key],original[key])
        self.assertEqual(result['provenance']['candles_sha256'],original['provenance']['candles_sha256'])
        self.assertEqual(result['provenance']['snapshot_cutoff'],NOW)
        self.assertNotIn('analysis_observed_at',original)
        result['candles'][0]['c']=-1
        self.assertGreater(original['candles'][0]['c'],0)
        json.dumps(result,allow_nan=False)

    def test_repeated_refresh_keeps_original_analysis_observation(self):
        market=MarketContext(api=FakeAPI(),tradingview=FakeTV())
        original=market.snapshot('BTCTMN',NOW)
        first=market.refresh_execution_context(original,NOW+65)
        second=market.refresh_execution_context(first,NOW+130)
        self.assertEqual(second['analysis_observed_at'],NOW)
        self.assertEqual(second['observed_at'],NOW+130)
        self.assertEqual(second['technical']['last_closed_at'],original['technical']['last_closed_at'])

    def test_bad_market_identity_or_missing_source_rejected_before_request(self):
        api=FakeAPI();market=MarketContext(api=api,tradingview=FakeTV())
        original=market.snapshot('BTCTMN',NOW)
        variants=[]
        broken=deepcopy(original);broken['symbol']='BTCUSDT';variants.append(broken)
        broken=deepcopy(original);broken['market']['symbol']='ETHTMN';variants.append(broken)
        broken=deepcopy(original);broken['market_metadata']['source']='invented';variants.append(broken)
        broken=deepcopy(original);broken['market']['amount_decimals']=99;variants.append(broken)
        with patch.object(api,'get') as get:
            for broken in variants:
                with self.assertRaises(ValueError):market.refresh_execution_context(broken,NOW+65)
            get.assert_not_called()

    def test_refresh_reuses_strict_book_validation_and_propagates_failure(self):
        api=FakeAPI();market=MarketContext(api=api,tradingview=FakeTV())
        original=market.snapshot('BTCTMN',NOW)
        bad=depth();bad['result']['bid'][0]['quantity']='nan'
        with patch.object(api,'get',return_value=bad):
            with self.assertRaises(ValueError):market.refresh_execution_context(original,NOW+65)
        with patch.object(api,'get',side_effect=TimeoutError('offline')):
            with self.assertRaises(TimeoutError):market.refresh_execution_context(original,NOW+65)
        self.assertEqual(original['observed_at'],NOW)



class AnalysisBarRolloverChecks(TestCase):
    def test_depth_refresh_after_hour_rollover_cannot_authorize_stale_analysis_buy(self):
        from experiments.wallex_agent.risk import RiskConfig,resolve_action
        market=MarketContext(api=FakeAPI(),tradingview=FakeTV())
        original=market.snapshot('BTCTMN',NOW+3590)
        # A quote-only refresh preserves the completed bar at NOW after the next
        # UTC hour has closed. Fresh book timestamps must not refresh its analysis.
        refreshed=market.refresh_execution_context(original,NOW+3610)
        self.assertEqual(refreshed['technical']['last_closed_at'],NOW)
        self.assertEqual(refreshed['observed_at'],NOW+3610)
        forecast={'predicted_gross_return':.05,'horizon_hours':8,'training_cutoff':NOW-10000,
                  'calibration_cutoff':NOW-5000,'feature_cutoff':NOW}
        analysts=[{'status':'ok','proposed_action':'BUY','confidence':.9,'risk_flags':[]} for _ in range(3)]
        state={'positions':{},'halted':False}
        # Keep the fixture spread within the execution guard.
        valid_depth={'success':True,'result':{'bid':[{'price':'100','quantity':'100'}],
                                            'ask':[{'price':'100.1','quantity':'100'}]}}
        with patch.object(market.api,'get',return_value=valid_depth):
            before=market.refresh_execution_context(original,NOW+3590)
            after=market.refresh_execution_context(original,NOW+3610)
        allowed=resolve_action(before,forecast,analysts,state,NOW+3590,RiskConfig())
        self.assertEqual(allowed['action'],'BUY')
        blocked=resolve_action(after,forecast,analysts,state,NOW+3610,RiskConfig())
        self.assertEqual(blocked['action'],'HOLD')
        self.assertIn('completed_analysis_bar_not_current',blocked['reason'])

    def test_fresh_quote_stop_works_when_analysis_bar_is_old(self):
        from experiments.wallex_agent.risk import RiskConfig,resolve_action
        market=MarketContext(api=FakeAPI(),tradingview=FakeTV())
        original=market.snapshot('BTCTMN',NOW)
        fresh=market.refresh_execution_context(original,NOW+3610)
        state={'halted':False,'positions':{'BTCTMN':{'stop_price':'110','take_price':'130','expires_at':NOW+86400}}}
        proposal=resolve_action(fresh,{},[],state,NOW+3610,RiskConfig())
        self.assertEqual(proposal['action'],'SELL')
        self.assertEqual(proposal['reason'],['stop_loss'])
        self.assertTrue(proposal['forced_risk_exit'])
