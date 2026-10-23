import pytest
from experiments.wallex_agent.universe import derive_universe


def market(base,volume=1e9,spread=.001):
    return {'baseAsset':base,'quoteAsset':'TMN','stats':{'bidPrice':str(100*(1-spread/2)),'askPrice':str(100*(1+spread/2)),'24h_quoteVolume':str(volume)}}


def test_chat_list_is_exclusion_not_whitelist_and_generic_assets_rank_by_liquidity():
    markets={'BTCTMN':market('BTC',1e9),'PEPETMN':market('PEPE',2e9),'SOLTMN':market('SOL',3e9)}
    result=derive_universe(markets,'untrusted participant prose\nSOL/TMN\nSOL/USDT\n',limit=16)
    assert [x['symbol'] for x in result['selected']]==['PEPETMN','BTCTMN']
    assert result['excluded_pair_count']==2 and not result['live_account_eligibility_verified']


def test_bad_spreads_wrong_currency_and_thin_markets_are_not_promoted():
    markets={'DOGETMN':market('DOGE',1e9,.05),'ADATMN':market('ADA',1e5),'BTCUSDT':market('BTC',9e9),'SOLTMN':market('SOL',2e9)}
    result=derive_universe(markets,'')
    assert [x['symbol'] for x in result['selected']]==['SOLTMN']


def test_model_symbols_cannot_be_shell_commands_and_limit_is_bounded():
    markets={'BTC;echoTMN':market('BTC',9e9),'STMN':market('S',1e9)}
    assert [x['symbol'] for x in derive_universe(markets,'')['selected']]==['STMN']
    with pytest.raises(ValueError):derive_universe(markets,'',limit=1000)
