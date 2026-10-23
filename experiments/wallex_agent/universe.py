"""Research eligibility from supplied exclusions and current public liquidity, not account approval."""
import re,math,time,hashlib


def derive_universe(markets,exclusion_text,limit=16,min_volume_tmn=500_000_000,max_spread=.01):
    if not 1<=limit<=64:raise ValueError('Universe limit1..64')
    exclusions={base+quote for base,quote in re.findall(r'^([A-Z0-9]+)/(TMN|USDT)\s*$',exclusion_text,re.M)}
    candidates=[]
    for symbol,item in markets.items():
        if not re.fullmatch(r'[A-Z0-9]+TMN',symbol) or item.get('quoteAsset')!='TMN' or symbol in exclusions:continue
        try:
            stats=item['stats'];bid=float(stats['bidPrice']);ask=float(stats['askPrice']);volume=float(stats['24h_quoteVolume'])
            if not all(math.isfinite(x) for x in [bid,ask,volume]) or not 0<bid<ask:continue
            spread=(ask-bid)/((bid+ask)/2)
            if volume<min_volume_tmn or spread>max_spread:continue
        except (KeyError,TypeError,ValueError,ZeroDivisionError):continue
        candidates.append({'symbol':symbol,'base_asset':item.get('baseAsset'),'quote_currency':'TMN','volume24h_tmn':volume,'public_snapshot_spread':spread})
    candidates.sort(key=lambda x:(-x['volume24h_tmn'],x['symbol']))
    return {'schema_version':1,'observed_at':time.time(),'purpose':'research_only','live_account_eligibility_verified':False,
            'filter':{'min_volume24h_tmn':min_volume_tmn,'max_public_spread':max_spread,'limit':limit},
            'exclusion_source_sha256':hashlib.sha256(exclusion_text.encode()).hexdigest(),'excluded_pair_count':len(exclusions),
            'selected':candidates[:limit],'filtered_candidate_count':len(candidates),
            'warnings':['Posted participant exclusion list is not credited-account authorization',
                        'Current liquidity/eligibility selection is not historical point-in-time universe evidence',
                        'Execution rechecks fresh depth, actual costs and account market rules']}
