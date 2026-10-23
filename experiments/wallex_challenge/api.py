from dataclasses import asdict
from pathlib import Path
import json, time, urllib.request, urllib.parse, urllib.error
from .core import Candle, Market, atomic_json, normalize_history

class PublicAPI:
    # Deliberately restricted to these read-only public routes; no order or wallet methods.
    ALLOWED={"/v1/markets","/v1/depth","/v1/udf/history","/v1/udf/time"}
    def get(self,path,params=None):
        if path not in self.ALLOWED:raise ValueError("Unsupported public route")
        url="https://api.wallex.ir"+path+("?"+urllib.parse.urlencode(params) if params else "")
        for attempt in range(3):
            try:
                request=urllib.request.Request(url,headers={"User-Agent":"WallexChallengeResearch/0.1"})
                with urllib.request.urlopen(request,timeout=25) as response:return json.load(response)
            except urllib.error.HTTPError as e:
                if e.code not in [429,500,502,503,504] or attempt==2:raise
                time.sleep(min(8,2**attempt))
            except (TimeoutError,urllib.error.URLError):
                if attempt==2:raise
                time.sleep(2**attempt)

    def market(self,symbol):
        response=self.get("/v1/markets")
        if not response.get("success"):raise ValueError("Markets response unsuccessful")
        info=response["result"]["symbols"][symbol]
        # Legacy metadata uses stepSize as decimal places, not an increment.
        return Market(symbol,int(info["stepSize"]),str(info["minQty"]),str(info["minNotional"])),info

    def history(self,symbol,days,interval,save_dir=None):
        if interval not in [60,900,3600]:raise ValueError("Supported intervals: 60,900,3600")
        if not 1<=days<=60:raise ValueError("History range must be 1..60 days")
        exchange_time=int(self.get("/v1/udf/time"))
        if abs(time.time()-exchange_time)>120:raise ValueError("Exchange/local clock difference exceeds 120s")
        end=exchange_time//interval*interval
        # Verify actual timestamps: a 15-minute request has returned minute candles.
        resolution=str(interval//60)
        payload={"s":"ok",**{k:[] for k in ["t","o","h","l","c","v"]}}
        root=Path(save_dir) if save_dir else None
        if root:root.mkdir(parents=True,exist_ok=True)
        # Bound request ranges: the API returns no_data for oversized minute requests.
        step=7*86400
        start=end-days*86400
        request_count=0
        for lower in range(start,end,step):
            upper=min(lower+step,end)
            part=self.get("/v1/udf/history",{"symbol":symbol,"resolution":resolution,"from":lower,"to":upper-1})
            if root:atomic_json(root/("raw-history-%02d.json" % request_count),{"from":lower,"to":upper-1,"response":part})
            request_count+=1
            if part.get("s")!="ok":raise ValueError("History page did not return s=ok; inspect saved raw response")
            keys=["t","o","h","l","c","v"]
            if len({len(part[k]) for k in keys})!=1:raise ValueError("History page arrays are misaligned")
            if any(t<lower or t>=upper for t in part["t"]):raise ValueError("History API returned timestamps outside requested page")
            for key in keys:payload[key].extend(part[key])
            if upper<end:time.sleep(.5)
        rows,quality=normalize_history(payload,interval,end)
        quality.update({"symbol":symbol,"fetched_at":exchange_time,"requested_days":days,"requested_resolution":resolution,"request_count":request_count})
        if root:atomic_json(root/"quality.json",quality)
        return rows,quality

    def book(self,symbol):
        response=self.get("/v1/depth",{"symbol":symbol})
        if not response.get("success"):raise ValueError("Depth response unsuccessful")
        result=response["result"]
        asks=sorted((float(x["price"]),float(x["quantity"])) for x in result["ask"])
        bids=sorted(((float(x["price"]),float(x["quantity"])) for x in result["bid"]),reverse=True)
        if not asks or not bids or bids[0][0]<=0 or asks[0][0]<=bids[0][0]:raise ValueError("Invalid or crossed order book")
        return {"bid":bids[0][0],"ask":asks[0][0],"spread_fraction":(asks[0][0]-bids[0][0])/((asks[0][0]+bids[0][0])/2),
            "observed_at":int(time.time()),"source_timestamp_available":False,
            "note":"REST book has no source timestamp; observation is not proof of quote freshness"}
