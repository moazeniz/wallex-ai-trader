from dataclasses import dataclass
from datetime import datetime
import math,re

@dataclass(frozen=True)
class ChallengePolicy:
    approved_markets: tuple[str,...] = ()
    official_cutoff_utc: str | None = None
    settlement_buffer_seconds: int = 1800
    account_verified: bool = False
    fee_verified: bool = False

    def __post_init__(self):
        if not isinstance(self.account_verified,bool) or not isinstance(self.fee_verified,bool):raise ValueError("Verification flags must be JSON booleans")
        if any(not isinstance(s,str) or not re.fullmatch(r"[A-Z0-9]+TMN",s) for s in self.approved_markets):raise ValueError("Approved markets must be TMN spot symbols")
        if self.settlement_buffer_seconds<60:raise ValueError("Settlement buffer must be at least 60 seconds")
        if self.official_cutoff_utc:
            value=datetime.fromisoformat(self.official_cutoff_utc.replace("Z","+00:00"))
            if value.tzinfo is None:raise ValueError("Trading cutoff must include timezone")

    def gate(self, symbol, now):
        if not math.isfinite(now):raise ValueError("Invalid current timestamp")
        reasons=[]
        if symbol not in self.approved_markets:reasons.append("market_not_confirmed_for_credited_subaccount")
        if not self.account_verified:reasons.append("credited_subaccount_unverified")
        if not self.fee_verified:reasons.append("account_fee_unverified")
        if self.official_cutoff_utc is None:reasons.append("official_trading_cutoff_unset")
        else:
            cutoff=datetime.fromisoformat(self.official_cutoff_utc.replace("Z","+00:00")).timestamp()
            if now>=cutoff:reasons.append("official_cutoff_passed")
            elif now>=cutoff-self.settlement_buffer_seconds:reasons.append("settlement_window_no_new_entries")
        return {"new_entry_allowed":not reasons,"reasons":reasons}
