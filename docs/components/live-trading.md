# Live trading client (`experiments/wallex_arena/live.py`)

`live.py` places real orders on Wallex. It stays **disabled** until a person turns it on. The arena engine uses it
only for accounts marked `"live"` in the arena config (`"live": "dry_run"` uses a key-less client that never sends);
`live_account.py` adapts it to the paper-account interface and `watchdog.py` uses it for emergency stops.

- `WallexPrivateAPI` is a thin authenticated REST client. It uses stdlib `urllib`, and you can
  swap in a fake transport for tests.
- `LiveExecutor` offers the paper-account actions: `market_buy`, `market_sell`, `place_limit`,
  `cancel`, `sync` and `balances`. Every order goes through the safety checks below.
- Tests are in `tests/test_wallex_arena_live.py`. They use a fake transport and never touch the network.

## Endpoints used

Base URL is `https://api.wallex.ir`. Auth header is `x-api-key`.

| Purpose | Request | Docs |
|---|---|---|
| Create order | `POST /v1/account/orders`. JSON body: `symbol, side, type, price, quantity, client_id` (strings). `stop_Price` only for STOP types. Success is HTTP 201 | [spot-create-order](https://developers.wallex.ir/docs/spot-create-order), [generated/post-account-orders](https://developers.wallex.ir/docs/generated/post-account-orders) |
| Cancel (used) | `DELETE /v1/account/orders/{client_id}` | [spot-cancel-order](https://developers.wallex.ir/docs/spot-cancel-order), [generated/delete-orders-client](https://developers.wallex.ir/docs/generated/delete-orders-client) |
| Cancel (alternative) | `DELETE /v1/account/orders` with body `{"clientOrderId": ...}` | [generated/delete-account-orders](https://developers.wallex.ir/docs/generated/delete-account-orders) |
| Order detail | `GET /v1/account/orders/{client_id}` | [spot-single-order-detail](https://developers.wallex.ir/docs/spot-single-order-detail), [generated/get-orders-client](https://developers.wallex.ir/docs/generated/get-orders-client) |
| Open orders | `GET /v1/account/openOrders?symbol=&page=&per_page=`. `result` is an object whose values are arrays | [spot-active-orders](https://developers.wallex.ir/docs/spot-active-orders), [generated/get-open-order](https://developers.wallex.ir/docs/generated/get-open-order) |
| Balances | `GET /v1/account/balances`. Returns `result.balances.{ASSET}.{value, locked}` | [generated/get-account-balances](https://developers.wallex.ir/docs/generated/get-account-balances) |
| Credit sub-account balances | `GET /v1/account/balances-detail` with header `sub-account-client-id`. Returns `result.{ASSET}.{total, freeze, available}` | [generated/sub-account-balances-details](https://developers.wallex.ir/docs/generated/sub-account-balances-details), swagger `subAccountBalancesDetails` |
| Account fee | `GET /v1/account/fee`. Returns `result.{SYMBOL}.{makerFeeRate, takerFeeRate, recent_days_sum}` | [basic-services-account-fee](https://developers.wallex.ir/docs/basic-services-account-fee), [generated/get-account-fees](https://developers.wallex.ir/docs/generated/get-account-fees) |
| Market metadata (public) | `GET /v1/markets`. `tickSize` and `stepSize` are decimal places | basic-services-markets |

Pages that informed the design but are not called:
- [spot-order-list](https://developers.wallex.ir/docs/spot-order-list) (`GET /v1/account/orders`, the order history)
- [generated/active-sub-accounts-list](https://developers.wallex.ir/docs/generated/active-sub-accounts-list) (`GET /sub-accounts`, gives the `clientId` of each credit account)
- [trade-credit-intro](https://developers.wallex.ir/docs/trade-credit-intro), [trade-credit-active-packages](https://developers.wallex.ir/docs/trade-credit-active-packages) and [trade-credit-activate-sub-account](https://developers.wallex.ir/docs/trade-credit-activate-sub-account)
- the trade-credit swagger at `https://developers.wallex.ir/assets/files/trade-credit-ff5c60b5b97b77a5b2c2e4800f07729c.yaml`

Note that [generated/get-orders](https://developers.wallex.ir/docs/generated/get-orders) is `GET /margin-trade/v1/orders`. It covers margin orders, not spot.

## How orders are sent

- **"Market" orders are bounded LIMIT orders.**
  - A buy is sent at `max_price` rounded **down** to the tick, so it never goes above the bound.
    The best ask is always on a tick, so an ask at or below `max_price` still fills.
  - A sell is sent at `min_price` rounded **up** to the tick.
  - Quantity is always rounded **down** to the step size. A buy also sets aside the 0.30% taker fee
    in TMN, so `qty × price × 1.003 ≤ budget`.
  - After `ioc_wait` (1.5 s by default), the client re-reads the order. If part of it is still
    open, it cancels that part. This copies the paper account's taker-only fill. No resting
    remainder is left by accident.
- **`place_limit` makes a resting order.** A buy price rounds down and a sell price rounds up.
- **Every order carries our own `client_id`.** The format is `ARENA_<ms>_<8 hex>`, using only
  A–Z, 0–9 and `_`. This id is the order id used by `cancel` and `sync`, because Wallex cancels
  and looks up orders only by clientOrderId.
- **A create-order POST is never retried.** If the result is unclear (timeout, dropped
  connection, 408 or 5xx, or a 2xx without an order in it), the client looks the order up with
  `GET /v1/account/orders/{client_id}` up to 3 times. If that does not settle it, it raises
  `AmbiguousOrder`. The order stays tracked as `unknown`, and `sync()` keeps checking it. After
  120 s of 404 answers, `sync()` marks it `NOT_PLACED`. GET requests are retried on network
  errors. DELETE requests are not.
- **`sync()` polls every order that is still open or unknown.** It returns only the new fills,
  each with `qty, price, fee, fee_asset, fee_tmn` and `maker`/`taker`. It takes them from
  `fills[]`, or works them out from `executedQty`, `executedSum` and `fee` when `fills` is empty.
- **The client throttles order creation** to 20 per 10 s, which is the limit the docs give.
- **The API key never appears in output.** It is not logged and not shown in `repr` or in dry-run
  requests (`x-api-key: ***`). Logs hold only the method, URL and body. With `log_path` set, every
  action goes to a hash-chained `DecisionLog`. With `state_path` set, order tracking survives a restart.

## Safety switches

Each order method (`market_buy`, `market_sell`, `place_limit`, `cancel`) raises `LiveRefused`
**before any network call** if any of these is true:

1. The file `/var/lib/wallex-arena/LIVE_DISABLED` exists. This check wins over everything else.
2. The environment variable `WALLEX_LIVE_ENABLED` is not exactly `"1"`. This is the default state.
3. No API client is configured.
4. The symbol is not a TMN market in `/v1/markets`, is in `excluded_markets.json`, or is outside
   the optional `allowed=` set.
5. After rounding, the notional is below 100,000 TMN or the market `minNotional`, or the quantity
   is below `minQty`. You cannot lower the minimum below 100,000.
6. The notional is above `max_order_tmn` (2,000,000 TMN by default). Larger orders must be split.

`dry_run=True` never sends anything. It builds requests with a client that has no key, logs the
exact request and returns a simulated `DRY_RUN` result. Because of that, it still works while the
switches are off.

`sync()` and `balances()` only read data. The switches do not block them, but they need a key.
One more thing to know: when `LIVE_DISABLED` is set, `cancel()` is blocked too. To flatten the
account, remove the file, or use the Wallex UI.

## How to enable (a person does this)

1. Put the credit account's **own** API key into `/etc/wallex/wallex.env`.
   The file is owned by root and has mode 0600 and holds `WALLEX_API_KEY` and
   `WALLEX_SUBACCOUNT_ID`. The sub-account id is the `clientId` from `GET /sub-accounts` or from
   the activation response. Never commit or print this file.
2. Build the client:

   ```python
   api = WallexPrivateAPI.from_env_file()
   ex = LiveExecutor(api, feed.meta, max_order_tmn=..., log_path=..., state_path=...)
   ```

3. Check the wiring with reads only:
   - Confirm `ex.balances()` shows the credit capital, not the main account.
   - Confirm `api.account_fee()` matches the fees you expect.
4. Try one order first with `dry_run=True` and read the logged request.
5. Remove `/var/lib/wallex-arena/LIVE_DISABLED` if it exists, then set `WALLEX_LIVE_ENABLED=1` in
   the environment of the process that will trade. To stop, run `touch /var/lib/wallex-arena/LIVE_DISABLED`.
   The next order call is refused.

## Open questions in the docs

1. **Does the credit account need a sub-account header on order calls?**
   - No spot order page lists a header: not create, cancel, detail or openOrders. The trade-credit
     swagger has no order endpoints at all.
   - The intro says each credit account has its own wallet and its own API key. The likely answer
     is that you call order endpoints with that account's key and no header.
   - The client defaults to this (`sub_account_on_orders=False`), with a switch to send the header.
   - **Risk:** if the main account's key is used and Wallex ignores the header, orders hit the main
     wallet. That is why step 3 above checks balances first.
2. **How is the header spelled?** The swagger declares `sub-account-client-id` and the client uses
   that. The swagger description says `Sub-Account-Client-id`, which is the same name in a different
   case. The connect-to-wallet prose page says `sub_account_client_id`, with underscores. You can
   change it with `sub_account_header=`.
3. **Which asset is the fee taken in?** Each fill reports a `feeAsset`, but the docs do not say
   which asset it is. A buy might pay in the base coin or in TMN. The client handles both. Buys
   set the fee aside in TMN to be safe. The fee level of the credit account is unknown, so read
   `/v1/account/fee` with the credit key.
4. **What order statuses exist?** The docs never list them. They mention NEW, FILLED, CANCELED and
   FINISHED. The client trusts the `active` flag and treats FILLED, CANCELED, CANCELLED, FINISHED,
   EXPIRED and REJECTED as final.
5. **Does the POST answer already include fills?** The docs do not say. This is why the market
   path waits `ioc_wait` and re-reads the order before cancelling.
6. **What are the `openOrders` keys?** The examples show `additionalProp1`, so the key is probably
   the symbol. The client flattens all values.
7. **Can you cancel by exchange order id?** No. Cancel and lookup work only by clientOrderId. The
   `client_id` path variable is called `clientOrderId` in the parameter table, and the example uses
   a hyphen (`test-clientId`), which create-order does not allow. Neither the maximum length nor
   lowercase support is documented.
8. **Does `balances.value` include `locked`?** The docs do not say. The client assumes it does,
   and computes available as `value − locked`.
9. **Which minimum applies?** `/v1/markets` reports `minNotional` = 50,000. The organizers require
   100,000, and the client enforces the stricter one. `isMarketTypeEnable` is false on some markets
   (for example SHIBTMN). The client always sends LIMIT orders, so this does not matter.

## Update 2026-10-22
- developers.wallex.ir/docs/spot-intro says trade-credit (sub-account) spot calls must use that sub-account API key
  **and** the header sub-account-client-id: <clientId> (clientId from GET /sub-accounts). The client now sends the
  header on order calls by default (sub_account_on_orders=True).
- Order placement limit: 20 requests / 10 s (x-ratelimit-limit / x-ratelimit-remaining headers). MARKET orders are
  auto-cancelled when the book is too thin or the order would move price more than 2% (we use bounded LIMIT orders).
- Fees: /v1/account/fee reports maker 0.25% / taker 0.30% on TMN markets; other participants report that spot may be
  fee-free on credit accounts. Read fee and feeAsset from the first real fills and adjust the fee model accordingly.
- Official Wallex MCP server: https://api.wallex.ir/mcp (Streamable HTTP, x-api-key header, ~5 req rate limit).
