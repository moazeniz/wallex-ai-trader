// Fixed, read-only adapter for @mathieuc/tradingview 4.0.0. No credentials.
import { getCandles } from '@mathieuc/tradingview/data';
import { readFileSync } from 'node:fs';

try {
  const input = readFileSync(0, 'utf8');
  if (input.length > 4096) throw new Error('input_too_large');
  const query = JSON.parse(input);
  if (typeof query.symbol !== 'string' || query.symbol.trim() !== query.symbol || !/^BINANCE:[A-Z0-9]{1,20}USDT$/.test(query.symbol)) throw new Error('invalid_symbol');
  if (query.timeframe !== '60' || !Number.isInteger(query.count) || query.count < 1 || query.count > 500) throw new Error('invalid_query');
  if (!Number.isInteger(query.timeoutMs) || query.timeoutMs < 1000 || query.timeoutMs > 28000) throw new Error('invalid_timeout');
  if (!Number.isFinite(query.closedBefore) || query.closedBefore <= 0) throw new Error('invalid_cutoff');
  const metadata = JSON.parse(readFileSync(new URL('./node_modules/@mathieuc/tradingview/package.json', import.meta.url), 'utf8'));
  if (metadata.version !== '4.0.0') throw new Error('unexpected_package_version');
  const candles = await getCandles({symbol: query.symbol, timeframe: '60', count: query.count, timeoutMs: query.timeoutMs,
                                    signal: AbortSignal.timeout(query.timeoutMs)});
  const rows = candles.map(row => ({t: row.time, o: row.open, h: row.high, l: row.low, c: row.close, v: row.volume}));
  if (!rows.length || rows.length > query.count) throw new Error('invalid_candle_count');
  let previous;
  for (const row of rows) {
    if (!Number.isInteger(row.t) || row.t % 3600 || (previous !== undefined && row.t - previous !== 3600)) throw new Error('invalid_cadence');
    if (!Object.values(row).every(Number.isFinite) || Math.min(row.o, row.h, row.l, row.c) <= 0 || row.v < 0 || row.h < Math.max(row.o, row.c) || row.l > Math.min(row.o, row.c)) throw new Error('invalid_ohlcv');
    previous = row.t;
  }
  // Python performs independent cutoff, cadence, quality and freshness validation.
  process.stdout.write(JSON.stringify({source: 'tradingview', symbol: query.symbol, quote_currency: 'USDT',
    conversion_to_tmn_applied: false, package_version: metadata.version, fetched_at: Date.now() / 1000,
    runtime: {node: process.version}, candles: rows}));
} catch (error) {
  // Messages are intentionally generic; no remote response, session, or environment dump.
  process.stderr.write(JSON.stringify({error: 'tradingview_request_failed', kind: error?.code ?? error?.name ?? 'Error'}));
  process.exitCode = 1;
}
