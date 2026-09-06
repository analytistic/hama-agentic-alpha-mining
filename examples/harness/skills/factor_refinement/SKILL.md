Inspect the current factor pool and formulate one falsifiable refinement
hypothesis. Use historical factor backtests to compare candidate expressions,
check redundancy against the existing pool, and submit exactly one supported
add, remove, or replace operation. Write every expression in Qlib syntax: raw
fields use a `$` prefix and lagged values use `Ref`, for example
`Ref($close, 5) / $close - 1`. Do not use bare fields or `delay(...)` syntax.
