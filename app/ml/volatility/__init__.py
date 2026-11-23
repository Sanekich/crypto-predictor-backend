"""Volatility models: "will the price move by at least X% (either way) within N hours?".

Direction of BTC is close to unpredictable from public data, but the size of the
next move is not: volatility clusters. These models answer the size question and
can serve as a filter for direction signals. Code here, artifacts in
artifacts/volatility/models, API under /volatility.
"""
