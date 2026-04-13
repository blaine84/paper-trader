# 🚀 TASK: Implement Tier 1 Trading System Enhancements

You are working on an existing Python-based multi-agent trading system ("paper-trader") with the architecture defined in USER_GUIDE.md.

Your goal is to implement **Tier 1 enhancements**:

1. Position-level Edge Score
2. Case Library Similarity Matching
3. Portfolio-Level Risk Engine

The system already has:

* SQLite database (`db/paper_trader.db`)
* Case library (`cases` table)
* PM decision engine
* Trade validation pipeline
* Position tracking (`positions` table)

⚠️ IMPORTANT:

* Do NOT break existing functionality
* All new features must be additive and backward-compatible
* Keep logic modular and testable
* Prefer deterministic logic over LLM where possible

---

# 1. EDGE SCORE SYSTEM

## Objective

Create a continuous scoring system (0.0–1.0) that determines:

* whether a trade is allowed
* how large the position should be

## Implementation

### Create module:

`core/edge_score.py`

### Function:

```python
def compute_edge_score(signal: dict, case_stats: dict, similarity_stats: dict) -> float:
```

### Inputs:

* `signal` (from Analyst):

  * strength (weak/moderate/strong)
  * confidence (low/medium/high)
  * setup_type
  * indicators (VWAP, RSI, EMA, etc.)

* `case_stats`:

  * win_rate (setup_type + regime)
  * sample_size

* `similarity_stats`:

  * similarity_winrate
  * similarity_avg_r

### Scoring Logic (initial weights):

```python
score = (
    0.30 * normalize_winrate(case_stats.win_rate) +
    0.25 * normalize_similarity(similarity_stats.similarity_winrate) +
    0.20 * map_strength(signal.strength) +
    0.15 * map_confidence(signal.confidence) +
    0.10 * confluence_score(signal.indicators)
)
```

Clamp output:

```python
return max(0.0, min(1.0, score))
```

### Add helper functions:

* `normalize_winrate`
* `map_strength`
* `map_confidence`
* `confluence_score` (e.g., above VWAP + EMA trend alignment)

---

## Integration into PM Flow

Modify PM decision pipeline:

BEFORE trade execution:

```python
edge_score = compute_edge_score(...)

if edge_score < 0.4:
    reject_trade("edge_score_too_low")

position_size *= edge_score
```

Store in DB:

* add column `edge_score` to `trades`

---

# 2. SIMILARITY MATCHING ENGINE

## Objective

Find similar past trades and compute performance statistics.

## Create module:

`core/similarity.py`

### Function:

```python
def find_similar_cases(signal: dict, db_conn) -> list[dict]:
```

### Matching Criteria:

* setup_type (exact match)
* market_regime
* RSI within ±10
* same bias (LONG/SHORT)
* VWAP position (above/below)

Return top 5–10 matches ranked by similarity score.

---

### Function:

```python
def compute_similarity_stats(cases: list[dict]) -> dict:
```

Return:

```python
{
    "similarity_winrate": float,
    "similarity_avg_r": float,
    "sample_size": int
}
```

---

## Integration

In PM pipeline BEFORE edge score:

```python
similar_cases = find_similar_cases(signal, db)
similarity_stats = compute_similarity_stats(similar_cases)
```

Pass into edge score function.

---

# 3. PORTFOLIO RISK ENGINE

## Objective

Prevent overexposure across correlated positions.

## Create module:

`core/portfolio_risk.py`

---

### Define exposure buckets:

```python
BUCKETS = {
    "index": ["SPY", "QQQ", "IWM"],
    "semis": ["NVDA", "AMD"],
    "ev": ["TSLA", "LCID"],
}
```

---

### Function:

```python
def compute_portfolio_risk(open_positions: list[dict]) -> dict:
```

Return:

```python
{
    "total_risk": float,
    "bucket_exposure": {
        "index": float,
        "semis": float,
        ...
    }
}
```

---

### Function:

```python
def validate_portfolio_risk(new_trade: dict, portfolio_state: dict) -> bool:
```

Rules:

* max 50% exposure per bucket
* max total risk threshold (e.g. 1.5x normal exposure)

---

## Integration

Before trade execution:

```python
risk_ok = validate_portfolio_risk(new_trade, portfolio_state)

if not risk_ok:
    reject_trade("portfolio_risk_exceeded")
```

---

# 4. DATABASE CHANGES

Add columns to `trades` table:

```sql
ALTER TABLE trades ADD COLUMN edge_score REAL;
ALTER TABLE trades ADD COLUMN similarity_winrate REAL;
ALTER TABLE trades ADD COLUMN similarity_sample_size INTEGER;
```

---

# 5. LOGGING

Log all decisions:

```text
EDGE SCORE: 0.62 (winrate=0.55, similarity=0.60, strength=strong)
SIMILAR CASES: 7 matches, winrate=57%
PORTFOLIO RISK: index=0.35, semis=0.20
```

---

# 6. TESTING

Create tests:

### Edge Score:

* strong signal + high winrate → score > 0.7
* weak + low winrate → score < 0.4

### Similarity:

* returns relevant cases
* empty set handled gracefully

### Portfolio Risk:

* blocks overexposed trades
* allows safe trades

---

# 7. CONSTRAINTS

* No LLM calls in these modules
* Must run fast (called frequently)
* Deterministic outputs
* Clean separation of concerns

---

# 8. DELIVERABLES

* New modules:

  * `core/edge_score.py`
  * `core/similarity.py`
  * `core/portfolio_risk.py`
* PM pipeline integration
* DB migration script
* Unit tests

---

# GOAL

After implementation:

* Every trade has a quantified edge score
* Decisions are influenced by historical similarity
* Portfolio risk is actively managed

This should upgrade the system from rule-based decisions to **adaptive, data-driven execution**.
