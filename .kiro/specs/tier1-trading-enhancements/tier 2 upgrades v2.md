Design Document: Tier 1 Trading Enhancements (v2)
Overview

This design enhances the Paper Trader system with three deterministic modules that enable adaptive, data-driven decision-making:

Edge Score Engine — continuous trade quality scoring (0.0–1.0)
Similarity Matching Engine — historical pattern recognition
Portfolio Risk Engine — cross-position exposure control

These modules integrate into the Portfolio Manager (PM) pipeline to:

Filter low-quality trades
Dynamically size positions
Prevent correlated overexposure
Leverage historical experience
🔁 Updated System Flow
flowchart TD
    A[Analyst Signal] --> B[Similarity Engine]
    B --> C[Similarity Stats]
    C --> D[Case Stats]
    D --> E[Edge Score Engine]
    E --> F{edge_score >= threshold?}

    F -->|No| X[Reject Trade]
    F -->|Yes| G[Scale Position Size]

    G --> H[Portfolio Risk Engine]
    H --> I{Risk OK?}

    I -->|No| X
    I -->|Yes| J[Existing Validation]
    J --> K[Execute Trade]
1. EDGE SCORE ENGINE (Enhanced)
🎯 Objective

Replace binary decision rules with a continuous edge model that:

gates trades
scales position size
adapts over time
🔧 Updated Formula
edge_score =
    w1 * setup_winrate +
    w2 * similarity_winrate +
    w3 * signal_strength +
    w4 * signal_confidence +
    w5 * confluence_score +
    w6 * similarity_quality
Default Weights
w1 = 0.25   # setup/regime winrate
w2 = 0.20   # similarity winrate
w3 = 0.15   # signal strength
w4 = 0.10   # signal confidence
w5 = 0.15   # indicator confluence
w6 = 0.15   # similarity quality (NEW)
🆕 New: Similarity Quality Score

Prevents overconfidence from weak samples.

similarity_quality = min(1.0, similarity_sample_size / 10)

Example:

2 cases → 0.2 confidence
10+ cases → full weight
🆕 Edge Score Gating Rules
if case_sample_size >= 10 and setup_winrate < 0.35:
    reject_trade()

if edge_score < 0.4:
    reject_trade()
🆕 Position Sizing
scaled_size = base_size * edge_score

Optional cap:

scaled_size = min(scaled_size, base_size * 1.2)
2. SIMILARITY MATCHING ENGINE (Enhanced)
🎯 Objective

Move from coarse grouping → context-aware matching

🔧 Matching Criteria (Weighted)

Instead of strict filtering, use scoring:

similarity_score =
    w1 * setup_match +
    w2 * regime_match +
    w3 * rsi_distance +
    w4 * vwap_alignment +
    w5 * trend_alignment
Improvements:
RSI uses distance, not exact bucket
VWAP = boolean match
EMA trend alignment included
No longer requires ALL conditions to match
🔄 Ranking

Return top N (5–10) cases by similarity score

🆕 Output Stats
{
    similarity_winrate,
    similarity_avg_r,
    similarity_sample_size,
    similarity_confidence   # NEW
}

Where:

similarity_confidence = min(1.0, sample_size / 10)
🆕 Fallback Behavior
if sample_size == 0:
    skip similarity weighting (do NOT penalize)
3. PORTFOLIO RISK ENGINE (Enhanced)
🎯 Objective

Prevent hidden leverage and correlated drawdowns

🔧 Exposure Model
exposure = position_value / total_equity
🆕 Bucket System (Expanded)
BUCKETS = {
    "index": ["SPY", "QQQ", "IWM", "DIA"],
    "semis": ["NVDA", "AMD", "TSM", "INTC"],
    "ev": ["TSLA", "LCID", "RIVN"],
    "mega_growth": ["NVDA", "TSLA", "META", "AMZN"],
}

Symbols can belong to multiple buckets.

🆕 Risk Rules
Bucket Limits
max_bucket_exposure = 0.50
Total Exposure
max_total_exposure = 1.2–1.5 (configurable)
🆕 Adaptive Risk Throttling
if recent_losses >= 3:
    reduce_position_size(25–50%)
🆕 Risk Output
{
    total_exposure,
    bucket_exposure,
    risk_score   # NEW
}
4. PM PIPELINE (Updated)
New Execution Flow
# 1. Similarity
similar_cases = find_similar_cases(...)
similarity_stats = compute_similarity_stats(...)

# 2. Edge Score
edge_score = compute_edge_score(...)

if edge_score < 0.4:
    reject

# 3. Scale size
quantity *= edge_score

# 4. Portfolio Risk
risk_ok = validate_portfolio_risk(...)

if not risk_ok:
    reject

# 5. Existing validation
validate_trade(...)
5. DATABASE CHANGES (Extended)
ALTER TABLE trades ADD COLUMN edge_score REAL;
ALTER TABLE trades ADD COLUMN similarity_winrate REAL;
ALTER TABLE trades ADD COLUMN similarity_sample_size INTEGER;
ALTER TABLE trades ADD COLUMN similarity_confidence REAL;
6. LOGGING (Enhanced)
EDGE SCORE: 0.64
  setup_winrate=0.58 (n=12)
  similarity_winrate=0.60 (n=8)
  similarity_confidence=0.80
  confluence=0.80

PORTFOLIO RISK:
  total_exposure=0.42
  semis=0.25
  index=0.17

DECISION:
  size_scaled=0.64
  status=EXECUTED
7. FAILURE MODES (Improved)
Component	Failure	Behavior
Edge Score	fails	❌ reject trade (fail closed)
Similarity	fails	⚠️ skip similarity (neutral impact)
Risk Engine	fails	⚠️ fallback to correlation rules
8. PERFORMANCE CONSTRAINTS
All modules must execute < 10ms
No LLM calls
SQLite queries must be indexed (setup_type, regime)
9. KEY IMPROVEMENTS OVER V1
✅ Smarter scoring
sample-size-aware weighting
continuous vs binary logic
✅ Better learning
similarity confidence
weighted matching vs strict filters
✅ Real risk control
multi-bucket exposure
adaptive throttling
✅ More realistic behavior
scaling vs yes/no decisions
🎯 END STATE

After implementation:

Every trade is scored probabilistically
Decisions are influenced by historical analogs
Risk is controlled at portfolio level
System adapts without LLM dependence