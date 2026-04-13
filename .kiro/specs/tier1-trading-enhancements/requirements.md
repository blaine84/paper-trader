# Requirements Document

## Introduction

This document specifies the Tier 1 enhancements (v2) to the Paper Trader multi-agent system. The enhancements introduce three new deterministic, LLM-free modules — an Edge Score calculator, a Similarity Matching engine, and a Portfolio Risk engine — that integrate into the existing Portfolio Manager decision pipeline. These modules upgrade the system from rule-based trade decisions to adaptive, data-driven execution by quantifying trade edge, leveraging historical case similarity, and actively managing portfolio-level risk. V2 adds sample-size-aware confidence scoring, weighted similarity matching, adaptive risk throttling, a new mega_growth exposure bucket, configurable total exposure thresholds, and enhanced structured logging. All changes are additive and backward-compatible with the existing system.

## Glossary

- **Edge_Score_Calculator**: A pure-function module (`core/edge_score.py`) that computes a continuous 0.0–1.0 score representing the quantified edge of a proposed trade, using weighted inputs from case library win rates, similarity statistics, signal strength, signal confidence, indicator confluence, and similarity quality.
- **Similarity_Engine**: A deterministic module (`core/similarity.py`) that queries the SQLite case library to find historically similar trades using weighted scoring across multiple criteria and computes aggregate performance statistics from the matched cases.
- **Portfolio_Risk_Engine**: A deterministic module (`core/portfolio_risk.py`) that computes portfolio-level exposure across correlated asset buckets, validates whether a new trade would exceed risk thresholds, computes a risk score, and applies adaptive risk throttling based on recent loss streaks.
- **PM_Pipeline**: The existing Portfolio Manager decision flow in `agents/portfolio_manager.py` where analyst signals are evaluated, trades are decided, validated, and executed.
- **Case_Library**: The SQLite `cases` table containing structured records of past closed trades with setup type, market regime, indicators, outcome, and P&L data (defined in `models/case.py`).
- **Signal**: The JSON output from the Analyst agent containing direction (LONG/SHORT/HOLD), strength, confidence, setup_type, key_levels, invalidation, and indicator readings.
- **Exposure_Bucket**: A named group of correlated symbols (e.g., "index" for SPY/QQQ/IWM, "semis" for NVDA/AMD, "mega_growth" for NVDA/TSLA/META/AMZN) used to track and limit concentrated portfolio risk. A symbol may belong to multiple buckets.
- **Trades_Table**: The SQLite `trades` table (defined in `db/schema.py`) that stores all paper trade records including entry/exit prices, P&L, stop/target levels, review scores, and similarity confidence.
- **Confluence_Score**: A sub-score within the Edge_Score_Calculator that measures how many technical indicators align in the same direction (e.g., above VWAP + bullish EMA trend + RSI in favorable range).
- **Similarity_Quality**: A sub-score within the Edge_Score_Calculator that measures confidence in similarity data based on sample size, computed as `min(1.0, similarity_sample_size / 10)`.
- **Similarity_Confidence**: An output field from the Similarity_Engine representing confidence in the similarity statistics, computed as `min(1.0, sample_size / 10)`.
- **Risk_Score**: A composite score output by the Portfolio_Risk_Engine summarizing overall portfolio risk state.

## Requirements

### Requirement 1: Edge Score Computation

**User Story:** As a Portfolio Manager agent, I want each proposed trade to have a quantified edge score so that I can make data-driven decisions about whether to execute and how to size the position.

#### Acceptance Criteria

1. WHEN a Signal, case statistics dictionary, and similarity statistics dictionary are provided, THE Edge_Score_Calculator SHALL compute a weighted score using the formula: 0.25 × normalized setup win rate + 0.20 × normalized similarity win rate + 0.15 × mapped signal strength + 0.10 × mapped signal confidence + 0.15 × Confluence_Score + 0.15 × Similarity_Quality.
2. THE Edge_Score_Calculator SHALL return a floating-point value clamped to the range [0.0, 1.0] inclusive.
3. WHEN the signal strength is "strong", THE Edge_Score_Calculator SHALL map the strength component to 1.0; WHEN "moderate", to 0.6; WHEN "weak", to 0.3.
4. WHEN the signal confidence is "high", THE Edge_Score_Calculator SHALL map the confidence component to 1.0; WHEN "medium", to 0.6; WHEN "low", to 0.3.
5. WHEN the case statistics win rate is 1.0, THE Edge_Score_Calculator SHALL normalize the win rate component to 1.0; WHEN 0.0, to 0.0; WHEN 0.5, to 0.5.
6. WHEN the similarity statistics win rate is 1.0, THE Edge_Score_Calculator SHALL normalize the similarity component to 1.0; WHEN 0.0, to 0.0.
7. THE Edge_Score_Calculator SHALL compute the Confluence_Score by evaluating indicator alignment from the Signal's indicators dictionary, where each aligned indicator (above_vwap is true, ema_trend matches bias direction, RSI in favorable range, MACD bias matches direction, bb_position is favorable) contributes equally to a 0.0–1.0 score.
8. THE Edge_Score_Calculator SHALL execute without making any LLM API calls.
9. WHEN identical inputs are provided on separate invocations, THE Edge_Score_Calculator SHALL return identical output values.
10. THE Edge_Score_Calculator SHALL compute the Similarity_Quality component as `min(1.0, similarity_sample_size / 10)`, where similarity_sample_size is obtained from the similarity statistics dictionary.
11. WHEN the case statistics sample_size is 10 or greater and the setup win rate is below 0.35, THE Edge_Score_Calculator SHALL reject the trade outright before computing the edge score.
12. WHEN the edge score is used for position sizing, THE PM_Pipeline SHALL cap the scaled size at `min(scaled_size, base_size * 1.2)` to prevent oversizing.

### Requirement 2: Similarity Matching

**User Story:** As a Portfolio Manager agent, I want to find historically similar trades so that I can leverage past experience to inform current decisions.

#### Acceptance Criteria

1. WHEN a Signal with setup_type, market_regime, RSI value, bias, above_vwap, and ema_trend is provided, THE Similarity_Engine SHALL compute a weighted similarity score for each case in the Case_Library using the criteria: setup_type match, market_regime match, RSI distance (continuous, not bucketed), VWAP alignment (boolean match), and EMA trend alignment.
2. THE Similarity_Engine SHALL return the top N (5–10) cases ranked by descending similarity score.
3. THE Similarity_Engine SHALL compute aggregate statistics from matched cases including similarity_winrate (fraction of cases with outcome "success"), similarity_avg_r (average pnl_pct), similarity_sample_size (count of matched cases), and similarity_confidence computed as `min(1.0, sample_size / 10)`.
4. WHEN the similarity_sample_size is 0, THE Similarity_Engine SHALL skip similarity weighting entirely in the edge score computation instead of penalizing with zero values.
5. THE Similarity_Engine SHALL execute without making any LLM API calls.
6. THE Similarity_Engine SHALL use weighted scoring rather than strict all-must-match filtering, so that cases matching on most criteria are still returned even if one criterion does not match.

### Requirement 3: Portfolio Risk Management

**User Story:** As a Portfolio Manager agent, I want portfolio-level risk controls so that I can prevent hidden leverage and correlated drawdowns.

#### Acceptance Criteria

1. THE Portfolio_Risk_Engine SHALL classify symbols into exposure buckets: "index" (SPY, QQQ, IWM, DIA), "semis" (NVDA, AMD, INTC, TSM), "ev" (TSLA, LCID, RIVN), and "mega_growth" (NVDA, TSLA, META, AMZN), where a symbol may belong to multiple buckets simultaneously.
2. THE Portfolio_Risk_Engine SHALL compute per-bucket exposure as the sum of position values in that bucket divided by total equity.
3. WHEN a new trade would cause any single bucket exposure to exceed 50% of total equity, THE Portfolio_Risk_Engine SHALL reject the trade.
4. WHEN a new trade would cause total portfolio exposure to exceed a configurable threshold (default range 1.2–1.5× total equity), THE Portfolio_Risk_Engine SHALL reject the trade.
5. THE Portfolio_Risk_Engine SHALL output a risk result dictionary containing total_exposure, bucket_exposure, and a risk_score field summarizing overall portfolio risk state.
6. THE Portfolio_Risk_Engine SHALL execute without making any LLM API calls.
7. WHEN the recent_losses count (consecutive recent losing trades) is 3 or greater, THE Portfolio_Risk_Engine SHALL reduce position size by 25–50% as adaptive risk throttling.

### Requirement 4: PM Pipeline Integration

**User Story:** As a system operator, I want the Edge Score, Similarity, and Portfolio Risk modules integrated into the PM decision pipeline so that every trade is scored, sized, and risk-checked automatically.

#### Acceptance Criteria

1. WHEN a BUY or SHORT action is decided by the PM, THE PM_Pipeline SHALL call the Similarity_Engine, then the Edge_Score_Calculator, then the Portfolio_Risk_Engine before executing the trade.
2. WHEN the edge score is below 0.4, THE PM_Pipeline SHALL reject the trade.
3. WHEN the edge score is 0.4 or above, THE PM_Pipeline SHALL scale the position quantity by the edge score, capped at `min(scaled_size, base_size * 1.2)`.
4. WHEN the Portfolio_Risk_Engine rejects the trade, THE PM_Pipeline SHALL not execute the trade.
5. THE PM_Pipeline SHALL store edge_score, similarity_winrate, similarity_sample_size, and similarity_confidence on the Trade record.
6. IF the Edge_Score_Calculator raises an unexpected exception, THEN THE PM_Pipeline SHALL reject the trade (fail-closed).
7. IF the Similarity_Engine raises an unexpected exception, THEN THE PM_Pipeline SHALL proceed with zero similarity stats (fail-open).
8. IF the Portfolio_Risk_Engine raises an unexpected exception, THEN THE PM_Pipeline SHALL proceed with existing correlation check only (fail-open).

### Requirement 5: Database Schema Extension

**User Story:** As a system operator, I want trade records to include edge score and similarity data so that I can analyze decision quality over time.

#### Acceptance Criteria

1. THE Trades_Table SHALL include a `similarity_confidence` column of type REAL, nullable, to store the similarity confidence value for each trade.
2. THE Trades_Table SHALL include `edge_score` (REAL, nullable), `similarity_winrate` (REAL, nullable), and `similarity_sample_size` (INTEGER, nullable) columns.
3. WHEN a new trade is created, THE PM_Pipeline SHALL populate the edge_score, similarity_winrate, similarity_sample_size, and similarity_confidence columns with the computed values.

### Requirement 6: Structured Logging

**User Story:** As a system operator, I want detailed structured logging of edge score components, portfolio risk state, and trade decisions so that I can debug and audit the system.

#### Acceptance Criteria

1. WHEN an edge score is computed, THE PM_Pipeline SHALL log the edge score value along with each component: setup_winrate (with sample size), similarity_winrate (with sample size), similarity_confidence, confluence score, and similarity quality.
2. WHEN a portfolio risk check is performed, THE PM_Pipeline SHALL log the total_exposure and per-bucket exposure values.
3. WHEN a trade decision is made (executed or rejected), THE PM_Pipeline SHALL log a DECISION block containing the scaled size, execution status, and rejection reason if applicable.
