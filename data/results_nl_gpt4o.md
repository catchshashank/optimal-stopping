# Optimal Stopping — NL Negotiations Results (GPT-4o)

**Date:** 2026-04-25  
**Backbone:** GPT-4o  
**Dataset:** NL negotiation transcripts (178 conversations → 7,565 utterance rows)  
**Data file:** `data/nego-data-final.csv`  
**Script:** `run_nego.py`

---

## Dataset Split

| Split | Conversations |
|-------|--------------|
| Train | 99 |
| Val   | 34 |
| Test  | 45 |

---

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Cost per unit time | 0.1 |
| Benefit per positive outcome | 10.0 |
| Decision opportunities | 45s, 60s |

---

## Optimal Action Distribution (Train)

| Action | Count |
|--------|-------|
| no (quit) | 193 |
| yes (continue) | 4 |

> Note: heavy class imbalance — most conversations are optimally quit early under these cost parameters.

---

## ROC-AUC (GPT-4o zero-shot prediction of sale outcome)

| Split | 45s window | 60s window |
|-------|-----------|-----------|
| Val   | 0.468 | 0.414 |
| Test  | 0.520 | 0.488 |

---

## Stopping Agent Performance (Test Set)

### Baseline (no stopping agent — all calls run to completion)

| Metric | Value |
|--------|-------|
| Total reward | -1533.08 |
| Avg reward / conversation | -34.07 |
| Total sales | 27 |
| Total time | 18,030.8s |

### With Stopping Agent (thresholds tuned on val set)

| Metric | Value |
|--------|-------|
| Threshold at 45s | 0.9962 |
| Threshold at 60s | 0.9985 |
| Total reward | -205.50 |
| Avg reward / conversation | -4.57 |
| Total sales | 0 |
| Total time | 2,055.0s |

### Comparative Summary

| Metric | Value |
|--------|-------|
| Sales lost | 27 (all) |
| Time saved | 15,975.8s (88.6%) |
| Reward gained | +1,327.58 |

---

## Interpretation

The stopping agent aggressively quits early (thresholds ~0.996–0.999), sacrificing all 27 sales
to avoid the large time cost. This reflects the cost structure (0.1/s × long durations outweighs
10.0 per sale for most conversations) combined with GPT-4o's near-chance AUC (~0.47–0.52),
meaning the model cannot reliably distinguish sale vs no-sale conversations early on.

The low AUC is expected given the NL data has only word-level turns distributed proportionally
over duration (no real timestamps), so the 45s/60s transcript windows are sparse for many
short conversations.
