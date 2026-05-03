# Optimal Stopping — NL Negotiations: Project Context

## Dataset

- **Source:** `negotiations_public_release/nl/` — 178 JSON files, one per negotiation
- **Original format:** Word-level turns (`ID`, `Role`, `Word`, `Span`, `Category`, ...)
- **Converted to:** `data/nego-data-final.csv` — repo format with columns:
  `conversation_id`, `speaker_id` (0=buyer, 1=seller), `start_time`, `end_time`, `text`, `outcome`
- **Conversion logic:**
  - Words within each turn joined into utterances
  - Timestamps distributed proportionally over `duration_min × 60` seconds (no real timestamps in source)
  - Outcome = "sale" if any word has Category `"a"` (agreement), else "no sale"
- **Stats:** 178 conversations | 105 sale | 73 no-sale | avg duration 389s (6.5 min) | range 46s–946s
- **Split:** 99 train | 34 val | 45 test (stratified, conversation-level)

## Model

- **Backbone:** GPT-4o (zero-shot, no fine-tuning)
- **Task:** Given partial transcript up to decision window m, predict "yes" (deal) or "no" (no deal)
- **Probability:** extracted from GPT-4o logprobs, normalised over {yes, no}
- **Policy:** threshold-based stopping — quit if `prob_yes < threshold`, continue otherwise
- **Threshold tuning:** backward induction grid search on validation set (10,000 points)

## Reward Function

`reward = (sales × benefit_per_sale) − (total_time_s × cost_per_s)`

Optimal actions per conversation computed via backward induction over two decision windows.

## Results

| Run | Windows | Cost/s | Benefit | AUC m₁ | AUC m₂ | Sales retained | Time saved | Reward gained |
|-----|---------|--------|---------|--------|--------|----------------|------------|--------------|
| Original | 45s / 60s | 0.10 | 10 | 0.520 | 0.488 | 0 / 27 | 88.6% | +1,328 |
| RF1 × DP1 | 115s / 230s | 0.01 | 10 | 0.585 | 0.447 | 24 / 27 | 5.0% | −21 |
| RF2 × DP1 | 115s / 230s | 0.05 | 10 | 0.585 | 0.447 | 0 / 27 | 71.3% | +373 |
| **RF1 × DP2 ★** | **90s / 180s** | **0.01** | **10** | **0.753** | **0.532** | **27 / 27** | **1.8%** | **+3** |
| RF2 × DP2 | 90s / 180s | 0.05 | 10 | 0.753 | 0.532 | 0 / 27 | 77.5% | +429 |

## Key Finding

**90s/180s windows with cost=0.01** is the only configuration that retains all sales and improves on the no-agent baseline. The 90-second mark (~25% into the average call) is where GPT-4o crosses from noise (AUC ~0.52) into genuine signal (AUC 0.75). Below AUC ~0.65, cost/benefit tuning cannot rescue the agent — it either quits everything or nothing useful.
