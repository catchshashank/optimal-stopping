# All Results — Optimal Stopping on NL Negotiation Data

**Model:** GPT-4o (zero-shot)  
**Test set:** 45 conversations (27 sales, 18 no-sales)  
**Baseline:** no stopping agent — all negotiations run to completion

---

## What the metrics mean (plain English)

### AUC at decision window m
> "If you randomly pick one *sale* negotiation and one *no-sale* negotiation from the test set,
> what fraction of the time does GPT-4o correctly give the sale conversation a higher
> 'yes' score at the m-second mark?"

| AUC | Meaning |
|-----|---------|
| 0.50 | Coin flip — model has no signal |
| 0.60 | Correct 6 out of 10 random pairs |
| 0.75 | Correct 3 out of 4 random pairs |
| 1.00 | Perfect — always distinguishes sale from no-sale |

An AUC below 0.55 is effectively noise. The model needs consistent signal above ~0.65 
to produce a stopping policy that reliably retains sales while cutting time.

### Reward gained (per conversation, in value units)
> "On average, how much better off is the agent per negotiation compared to letting
> every call run to completion?"

The reward function is:  
`reward = (sales × benefit_per_sale) − (total_time_s × cost_per_s)`

- **Positive reward gained** → the stopping agent creates net value (saves more cost than it loses in foregone sales)
- **Negative reward gained** → the agent destroys value (drops profitable sales it shouldn't have)
- The *magnitude* scales with the cost/benefit parameters — runs with different RF values are not directly comparable in absolute reward, but the trade-off pattern is.

---

## Full Results Table

| Run | Windows | Cost/s | Benefit | AUC m₁ | AUC m₂ | Baseline avg reward | Agent avg reward | Reward gained (total) | Sales retained | Time saved |
|-----|---------|--------|---------|--------|--------|--------------------|-----------------|-----------------------|----------------|------------|
| **Original** | 45s / 60s | 0.10 | 10 | 0.520 | 0.488 | −34.07 | −4.57 | **+1,327.58** | 0 / 27 | 88.6% |
| RF1 × DP1 | 115s / 230s | 0.01 | 10 | 0.585 | 0.447 | +1.99 | +1.53 | −20.90 | 24 / 27 | 5.0% |
| RF2 × DP1 | 115s / 230s | 0.05 | 10 | 0.585 | 0.447 | −14.03 | −5.75 | +372.79 | 0 / 27 | 71.3% |
| **RF1 × DP2** ★ | **90s / 180s** | **0.01** | **10** | **0.753** | **0.532** | **+1.99** | **+2.06** | **+3.20** | **27 / 27** | **1.8%** |
| RF2 × DP2 | 90s / 180s | 0.05 | 10 | 0.753 | 0.532 | −14.03 | −4.50 | +429.04 | 0 / 27 | 77.5% |

★ Only run that retains all sales while still improving reward over baseline.

---

## Intuitive read of each run

### Original (45s / 60s, cost=0.10)
The 45-second window captures roughly the greeting — almost no negotiation content.
AUC of 0.52 is barely above chance. With a high cost per second (0.10), the math
dictates quitting almost every call early. The agent saves 88.6% of time but drops all 27 sales.
The large "reward gained" (+1,328) is an artefact of the punishing cost rate making
time so expensive that saving time always wins, even at the cost of all revenue.

### RF1 × DP1 (115s / 230s, cost=0.01)
Longer windows improve AUC to 0.585 — the model has seen more of the conversation.
Low cost makes it cheap to stay on the call, so the agent continues most negotiations.
It loses 3 sales (threshold set too aggressively on a few borderline conversations)
and saves only 5% of time. Net result: slightly worse than just letting all calls run.

### RF2 × DP1 (115s / 230s, cost=0.05)
Same windows, higher cost. Cost now outweighs expected benefit for most calls so the
agent quits everything — identical failure mode to the original run. The AUC of 0.585
is not strong enough to separate the minority of genuinely promising calls.

### RF1 × DP2 ★ (90s / 180s, cost=0.01) — best run
The 90-second window is the sweet spot: enough content for GPT-4o to read the tone
and early offers, yielding AUC 0.753 — correct on 3 in 4 random sale/no-sale pairs.
Low cost keeps the agent from quitting prematurely. Result: all 27 sales retained,
1.8% of wasted time eliminated, and a small but genuine improvement in avg reward
(+1.99 → +2.06 per conversation). This is the only configuration where the agent
is strictly better than no agent without sacrificing any sales.

### RF2 × DP2 (90s / 180s, cost=0.05)
Best AUC (0.753) but cost is too high — the model correctly identifies many no-sales
but the threshold collapses under the cost pressure and quits all calls including the
sales. Saves 77.5% of time but loses all revenue. The reward gain (+429) reflects
cost savings dominating, not intelligent discrimination.

---

## Key takeaway

The AUC at the first window is the strongest predictor of whether the agent can
do useful work. Below ~0.65 (original, DP1 runs), no cost/benefit tuning rescues
the agent — it either quits too much or too little. At 0.753 (DP2), the model has
enough signal to actually discriminate, and a low cost rate (RF1) lets it act on
that signal conservatively. The 90-second mark — roughly 25% into the average
call — appears to be where GPT-4o crosses from noise into meaningful prediction
for this NL negotiation dataset.
