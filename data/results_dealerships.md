# Dealerships Dataset — Optimal Stopping Results & Analysis

**Model:** GPT-4o (zero-shot)  
**Dataset:** `dealerships-nego.csv` — 48 real car dealership sales calls with diarized timestamps  
**Split:** 27 train | 9 val | 12 test (stratified by outcome)  
**Baseline:** no stopping agent — all calls run to completion  
**Outcomes:** 23 sale / 25 no-sale  

---

## Dataset Characteristics

Unlike the NL negotiation dataset (estimated timestamps, 178 conversations), the dealerships
dataset uses real diarized timestamps from actual recorded calls. This changes the nature
of the problem in two important ways:

1. **Small dataset (48 conversations):** Val and test sets are very small (9 and 12 conversations),
   making threshold tuning on val and evaluation on test both high-variance.

2. **Bimodal duration distribution:** Calls are not uniformly distributed across time.

| Duration bin | Count | Typical outcome |
|---|---|---|
| < 100s | 2 | no-sale |
| 100–175s | 15 | no-sale |
| 175–400s | 6 | mixed |
| 400–600s | 3 | sale |
| 600–700s | 5 | sale |
| > 700s | 14 | sale |

- **No-sale median: 163s** | **Sale median: 716s**
- The 25th percentile of all calls is 154s — well below any decision window
- The 175–400s range (where all our decision windows sit) contains only 6 calls total

This bimodal gap has major consequences for the stopping pipeline, discussed below.

---

## Experiment Progression

### Experiment 1 — Early Windows (2×2 Grid)

**Motivation:** Replicate the NL pipeline on dealerships with the same window structure.  
**Windows tested:** 90s/180s and 115s/230s  
**Reward functions:** cost=1.0 & 5.0, benefit=1000

**Result:** Poor. AUC at 90s was below 0.5 (worse than random), and the agent either quit
everything or did nothing. The 90–230s range captures only the opening phase of dealership
calls — customer introduction, car interest, inventory checks — before any negotiation
language emerges.

**Lesson:** Window placement must reflect the structure of *this* dataset, not the NL dataset.
The average dealership call is 497s (NL average: 389s), and no-sale calls end much earlier
than sale calls. Decision windows need to sit where meaningful closing language occurs.

---

### Experiment 2 — 4-Window Sequential Stopping

**Motivation:** Cover more of the call with finer-grained checkpoints.  
**Windows:** 175s, 260s, 350s, 400s (~33%, 50%, 67%, 75% of median 520s call)  
**Reward functions:** cost=0.01 & 0.05 (benefit=10) and cost=1.0 & 5.0 (benefit=1000)  
**Architecture:** Backward induction — thresholds tuned m4→m3→m2→m1 on validation set

**Key findings:**
- AUC improved to 0.639–0.667 at 175s/260s — early closing signals present by one-third of the call
- AUC collapsed to 0.333 at 350s — mid-negotiation "noise zone" where neither outcome is clear
- AUC partially recovered at 400s (0.611) as final negotiation signals emerge
- Agent performance was mixed: low-cost RFs retained all sales but saved little time;
  high-cost RFs saved time aggressively at the cost of most sales

---

### Experiment 3 — 3-Pair Grid (Main Experiment)

**Motivation:** Decompose the 4-window sequential run into three independent 2-checkpoint
problems to isolate the contribution of each window pair and test all four reward functions
systematically.

**Structure:** 3 window pairs × 4 reward functions = 12 runs  
Inference runs once per window pair (shared across all 4 reward functions).

#### Window Pairs
| Name | m1 | m2 | Rationale |
|---|---|---|---|
| WP1 | 175s | 260s | Early-call closing signals (~33–50% of median) |
| WP2 | 260s | 350s | Mid-call negotiation phase (~50–67%) |
| WP3 | 350s | 400s | Late-call closing phase (~67–75%) |

#### Reward Functions
| Name | Cost/s | Benefit | Interpretation |
|---|---|---|---|
| RF1 | 0.1 | 10 | Low-stakes, time-sensitive |
| RF2 | 0.5 | 10 | Moderate cost pressure |
| RF3 | 1.0 | 1000 | High benefit, conservative stopping |
| RF4 | 5.0 | 1000 | Aggressive cost pressure |

---

## Results — 3-Pair Grid (Conditional AUC)

> **Note on AUC:** Results below report *conditional AUC* — computed only on conversations
> where `duration >= m`. Conversations that ended before the window receive no prediction
> from GPT-4o (their calls were already over). Including them via `fillna(0)` would
> artificially inflate or suppress AUC depending on their outcome distribution.
> Conditional AUC measures the model's true discriminative power among calls that are
> actually live at the decision point.

### WP1 — 175s / 260s

| AUC at 175s | AUC at 260s | Calls live at 175s | Calls live at 260s |
|---|---|---|---|
| **0.833** | 0.533 | 10 / 12 | 8 / 12 |

| Run | Sales lost | Time saved | Reward gained |
|---|---|---|---|
| RF1 (c=0.1, b=10) | 4 / 6 | 54.3% | +378 |
| RF2 (c=0.5, b=10) | 5 / 6 | 71.9% | +2,718 |
| **RF3 (c=1.0, b=1000)** | **0 / 6** | **3.1%** | **+235** |
| RF4 (c=5.0, b=1000) | 4 / 6 | 54.3% | +16,891 |

**Best window pair overall.** AUC of 0.833 at 175s means GPT-4o correctly ranks a
sale above a no-sale 5 out of 6 times among calls still live at that point — strong signal.
RF3 is the only configuration that retains all sales while improving reward (+235 total).
RF4's massive reward gain is misleading: it drops 4 of 6 sales but the high cost-per-second
makes time savings dominate the reward arithmetic.

---

### WP2 — 260s / 350s

| AUC at 260s | AUC at 350s | Calls live at 260s | Calls live at 350s |
|---|---|---|---|
| 0.600 | 0.533 | 8 / 12 | 8 / 12 |

| Run | Sales lost | Time saved | Reward gained |
|---|---|---|---|
| RF1 (c=0.1, b=10) | 5 / 6 | 55.4% | +376 |
| RF2 (c=0.5, b=10) | 5 / 6 | 55.4% | +2,082 |
| **RF3 (c=1.0, b=1000)** | **0 / 6** | **0.0%** | **0** |
| RF4 (c=5.0, b=1000) | 5 / 6 | 55.4% | +16,321 |

**Worst window pair.** AUC drops to 0.600/0.533 — mid-negotiation is linguistically
ambiguous. RF3 retains all sales but sets thresholds at 0, meaning the agent takes no
action at all (equivalent to no stopping). The model cannot reliably distinguish outcomes
at this stage of the call, so the safe policy is to always continue.

---

### WP3 — 350s / 400s

| AUC at 350s | AUC at 400s | Calls live at 350s | Calls live at 400s |
|---|---|---|---|
| 0.400 | **0.900** | 8 / 12 | 7 / 12 |

| Run | Sales lost | Time saved | Reward gained |
|---|---|---|---|
| RF1 (c=0.1, b=10) | 5 / 6 | 49.5% | +331 |
| RF2 (c=0.5, b=10) | 5 / 6 | 49.5% | +1,856 |
| **RF3 (c=1.0, b=1000)** | **0 / 6** | **0.0%** | **0** |
| RF4 (c=5.0, b=1000) | 5 / 6 | 49.5% | +14,056 |

**Most interesting pattern.** The 350s mark is the worst discriminator (AUC 0.400 —
below chance), but 400s recovers to AUC 0.900 — the strongest signal across all windows.
Among the 7 test calls still live at 400s, GPT-4o is nearly perfect. However, only 7 of
12 test calls reach 400s, and the 5 that ended earlier include 5 of the 6 total sales
(those long sales all run past 400s naturally, so no stopping opportunity is missed —
but 5 no-sales already self-terminated, leaving a survivor population that is sales-dominant).
RF3 again takes no action because the threshold collapses to 0.

---

## The Bimodal Duration Problem

The fundamental challenge across all dealerships experiments is structural, not model quality:

**Call duration strongly predicts outcome independent of linguistic content:**
- Short calls (< 175s) → almost always no-sale (15 of 17 such calls)
- Long calls (> 400s) → almost always sale

This means:
1. **Many calls self-resolve before any window is reached.** By 400s, 47.9% of calls
   have already ended — 6 of 23 total sales are among them (they ended too early to
   be captured). The agent can only act on surviving calls.

2. **The "signal" GPT-4o extracts may partly proxy call length,** not purely linguistic
   content. A call that has produced 175 seconds of transcript is inherently more likely
   to be a sale than one that produced 100 seconds — independent of what was said.

3. **The 260–400s window range is nearly empty.** Only 6 conversations fall in the
   175–400s range. This makes threshold tuning on a 9-conversation validation set
   extremely high-variance.

**Implication:** The high AUC values (especially 0.833 at 175s and 0.900 at 400s) should
be interpreted cautiously — they may reflect GPT-4o's implicit sensitivity to transcript
length rather than semantic content. A duration-only baseline (predict sale if call is
still running at window m) would be a useful control to run in future experiments.

---

## Comparison: NL vs Dealerships

| Dimension | NL Negotiations | Dealerships |
|---|---|---|
| N conversations | 178 | 48 |
| Avg duration | 389s | 497s |
| Duration variance | Moderate (std ~180s) | High (std ~356s) |
| Duration as outcome predictor | Weak | **Strong** |
| Best AUC achieved | 0.753 (90s, unconditional) | 0.900 (400s, conditional) |
| Best run | RF1×DP2: 0 sales lost, +1.8% time saved | RF3×WP1: 0 sales lost, +3.1% time saved |
| Dataset limitation | Estimated timestamps | Bimodal distribution, small N |

---

## Key Takeaways

1. **WP1 (175s/260s) with RF3 (cost=1.0, benefit=1000) is the only configuration that
   retains all sales and improves reward** — the dealerships analogue of the NL best run.
   The gain is modest (+235 total, +3.1% time saved) but the agent is strictly non-harmful.

2. **Conditional AUC is the correct metric** for this pipeline. Unconditional AUC
   conflates two different populations (pre-window terminated calls and live calls) and
   systematically underestimates model quality. After correction, AUC at 175s rises from
   0.611 to 0.833 and at 400s from 0.583 to 0.900.

3. **The 350s mark is a dead zone.** AUC falls to 0.400 (below chance) regardless of
   whether AUC is conditional or not. Mid-negotiation language is ambiguous — price
   anchoring and objection handling occur in both sale and no-sale calls.

4. **Dataset size is a binding constraint.** With only 12 test conversations (6 sales,
   6 no-sales), a single misclassification shifts "sales lost" by 17%. Results should
   be treated as directional, not definitive.

5. **A duration-only baseline is a necessary control** for future work. The strong
   correlation between call length and outcome in this dataset means we cannot yet
   attribute model performance to linguistic understanding vs. duration sensitivity.
