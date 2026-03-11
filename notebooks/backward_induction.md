## **Backward Induction and Block 12: Finding Optimal Threhold Values**
---

### **Backward Induction**

---

#### The Core Idea

Backward induction is a problem-solving strategy borrowed from **game theory and
dynamic programming**. The central insight is simple:

> *When decisions are made in sequence, solve the last decision first — then work
> backwards.*

This works because the last decision in a sequence is always the easiest to
optimise: it has no future consequences to worry about. Once you know the best
last decision, you can treat it as a fixed fact and optimise the second-to-last
decision. And so on, all the way back to the first decision.

---

#### A Familiar Analogy — Chess

A chess player using backward induction does not ask:
*"What is the best move right now?"*

They ask:
*"What board positions lead to checkmate? Now, what moves lead to those positions?
Now, what moves lead to those moves?"*

They reason from the **end state backwards** to the current move. The same logic
applies here — start from the last checkpoint (60s) and reason backwards to the
first (45s).

---

#### Why Sequential Decisions Break Simple Search

Suppose you have two decisions, each with B = 10,000 possible values. If the
decisions were **independent** — meaning the first choice does not affect the
second — you could optimise them separately:

$$\text{Total evaluations} = B + B = 20,000$$

But if you mistakenly treated them as **jointly dependent** and searched all
combinations:

$$\text{Total evaluations} = B \times B = 100,000,000$$

The key question is: *are the two thresholds actually independent?*

In this problem, they are **conditionally independent** given the sequential
structure:
- The best 60s threshold does not change based on what the 45s threshold is,
  because the 60s decision only ever applies to calls that *already survived* the
  45s checkpoint
- The 45s threshold simply controls *which calls* reach the 60s decision — it does
  not alter what the *best* 60s decision is for those calls

This conditional independence is what makes backward induction valid and exact
here — not an approximation, but the provably correct solution.

---

#### The General Framework

Backward induction applies whenever a problem has these three properties:

| Property | This Problem |
|---|---|
| **Sequential decisions** | Agent decides at 45s, then at 60s — in that order |
| **No revisiting** | Once a call is ended at 45s, you cannot restart it at 60s |
| **Later decisions do not affect earlier ones** | The best 60s policy is the same regardless of how calls arrived there |

When all three hold, the problem can be decomposed into T independent
single-decision optimisations solved in reverse order — reducing exponential
complexity O(Bᵀ) to linear complexity O(B × T).

| Approach | Formula | This Problem (T=2, B=10,000) |
|---|---|---|
| Joint grid search | O(Bᵀ) | 100,000,000 evaluations |
| Backward induction | O(B × T) | 20,000 evaluations |
| Speedup | Bᵀ⁻¹ | **5,000×** |

---

#### Step-by-Step Logic Applied Here

**Step 1 — Solve the last decision (60s checkpoint)**

Imagine you are standing at the 60s mark of a call. The 45s decision has already
been made — this call survived it. Your only question is:

> *"Given the model's confidence score at 60s, should I quit or continue?"*

There is no future checkpoint after 60s. The reward from this decision depends
only on the call's outcome and duration — both fixed facts. So we can find the
optimal λ₂ by simply trying all 10,000 candidates and picking the best.

**Step 2 — Solve the first decision (45s checkpoint), treating Step 1 as given**

Now imagine you are at the 45s mark. You know that *if* this call survives to 60s,
it will be governed by the optimal λ₂ you just found. Your only question is:

> *"Given the model's confidence score at 45s, and knowing that the 60s policy is
> already optimised, should I quit now or let this call reach 60s?"*

Again, try all 10,000 candidates for λ₁ and pick the one that produces the best
average reward across the full two-checkpoint simulation.

---

#### What Backward Induction Is NOT Doing Here

It is worth being explicit about what this approach does *not* assume:

- It does **not** assume the two thresholds are unrelated — it correctly models
  the fact that calls quitting at 45s never reach the 60s decision
- It does **not** approximate — the result is the same optimal pair (λ₁, λ₂) that
  a full joint search would find, given the conditional independence property holds
- It does **not** require the model to be perfect — it works purely on the
  observed `prob_yes` scores, whatever quality they happen to be

---

### Block 12 — Finding Optimal Decision Thresholds: Backward Induction

---

#### What Problem Are We Solving?

After training, the model outputs a continuous score — `prob_yes` ∈ [0, 1] — for
every call at every checkpoint. But the agent needs to make a **hard binary
decision**: quit or wait.

The question becomes: **at what confidence level should the agent decide to quit?**

For example:
- If `prob_yes < 0.2` → quit (model thinks there is less than 20% chance of a sale)
- If `prob_yes ≥ 0.2` → wait

That cut-off value (0.2 in this example) is the **threshold**. The entire job of
Block 12 is to find the *best* threshold for each checkpoint — not best in terms
of prediction accuracy, but best in terms of **actual financial reward**.

---

#### Why Two Thresholds?

There are two decision points — 45s and 60s — and each needs its own threshold
because the information available at each point is different:

- At 45s, the agent has heard less of the call → confidence scores will be different
- At 60s, it has heard more → scores will generally be more reliable

So we need λ₁ for the 45s checkpoint and λ₂ for the 60s checkpoint. The challenge
is finding the best *combination* of both.

---

#### Why Not Search Both Together?

The naive approach would be a **joint grid search**: try every combination of
λ₁ and λ₂.

With 10,000 candidate values each:

$$10,000 \times 10,000 = 100,000,000 \text{ evaluations}$$

Each evaluation runs `simulate_threshold` across all validation calls. This is
computationally prohibitive.

**Backward induction** exploits the sequential structure of the problem to reduce
this to:

$$10,000 + 10,000 = 20,000 \text{ evaluations}$$

A **5,000× speedup**.

---

#### Step 1 — Fix the 60s Threshold First
```python
m = m1
for candidate_threshold in np.linspace(min_prob, max_prob, num=10000):
    total_reward, avg_reward, ... = simulate_threshold(0, candidate_threshold, val_data)
    if avg_reward > best_reward:
        best_threshold_at_m[m] = candidate_threshold
```

**Why start at 60s?** The 60s checkpoint is the *last* decision point — there are
no future decisions after it. This makes it the cleanest place to start because
its reward depends only on itself, not on what happens at 45s.

**The trick — `simulate_threshold(0, candidate_threshold, ...)`:** The first
argument is the 45s threshold, set to **0**. Since `prob_yes` is always ≥ 0, a
threshold of 0 means *no call ever quits at 45s* — every call is assumed to reach
60s. This isolates the 60s decision completely, letting us optimise it in a vacuum.

**What is `np.linspace` doing?**
```python
np.linspace(min_prob, max_prob, num=10000)
```

This generates 10,000 evenly spaced candidate threshold values between the minimum
and maximum `prob_yes` scores observed in the validation set. For example, if
scores range from 0.001 to 0.999:
```
[0.001, 0.001099, 0.001198, ..., 0.999]
```

Every candidate is tested and the one that produces the highest average reward is
kept as `best_threshold_at_m[m2]`.

---

#### Step 2 — Fix the 45s Threshold With 60s Locked In
```python
m = m2
for candidate_threshold in np.linspace(min_prob, max_prob, num=10000):
    total_reward, avg_reward, ... = simulate_threshold(
        candidate_threshold, best_threshold_at_m[m1], val_data)
    if avg_reward > best_reward:
        best_threshold_at_m[m] = candidate_threshold
```

Now the best 60s threshold is fixed and held constant. The search only moves the
45s threshold through 10,000 candidates.

For each candidate λ₁, `simulate_threshold` runs the *full two-checkpoint policy*:
- At 45s: quit if `prob_yes_45 < λ₁`, else continue
- At 60s: quit if `prob_yes_60 < λ₂` (the already-fixed best value), else let run

The candidate λ₁ that produces the highest average reward across all validation
calls becomes `best_threshold_at_m[m1]`.

**Why is this valid?** Because the decisions are sequential — what happens at 45s
does not change what the *best* 60s policy is. The 60s threshold was already
optimised assuming all calls reach it. Now we are just deciding which calls to
filter out before they even get to 60s.

---

#### Inside `simulate_threshold` — Four Buckets
```python
def simulate_threshold(threshold_m1, threshold_m2, df):

    # Bucket 1: quit at 45s
    calls_quit_at_m1 = df.loc[df["prob_yes_45"] < threshold_m1]

    # Bucket 2: passed 45s, but call ended naturally before 60s
    calls_continued_at_m1_and_ended = df.loc[
        (df["prob_yes_45"] >= threshold_m1) &
        (df["duration"] < m2)]

    # Bucket 3: passed 45s, reached 60s, quit at 60s
    calls_continued_at_m1_and_quit_at_m2 = df.loc[
        (df["prob_yes_45"] >= threshold_m1) &
        (df["prob_yes_60"] < threshold_m2) &
        (df["duration"] >= m2)]

    # Bucket 4: passed both checkpoints, let run to natural end
    calls_continued_at_m2 = df.loc[
        (df["prob_yes_45"] >= threshold_m1) &
        (df["prob_yes_60"] >= threshold_m2) &
        (df["duration"] >= m2)]
```

Every call in the validation set falls into exactly one of these four buckets.
The assertion at the end confirms this:
```python
assert len(bucket1) + len(bucket2) + len(bucket3) + len(bucket4) == len(df)
```

| Bucket | Condition | What Happened | Sales Possible? |
|---|---|---|:---:|
| 1 | `prob_yes_45 < λ₁` | Agent quit at 45s | No |
| 2 | `prob_yes_45 ≥ λ₁` AND `duration < 60s` | Agent waited at 45s, call ended naturally | Yes |
| 3 | `prob_yes_45 ≥ λ₁` AND `prob_yes_60 < λ₂` AND `duration ≥ 60s` | Agent waited at 45s, quit at 60s | No |
| 4 | Both above thresholds AND `duration ≥ 60s` | Agent waited at both checkpoints | Yes |

---

#### Computing Reward Inside Each Bucket
```python
total_sales = calls_continued_at_m1_and_ended["is_sale"].sum() + \
              calls_continued_at_m2["is_sale"].sum()
total_sales_benefit = total_sales * BENEFIT_PER_POSITIVE_OUTCOME

total_time = (len(calls_quit_at_m1) * m1 +
              len(calls_continued_at_m1_and_quit_at_m2) * m2 +
              calls_continued_at_m1_and_ended["duration"].sum() +
              calls_continued_at_m2["duration"].sum())
total_cost = total_time * COST_PER_UNIT_TIME

total_reward = total_sales_benefit - total_cost
average_reward = total_reward / len(df)
```

**Sales benefit:** Only Buckets 2 and 4 can generate sales — because only those
calls were allowed to reach their natural conclusion. Bucket 1 and 3 calls were
terminated by the agent, so whatever sale might have happened is forfeited.

**Time cost per bucket:**

| Bucket | Time Charged |
|---|---|
| 1 — quit at 45s | Flat 45s per call |
| 3 — quit at 60s | Flat 60s per call |
| 2 — ended naturally before 60s | Actual call duration |
| 4 — let run to natural end | Actual call duration |

**Average reward** is used rather than total reward so the metric does not change
simply because the validation set is large or small — it represents reward
*per call*, which is comparable across all threshold candidates.

---

#### A Concrete Numerical Example

Suppose the validation set has 5 calls and we test threshold λ₁ = 0.4, λ₂ = 0.3:

| Call | prob_yes_45 | prob_yes_60 | Duration | Sale? | Bucket | Time Cost | Sales Benefit | Reward |
|---|---:|---:|---:|:---:|---|---:|---:|---:|
| A | 0.2 | — | 80s | No | 1 (quit @45s) | 0.45 | 0 | −0.45 |
| B | 0.6 | — | 50s | Yes | 2 (ended naturally) | 0.50 | 10 | +9.50 |
| C | 0.5 | 0.1 | 75s | No | 3 (quit @60s) | 0.60 | 0 | −0.60 |
| D | 0.7 | 0.8 | 90s | Yes | 4 (let run) | 0.90 | 10 | +9.10 |
| E | 0.9 | 0.4 | 70s | No | 4 (let run) | 0.70 | 0 | −0.70 |
```
Total reward   = −0.45 + 9.50 − 0.60 + 9.10 − 0.70 = +16.85
Average reward = 16.85 / 5 = +3.37 per call
```

This average reward (+3.37) is compared against all other 9,999 candidate threshold
combinations. The combination that produces the highest average reward across the
validation set is selected as the final policy.

---

#### Why Optimise Reward and Not Accuracy?

A threshold optimised for **classification accuracy** asks:
*"does the agent correctly predict sale vs. no-sale?"*

A threshold optimised for **reward** asks:
*"does the agent make the decision that makes the most money?"*

These are different because mistakes are not symmetric:

| Error Type | Consequence |
|---|---|
| Quit a call that would have been a sale | Lose full sale benefit |
| Stay on a call that was never going to sell | Lose only the extra time cost |

Optimising accuracy treats both errors equally. Optimising reward respects the
fact that one mistake is far more expensive than the other — which is the correct
objective for a business deployment.

---

#### The Full Picture
```
Trained model outputs prob_yes ∈ [0,1] for every call at every checkpoint
        ↓
Step 1: Search 10,000 values of λ₂ (60s threshold)
        Fix λ₁ = 0 (all calls reach 60s)
        Keep λ₂ that maximises avg reward on validation set
        ↓
Step 2: Search 10,000 values of λ₁ (45s threshold)
        Fix λ₂ = best value from Step 1
        Keep λ₁ that maximises avg reward on validation set
        ↓
Apply both thresholds to test set:
  prob_yes_45 < λ₁  →  QUIT at 45s
  prob_yes_45 ≥ λ₁  →  continue to 60s
  prob_yes_60 < λ₂  →  QUIT at 60s
  prob_yes_60 ≥ λ₂  →  let call run to natural end
        ↓
Report: total reward, sales closed, time saved vs. no-agent baseline
```
> In this notebook, backward induction transforms an intractable 100-million-step
> search into a clean 20,000-step procedure — without sacrificing any optimality.
> That is the practical power of reasoning backwards.
