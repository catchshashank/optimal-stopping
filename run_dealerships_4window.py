"""
4-window sequential stopping pipeline on dealerships-nego.csv.
Decision windows: 175s, 260s, 350s, 400s (~33/50/67/75% of median 520.7s call).

Reward functions:
  Case 1 — RF1a: cost=0.01, benefit=10   | RF2a: cost=0.05, benefit=10
  Case 2 — RF1b: cost=1.0,  benefit=1000 | RF2b: cost=5.0,  benefit=1000

Backward induction tunes thresholds m4 → m3 → m2 → m1 on val set.
"""

import math, os
import numpy as np
import pandas as pd
import openai
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dealerships-nego.csv")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY environment variable before running.")

OPENAI_MODEL = "gpt-4o"

WINDOWS = [175, 260, 350, 400]   # m1, m2, m3, m4

REWARD_FUNCTIONS = {
    "RF1a_cost0.01_ben10":  {"cost": 0.01, "benefit": 10.0},
    "RF2a_cost0.05_ben10":  {"cost": 0.05, "benefit": 10.0},
    "RF1b_cost1.0_ben1000": {"cost": 1.0,  "benefit": 1000.0},
    "RF2b_cost5.0_ben1000": {"cost": 5.0,  "benefit": 1000.0},
}

# ── Load & split ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["is_sale"] = df["outcome"].str.strip().apply(
    lambda x: 1 if x == "sale" else 0 if x == "no sale" else np.nan)
df["duration"] = df.groupby("conversation_id")["end_time"].transform("max")

conv_meta  = df[["conversation_id", "is_sale"]].drop_duplicates()
all_ids    = conv_meta["conversation_id"].values
all_labels = conv_meta["is_sale"].values

train_ids, test_ids, train_y, test_y = train_test_split(
    all_ids, all_labels, test_size=0.25, random_state=42, stratify=all_labels)
train_ids, val_ids, train_y, val_y = train_test_split(
    train_ids, train_y, test_size=0.25, random_state=42, stratify=train_y)

splits = {
    "train": df[df["conversation_id"].isin(train_ids)],
    "val":   df[df["conversation_id"].isin(val_ids)],
    "test":  df[df["conversation_id"].isin(test_ids)],
}
print(f"  {len(train_ids)} train | {len(val_ids)} val | {len(test_ids)} test")
print(f"  Windows: {WINDOWS}s")

# ── Build transcripts ──────────────────────────────────────────────────────────
def build_transcripts(split_df, windows):
    d = split_df.copy()
    d["turn_text"] = "Speaker " + d["speaker_id"].astype(str) + ": " + d["text"]
    result = d[["conversation_id", "duration", "is_sale"]].drop_duplicates()
    for m in windows:
        partial = (d[d["end_time"] < m]
                   .groupby("conversation_id")["turn_text"]
                   .apply(lambda x: "\n".join(x))
                   .reset_index(name=f"transcript_{m}"))
        result = result.merge(partial, on="conversation_id", how="left")
        result[f"transcript_{m}"] = result[f"transcript_{m}"].fillna("")
    return result

def make_prompt(transcript, t):
    return (f"Below is the first {t} seconds of a car dealership sales call "
            f"between the customer (Speaker 0) and the sales agent (Speaker 1):\n"
            f"{transcript}\n"
            f"Will this call end in a sale (respond with 'yes' or 'no'):  ")

data = {k: build_transcripts(v, WINDOWS) for k, v in splits.items()}
for split_df in data.values():
    for m in WINDOWS:
        split_df[f"s{m}"] = split_df[f"transcript_{m}"].apply(make_prompt, t=m)

# ── Optimal actions (backward induction over 4 windows) ────────────────────────
def compute_optimal_actions(split_df, windows, cost, benefit):
    m1, m2, m3, m4 = windows
    d = split_df.copy()

    d["rq_m1"] = -m1 * cost
    for m in [m2, m3, m4]:
        d[f"rq_m{m}"] = (d["is_sale"] * benefit * (d["duration"] <= m).astype(int)
                         - d["duration"].apply(lambda x: min(m, x)) * cost)
    d["rc"] = d["is_sale"] * benefit - d["duration"] * cost

    d["max_reward"] = d[["rq_m1", f"rq_m{m2}", f"rq_m{m3}",
                          f"rq_m{m4}", "rc"]].max(axis=1)

    for m in windows:
        d[f"a{m}"] = pd.NA
        d[f"a{m}"] = d[f"a{m}"].astype(object)

    mask = d["max_reward"] == d["rq_m1"]
    for m in windows: d.loc[mask, f"a{m}"] = "no"

    mask = d["max_reward"] == d[f"rq_m{m2}"]
    d.loc[mask, f"a{m1}"] = "yes"
    for m in [m2, m3, m4]: d.loc[mask, f"a{m}"] = "no"

    mask = d["max_reward"] == d[f"rq_m{m3}"]
    for m in [m1, m2]: d.loc[mask, f"a{m}"] = "yes"
    for m in [m3, m4]: d.loc[mask, f"a{m}"] = "no"

    mask = d["max_reward"] == d[f"rq_m{m4}"]
    for m in [m1, m2, m3]: d.loc[mask, f"a{m}"] = "yes"
    d.loc[mask, f"a{m4}"] = "no"

    mask = d["max_reward"] == d["rc"]
    for m in windows: d.loc[mask, f"a{m}"] = "yes"

    rows = []
    for m in windows:
        subset = d[d["duration"] >= m][
            ["conversation_id", f"s{m}", f"a{m}", "is_sale", "duration"]
        ].copy()
        subset.columns = ["conversation_id", "state", "action", "is_sale", "duration"]
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)

# ── OpenAI inference ───────────────────────────────────────────────────────────
def prob_yes_from_logprobs(top_logprobs):
    lp = {e.token.strip().lower(): e.logprob for e in top_logprobs}
    p_yes = math.exp(lp.get("yes", -100.0))
    p_no  = math.exp(lp.get("no",  -100.0))
    total = p_yes + p_no
    return p_yes / total if total > 0 else 0.5

def run_inference(prompts, label=""):
    import time
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    out = []
    for prompt in tqdm(prompts, desc=label):
        for attempt in range(8):
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1, logprobs=True, top_logprobs=20,
                )
                out.append(prob_yes_from_logprobs(
                    resp.choices[0].logprobs.content[0].top_logprobs))
                break
            except openai.RateLimitError:
                wait = 2 ** attempt
                time.sleep(wait)
        else:
            out.append(0.5)  # fallback if all retries fail
    return out

# Run inference once — reused across all reward functions
print("\nRunning GPT-4o inference (once, shared across all reward functions)...")
inference_cache = {}
for split_name in ["val", "test"]:
    inference_cache[split_name] = {}
    for m in WINDOWS:
        prompts  = data[split_name][f"s{m}"].tolist()
        conv_ids = data[split_name]["conversation_id"].tolist()
        probs = run_inference(prompts, label=f"{split_name} {m}s")
        inference_cache[split_name][m] = dict(zip(conv_ids, probs))

for split_name in ["val", "test"]:
    for m in WINDOWS:
        data[split_name][f"prob_yes_{m}"] = (
            data[split_name]["conversation_id"]
            .map(inference_cache[split_name][m]).fillna(0))

print("\nROC-AUC scores:")
for split_name in ["val", "test"]:
    for m in WINDOWS:
        auc = roc_auc_score(data[split_name]["is_sale"],
                            data[split_name][f"prob_yes_{m}"])
        print(f"  {split_name} {m}s: {auc:.3f}")

# ── Simulation ─────────────────────────────────────────────────────────────────
def simulate(thresholds, preds, windows, cost, benefit):
    m1, m2, m3, m4 = windows
    t1, t2, t3, t4 = [thresholds[m] for m in windows]

    quit_m1    = preds[preds[f"prob_yes_{m1}"] < t1]
    after_m1   = preds[preds[f"prob_yes_{m1}"] >= t1]
    ended_m1m2 = after_m1[after_m1["duration"] < m2]
    after_m2   = after_m1[after_m1["duration"] >= m2]
    quit_m2    = after_m2[after_m2[f"prob_yes_{m2}"] < t2]
    after_m2c  = after_m2[after_m2[f"prob_yes_{m2}"] >= t2]
    ended_m2m3 = after_m2c[after_m2c["duration"] < m3]
    after_m3   = after_m2c[after_m2c["duration"] >= m3]
    quit_m3    = after_m3[after_m3[f"prob_yes_{m3}"] < t3]
    after_m3c  = after_m3[after_m3[f"prob_yes_{m3}"] >= t3]
    ended_m3m4 = after_m3c[after_m3c["duration"] < m4]
    after_m4   = after_m3c[after_m3c["duration"] >= m4]
    quit_m4    = after_m4[after_m4[f"prob_yes_{m4}"] < t4]
    continued  = after_m4[after_m4[f"prob_yes_{m4}"] >= t4]

    total = (len(quit_m1) + len(ended_m1m2) + len(quit_m2) + len(ended_m2m3) +
             len(quit_m3) + len(ended_m3m4) + len(quit_m4) + len(continued))
    assert total == len(preds), f"Partition error: {total} != {len(preds)}"

    sales = (ended_m1m2["is_sale"].sum() + ended_m2m3["is_sale"].sum() +
             ended_m3m4["is_sale"].sum() + continued["is_sale"].sum())
    time  = (len(quit_m1)*m1 + len(quit_m2)*m2 + len(quit_m3)*m3 + len(quit_m4)*m4 +
             ended_m1m2["duration"].sum() + ended_m2m3["duration"].sum() +
             ended_m3m4["duration"].sum() + continued["duration"].sum())

    reward = sales * benefit - time * cost
    return reward, reward / len(preds), int(sales), time

# ── Tune thresholds backward: m4 → m3 → m2 → m1 ──────────────────────────────
def tune_thresholds(val_preds, windows, cost, benefit, N=10000):
    thresholds = {m: 0.0 for m in windows}
    for m in reversed(windows):
        best_r, best_t = -1e12, thresholds[m]
        col = val_preds[f"prob_yes_{m}"]
        for thr in np.linspace(col.min()-1e-12, col.max()+1e-12, N):
            thresholds[m] = thr
            _, ar, _, _ = simulate(thresholds, val_preds, windows, cost, benefit)
            if ar > best_r:
                best_r, best_t = ar, thr
        thresholds[m] = best_t
    return thresholds

# ── Grid ───────────────────────────────────────────────────────────────────────
records = []

for rf_name, rf in REWARD_FUNCTIONS.items():
    cost, benefit = rf["cost"], rf["benefit"]
    print(f"\n{'='*60}")
    print(f"{rf_name}  (cost={cost}, benefit={benefit})")

    osap = compute_optimal_actions(data["train"], WINDOWS, cost, benefit)
    print(f"  Train action dist: {osap['action'].value_counts().to_dict()}")

    val_preds  = data["val"].copy()
    test_preds = data["test"].copy()

    base_thr = {m: 0.0 for m in WINDOWS}
    tr_base, ar_base, ts_base, tt_base = simulate(
        base_thr, test_preds, WINDOWS, cost, benefit)

    best_thr = tune_thresholds(val_preds, WINDOWS, cost, benefit)
    tr_agent, ar_agent, ts_agent, tt_agent = simulate(
        best_thr, test_preds, WINDOWS, cost, benefit)

    auc_per_window = {
        m: round(roc_auc_score(test_preds["is_sale"],
                               test_preds[f"prob_yes_{m}"]), 3)
        for m in WINDOWS}

    print(f"  Thresholds: " + " | ".join(
        f"{m}s={best_thr[m]:.4f}" for m in WINDOWS))
    print(f"  Baseline — reward: {tr_base:.2f}, sales: {ts_base}, time: {tt_base:.1f}s")
    print(f"  Agent    — reward: {tr_agent:.2f}, sales: {ts_agent}, time: {tt_agent:.1f}s")
    print(f"  Time saved: {tt_base-tt_agent:.1f}s ({(tt_base-tt_agent)/tt_base*100:.1f}%)")
    print(f"  Sales lost: {ts_base - ts_agent}")

    records.append({
        "run": rf_name,
        "cost_per_s": cost, "benefit_per_sale": benefit,
        **{f"roc_auc_{m}s": auc_per_window[m] for m in WINDOWS},
        **{f"threshold_{m}s": round(best_thr[m], 4) for m in WINDOWS},
        "baseline_avg_reward": round(ar_base, 2),
        "baseline_sales": ts_base,
        "baseline_time_s": round(tt_base, 1),
        "agent_avg_reward": round(ar_agent, 2),
        "agent_sales": ts_agent,
        "agent_time_s": round(tt_agent, 1),
        "sales_lost": ts_base - ts_agent,
        "time_saved_s": round(tt_base - tt_agent, 1),
        "time_saved_pct": round((tt_base - tt_agent) / tt_base * 100, 1),
        "reward_gained": round(tr_agent - tr_base, 2),
    })

# ── Save ───────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
out_csv = os.path.join(os.path.dirname(__file__), "data",
                       "results_dealerships_4window.csv")
results_df.to_csv(out_csv, index=False)
print(f"\nResults saved to {out_csv}")

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
auc_cols = [f"roc_auc_{m}s" for m in WINDOWS]
cols = ["run"] + auc_cols + ["baseline_avg_reward", "agent_avg_reward",
                              "sales_lost", "time_saved_pct", "reward_gained"]
print(results_df[cols].to_string(index=False))
