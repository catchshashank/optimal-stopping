"""
3 window pairs x 4 reward functions = 12 runs on dealerships-nego.csv.

Window pairs (2 checkpoints each):
  WP1: m1=175s, m2=260s
  WP2: m1=260s, m2=350s
  WP3: m1=350s, m2=400s

Reward functions:
  RF1: cost=0.1,  benefit=10
  RF2: cost=0.5,  benefit=10
  RF3: cost=1.0,  benefit=1000
  RF4: cost=5.0,  benefit=1000

Inference runs once per window pair (3x), shared across all reward functions.
Backward induction tunes thresholds m2 → m1 on val set.
"""

import math, os, time
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

WINDOW_PAIRS = {
    "WP1_175_260": [175, 260],
    "WP2_260_350": [260, 350],
    "WP3_350_400": [350, 400],
}

REWARD_FUNCTIONS = {
    "RF1_cost0.1_ben10":    {"cost": 0.1,  "benefit": 10.0},
    "RF2_cost0.5_ben10":    {"cost": 0.5,  "benefit": 10.0},
    "RF3_cost1.0_ben1000":  {"cost": 1.0,  "benefit": 1000.0},
    "RF4_cost5.0_ben1000":  {"cost": 5.0,  "benefit": 1000.0},
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_transcripts(split_df, m1, m2):
    d = split_df.copy()
    d["turn_text"] = "Speaker " + d["speaker_id"].astype(str) + ": " + d["text"]
    result = d[["conversation_id", "duration", "is_sale"]].drop_duplicates()
    for m in [m1, m2]:
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

def prob_yes_from_logprobs(top_logprobs):
    lp = {e.token.strip().lower(): e.logprob for e in top_logprobs}
    p_yes = math.exp(lp.get("yes", -100.0))
    p_no  = math.exp(lp.get("no",  -100.0))
    total = p_yes + p_no
    return p_yes / total if total > 0 else 0.5

def run_inference(prompts, label=""):
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
                time.sleep(2 ** attempt)
        else:
            out.append(0.5)
    return out

def compute_optimal_actions(split_df, m1, m2, cost, benefit):
    d = split_df.copy()
    d["rq_m1"] = -m1 * cost
    d["rq_m2"] = (d["is_sale"] * benefit * (d["duration"] <= m2).astype(int)
                  - d["duration"].apply(lambda x: min(m2, x)) * cost)
    d["rc"]    = d["is_sale"] * benefit - d["duration"] * cost
    d["max_reward"] = d[["rq_m1", "rq_m2", "rc"]].max(axis=1)

    for col in [f"a{m1}", f"a{m2}"]:
        d[col] = pd.NA
        d[col] = d[col].astype(object)

    d.loc[d["max_reward"] == d["rq_m1"], [f"a{m1}", f"a{m2}"]] = "no"
    d.loc[d["max_reward"] == d["rq_m2"], f"a{m1}"] = "yes"
    d.loc[d["max_reward"] == d["rq_m2"], f"a{m2}"] = "no"
    d.loc[d["max_reward"] == d["rc"],    f"a{m1}"] = "yes"
    d.loc[d["max_reward"] == d["rc"],    f"a{m2}"] = "yes"

    rows = []
    for m in [m1, m2]:
        subset = d[d["duration"] >= m][
            ["conversation_id", f"s{m}", f"a{m}", "is_sale", "duration"]
        ].copy()
        subset.columns = ["conversation_id", "state", "action", "is_sale", "duration"]
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)

def simulate(thr_m1, thr_m2, preds, m1, m2, cost, benefit):
    quit_m1  = preds[preds[f"prob_yes_{m1}"] < thr_m1]
    after_m1 = preds[preds[f"prob_yes_{m1}"] >= thr_m1]
    ended    = after_m1[after_m1["duration"] < m2]
    after_m2 = after_m1[after_m1["duration"] >= m2]
    quit_m2  = after_m2[after_m2[f"prob_yes_{m2}"] < thr_m2]
    cont_m2  = after_m2[after_m2[f"prob_yes_{m2}"] >= thr_m2]
    assert len(quit_m1)+len(ended)+len(quit_m2)+len(cont_m2) == len(preds)

    sales  = ended["is_sale"].sum() + cont_m2["is_sale"].sum()
    t_time = (len(quit_m1)*m1 + len(quit_m2)*m2
              + ended["duration"].sum() + cont_m2["duration"].sum())
    reward = sales * benefit - t_time * cost
    return reward, reward / len(preds), int(sales), t_time

def tune_thresholds(val_preds, m1, m2, cost, benefit, N=10000):
    best_thr = {m1: 0.0, m2: 0.0}
    # Backward: tune m2 first, then m1
    best_r = -1e12
    for thr in np.linspace(val_preds[f"prob_yes_{m2}"].min()-1e-12,
                           val_preds[f"prob_yes_{m2}"].max()+1e-12, N):
        _, ar, _, _ = simulate(0, thr, val_preds, m1, m2, cost, benefit)
        if ar > best_r:
            best_r, best_thr[m2] = ar, thr

    best_r = -1e12
    for thr in np.linspace(val_preds[f"prob_yes_{m1}"].min()-1e-12,
                           val_preds[f"prob_yes_{m1}"].max()+1e-12, N):
        _, ar, _, _ = simulate(thr, best_thr[m2], val_preds, m1, m2, cost, benefit)
        if ar > best_r:
            best_r, best_thr[m1] = ar, thr
    return best_thr

# ── Main grid ──────────────────────────────────────────────────────────────────
inference_cache = {}
records = []

for wp_name, (m1, m2) in WINDOW_PAIRS.items():
    print(f"\n{'='*60}")
    print(f"Window pair: {wp_name}  ({m1}s, {m2}s)")
    print(f"{'='*60}")

    data = {k: build_transcripts(v, m1, m2) for k, v in splits.items()}
    for split_df in data.values():
        for m in [m1, m2]:
            split_df[f"s{m}"] = split_df[f"transcript_{m}"].apply(make_prompt, t=m)

    # Inference once per window pair
    if wp_name not in inference_cache:
        print(f"Running GPT-4o inference...")
        cache = {}
        for split_name in ["val", "test"]:
            cache[split_name] = {}
            for m in [m1, m2]:
                prompts  = data[split_name][f"s{m}"].tolist()
                conv_ids = data[split_name]["conversation_id"].tolist()
                probs = run_inference(prompts, label=f"{split_name} {m}s")
                cache[split_name][m] = dict(zip(conv_ids, probs))
        inference_cache[wp_name] = cache

    cache = inference_cache[wp_name]
    for split_name in ["val", "test"]:
        for m in [m1, m2]:
            data[split_name][f"prob_yes_{m}"] = (
                data[split_name]["conversation_id"]
                .map(cache[split_name][m]).fillna(0))

    print("ROC-AUC (conditional: duration >= m only):")
    for split_name in ["val", "test"]:
        for m in [m1, m2]:
            sub = data[split_name][data[split_name]["duration"] >= m]
            auc = roc_auc_score(sub["is_sale"], sub[f"prob_yes_{m}"])
            print(f"  {split_name} {m}s: {auc:.3f}  (n={len(sub)})")

    for rf_name, rf in REWARD_FUNCTIONS.items():
        cost, benefit = rf["cost"], rf["benefit"]
        print(f"\n  -- {rf_name} --")

        osap = compute_optimal_actions(data["train"], m1, m2, cost, benefit)
        print(f"     Action dist: {osap['action'].value_counts().to_dict()}")

        val_preds  = data["val"].copy()
        test_preds = data["test"].copy()

        _, ar_base, ts_base, tt_base = simulate(0, 0, test_preds, m1, m2, cost, benefit)
        best_thr = tune_thresholds(val_preds, m1, m2, cost, benefit)
        tr_agent, ar_agent, ts_agent, tt_agent = simulate(
            best_thr[m1], best_thr[m2], test_preds, m1, m2, cost, benefit)

        sub_m1 = test_preds[test_preds["duration"] >= m1]
        sub_m2 = test_preds[test_preds["duration"] >= m2]
        auc_m1 = round(roc_auc_score(sub_m1["is_sale"], sub_m1[f"prob_yes_{m1}"]), 3)
        auc_m2 = round(roc_auc_score(sub_m2["is_sale"], sub_m2[f"prob_yes_{m2}"]), 3)

        print(f"     Thresholds: {m1}s={best_thr[m1]:.4f} | {m2}s={best_thr[m2]:.4f}")
        print(f"     Baseline — avg reward: {ar_base:.2f} | sales: {ts_base} | time: {tt_base:.1f}s")
        print(f"     Agent    — avg reward: {ar_agent:.2f} | sales: {ts_agent} | time: {tt_agent:.1f}s")
        print(f"     Time saved: {tt_base-tt_agent:.1f}s ({(tt_base-tt_agent)/tt_base*100:.1f}%) | Sales lost: {ts_base-ts_agent}")

        records.append({
            "run": f"{rf_name} x {wp_name}",
            "window_pair": wp_name, "m1_s": m1, "m2_s": m2,
            "cost_per_s": cost, "benefit_per_sale": benefit,
            "roc_auc_test_m1": auc_m1, "roc_auc_test_m2": auc_m2,
            "threshold_m1": round(best_thr[m1], 4),
            "threshold_m2": round(best_thr[m2], 4),
            "baseline_avg_reward": round(ar_base, 2),
            "baseline_sales": ts_base, "baseline_time_s": round(tt_base, 1),
            "agent_avg_reward": round(ar_agent, 2),
            "agent_sales": ts_agent, "agent_time_s": round(tt_agent, 1),
            "sales_lost": ts_base - ts_agent,
            "time_saved_s": round(tt_base - tt_agent, 1),
            "time_saved_pct": round((tt_base - tt_agent) / tt_base * 100, 1),
            "reward_gained": round(tr_agent - (ar_base * len(test_preds)), 2),
        })

# ── Save ───────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
out_csv = os.path.join(os.path.dirname(__file__), "data",
                       "results_dealerships_3pair_grid.csv")
results_df.to_csv(out_csv, index=False)
print(f"\nResults saved to {out_csv}")

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
cols = ["run", "roc_auc_test_m1", "roc_auc_test_m2",
        "threshold_m1", "threshold_m2", "baseline_avg_reward",
        "agent_avg_reward", "sales_lost", "time_saved_pct", "reward_gained"]
print(results_df[cols].to_string(index=False))
