"""
Optimal-stopping replication — negotiation (NL) data, GPT-4o backbone.
Converted from notebooks/optimal-stopping-nego.ipynb for local execution.
"""

import math, os
import numpy as np
import pandas as pd
import openai
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "nego-data-final.csv")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY environment variable before running.")

COST_PER_UNIT_TIME       = 0.1
BENEFIT_PER_POSITIVE_OUTCOME = 10.0
DECISION_OPPORTUNITIES   = [45, 60]   # seconds
m1, m2                   = sorted(DECISION_OPPORTUNITIES)

OPENAI_MODEL = "gpt-4o"

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)

df["is_sale"] = df["outcome"].str.strip().apply(
    lambda x: 1 if x == "sale" else 0 if x == "no sale" else np.nan)
df["duration"] = df.groupby("conversation_id")["end_time"].transform("max")

print(f"  {df['conversation_id'].nunique()} conversations, "
      f"{df['is_sale'].sum():.0f} sales, {(1-df['is_sale']).sum():.0f} no-sales")

# ── Train / val / test split (conversation-level, stratified) ──────────────────
conv_meta = df[["conversation_id", "is_sale"]].drop_duplicates()
all_ids     = conv_meta["conversation_id"].values
all_labels  = conv_meta["is_sale"].values

train_ids, test_ids, train_y, test_y = train_test_split(
    all_ids, all_labels, test_size=0.25, random_state=42, stratify=all_labels)
train_ids, val_ids, train_y, val_y = train_test_split(
    train_ids, train_y, test_size=0.25, random_state=42, stratify=train_y)

splits = {
    "train": df[df["conversation_id"].isin(train_ids)],
    "val":   df[df["conversation_id"].isin(val_ids)],
    "test":  df[df["conversation_id"].isin(test_ids)],
}
print(f"  Split: {len(train_ids)} train | {len(val_ids)} val | {len(test_ids)} test")

# ── Build partial transcripts at each decision window ─────────────────────────
def build_transcripts(split_df):
    split_df = split_df.copy()
    split_df["turn_text"] = ("Speaker " + split_df["speaker_id"].astype(str)
                             + ": " + split_df["text"])
    result = split_df[["conversation_id", "duration", "is_sale"]].drop_duplicates()
    for m in [m1, m2]:
        partial = (split_df[split_df["end_time"] < m]
                   .groupby("conversation_id")["turn_text"]
                   .apply(lambda x: "\n".join(x))
                   .reset_index(name=f"transcript_{m}"))
        result = result.merge(partial, on="conversation_id", how="left")
        result[f"transcript_{m}"] = result[f"transcript_{m}"].fillna("")
    return result

data = {k: build_transcripts(v) for k, v in splits.items()}

# ── Wrap transcripts in prompts ────────────────────────────────────────────────
def make_prompt(transcript, t):
    return (f"Below is the first {t} seconds of the sales call between "
            f"the sales agent Speaker 0 and the customer Speaker 1:\n"
            f"{transcript}\n"
            f"Will this call end in a sale (respond with 'yes' or 'no'):  ")

for split_df in data.values():
    for m in [m1, m2]:
        split_df[f"s{m}"] = split_df[f"transcript_{m}"].apply(make_prompt, t=m)

# ── Compute optimal state-action pairs (backward induction) ───────────────────
def compute_optimal_actions(split_df):
    d = split_df.copy()
    d[f"rq{m1}"] = -m1 * COST_PER_UNIT_TIME
    d[f"rq{m2}"] = (d["is_sale"] * BENEFIT_PER_POSITIVE_OUTCOME
                    * (d["duration"] <= m2).astype(int)
                    - d["duration"].apply(lambda x: min(m2, x)) * COST_PER_UNIT_TIME)
    d[f"rc{m2}"] = (d["is_sale"] * BENEFIT_PER_POSITIVE_OUTCOME
                    - d["duration"] * COST_PER_UNIT_TIME)
    d["max_reward"] = d[[f"rq{m1}", f"rq{m2}", f"rc{m2}"]].max(axis=1)

    for col in [f"a{m1}", f"a{m2}"]:
        d[col] = np.nan
    d.loc[d["max_reward"] == d[f"rq{m1}"], [f"a{m1}", f"a{m2}"]] = "no"
    d.loc[d["max_reward"] == d[f"rq{m2}"], f"a{m1}"] = "yes"
    d.loc[d["max_reward"] == d[f"rq{m2}"], f"a{m2}"] = "no"
    d.loc[d["max_reward"] == d[f"rc{m2}"], [f"a{m1}", f"a{m2}"]] = "yes"

    rows = []
    for m in [m1, m2]:
        subset = d[d["duration"] >= m][
            ["conversation_id", f"s{m}", f"a{m}", "is_sale", "duration"]
        ].copy()
        subset.columns = ["conversation_id", "state", "action", "is_sale", "duration"]
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)

osap = {k: compute_optimal_actions(v) for k, v in data.items()}
print("\nOptimal action distribution (train):")
print(osap["train"]["action"].value_counts())
if osap["train"]["action"].nunique() == 1:
    print("WARNING: Only one action class in training set — imitation learning may fail.")

# ── OpenAI inference ───────────────────────────────────────────────────────────
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def prob_yes_from_logprobs(top_logprobs):
    lp = {e.token.strip().lower(): e.logprob for e in top_logprobs}
    p_yes = math.exp(lp.get("yes", -100.0))
    p_no  = math.exp(lp.get("no",  -100.0))
    total = p_yes + p_no
    return p_yes / total if total > 0 else 0.5

def run_openai_inference(prompts):
    out = []
    for prompt in tqdm(prompts):
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )
        out.append(prob_yes_from_logprobs(
            resp.choices[0].logprobs.content[0].top_logprobs))
    return out

print("\nRunning GPT-4o inference on test set...")
osap["test"]["prob_yes"] = run_openai_inference(osap["test"]["state"].tolist())

print("Running GPT-4o inference on val set...")
osap["val"]["prob_yes"] = run_openai_inference(osap["val"]["state"].tolist())

# ── Merge predictions back ─────────────────────────────────────────────────────
def merge_predictions(split_data, split_osap):
    result = split_data.copy()
    for m in [m1, m2]:
        preds = split_osap[["conversation_id", "state", "prob_yes"]].copy()
        result = (result
                  .merge(preds, left_on=["conversation_id", f"s{m}"],
                         right_on=["conversation_id", "state"],
                         how="left", validate="one_to_one")
                  .drop(columns=["state"])
                  .rename(columns={"prob_yes": f"prob_yes_{m}"}))
        result[f"prob_yes_{m}"] = result[f"prob_yes_{m}"].fillna(0)
    return result

test_preds = merge_predictions(data["test"], osap["test"])
val_preds  = merge_predictions(data["val"],  osap["val"])

print("\nROC-AUC scores:")
for label, preds in [("Val", val_preds), ("Test", test_preds)]:
    for m in [m1, m2]:
        auc = roc_auc_score(preds["is_sale"], preds[f"prob_yes_{m}"])
        print(f"  {label} {m}s: {auc:.3f}")

# ── Simulate threshold-based stopping policy ───────────────────────────────────
def simulate_threshold(thr_m1, thr_m2, preds):
    quit_m1   = preds[preds[f"prob_yes_{m1}"] < thr_m1]
    ended_m1  = preds[(preds[f"prob_yes_{m1}"] >= thr_m1) & (preds["duration"] < m2)]
    quit_m2   = preds[(preds[f"prob_yes_{m1}"] >= thr_m1) &
                      (preds[f"prob_yes_{m2}"] < thr_m2) & (preds["duration"] >= m2)]
    cont_m2   = preds[(preds[f"prob_yes_{m1}"] >= thr_m1) &
                      (preds[f"prob_yes_{m2}"] >= thr_m2) & (preds["duration"] >= m2)]
    assert len(quit_m1)+len(ended_m1)+len(quit_m2)+len(cont_m2) == len(preds)

    total_sales = ended_m1["is_sale"].sum() + cont_m2["is_sale"].sum()
    total_time  = (len(quit_m1)*m1 + len(quit_m2)*m2
                   + ended_m1["duration"].sum() + cont_m2["duration"].sum())
    total_reward   = total_sales * BENEFIT_PER_POSITIVE_OUTCOME - total_time * COST_PER_UNIT_TIME
    average_reward = total_reward / len(preds)
    return total_reward, average_reward, total_sales, total_time

print("\nBaseline (no stopping agent) on test set:")
tr, ar, ts, tt = simulate_threshold(0, 0, test_preds)
print(f"  Total reward: {tr:.2f} | Avg reward: {ar:.2f} | Sales: {ts} | Time: {tt:.1f}s")

# Tune thresholds on validation set (backward induction)
N = 10000
best_thr = {}

best_reward = -1e9
for thr in np.linspace(val_preds[f"prob_yes_{m2}"].min()-1e-12,
                       val_preds[f"prob_yes_{m2}"].max()+1e-12, N):
    _, ar, _, _ = simulate_threshold(0, thr, val_preds)
    if ar > best_reward:
        best_reward, best_thr[m2] = ar, thr

best_reward = -1e9
for thr in np.linspace(val_preds[f"prob_yes_{m1}"].min()-1e-12,
                       val_preds[f"prob_yes_{m1}"].max()+1e-12, N):
    _, ar, _, _ = simulate_threshold(thr, best_thr[m2], val_preds)
    if ar > best_reward:
        best_reward, best_thr[m1] = ar, thr

print(f"\nBest thresholds — m1={m1}s: {best_thr[m1]:.4f} | m2={m2}s: {best_thr[m2]:.4f}")

print("\nWith stopping agent on test set:")
tr_a, ar_a, ts_a, tt_a = simulate_threshold(best_thr[m1], best_thr[m2], test_preds)
print(f"  Total reward: {tr_a:.2f} | Avg reward: {ar_a:.2f} | Sales: {ts_a} | Time: {tt_a:.1f}s")

print("\nComparative results:")
print(f"  Sales lost:    {ts - ts_a}")
print(f"  Time saved:    {tt - tt_a:.1f}s  ({(tt-tt_a)/tt*100:.1f}%)")
print(f"  Reward gained: {tr_a - tr:.2f}")
