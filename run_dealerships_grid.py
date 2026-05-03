"""
2x2 grid run on dealerships-nego.csv — same parameter space as NL grid for direct comparison.
Real diarized timestamps (not estimated), perfectly balanced 48-conversation dataset.
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

REWARD_FUNCTIONS = {
    "RF1_cost1.0_ben1000": {"cost": 1.0, "benefit": 1000.0},
    "RF2_cost5.0_ben1000": {"cost": 5.0, "benefit": 1000.0},
}
DECISION_PAIRS = {
    "DP1_115_230": [115, 230],
    "DP2_90_180":  [90,  180],
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
print(f"  {len(train_ids)} train | {len(val_ids)} val | {len(test_ids)} test conversations")
print(f"  Test sales: {test_y.sum()} | Test no-sales: {(1-test_y).sum()}")

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


def compute_optimal_actions(split_df, m1, m2, cost, benefit):
    d = split_df.copy()
    d[f"s{m1}"] = d[f"transcript_{m1}"].apply(make_prompt, t=m1)
    d[f"s{m2}"] = d[f"transcript_{m2}"].apply(make_prompt, t=m2)

    d[f"rq{m1}"] = -m1 * cost
    d[f"rq{m2}"] = (d["is_sale"] * benefit
                    * (d["duration"] <= m2).astype(int)
                    - d["duration"].apply(lambda x: min(m2, x)) * cost)
    d[f"rc{m2}"] = d["is_sale"] * benefit - d["duration"] * cost
    d["max_reward"] = d[[f"rq{m1}", f"rq{m2}", f"rc{m2}"]].max(axis=1)

    for col in [f"a{m1}", f"a{m2}"]:
        d[col] = pd.NA
    d[f"a{m1}"] = d[f"a{m1}"].astype(object)
    d[f"a{m2}"] = d[f"a{m2}"].astype(object)
    d.loc[d["max_reward"] == d[f"rq{m1}"], [f"a{m1}", f"a{m2}"]] = "no"
    d.loc[d["max_reward"] == d[f"rq{m2}"], f"a{m1}"] = "yes"
    d.loc[d["max_reward"] == d[f"rq{m2}"], f"a{m2}"] = "no"
    d.loc[d["max_reward"] == d[f"rc{m2}"], f"a{m1}"] = "yes"
    d.loc[d["max_reward"] == d[f"rc{m2}"], f"a{m2}"] = "yes"

    rows = []
    for m in [m1, m2]:
        subset = d[d["duration"] >= m][
            ["conversation_id", f"s{m}", f"a{m}", "is_sale", "duration"]
        ].copy()
        subset.columns = ["conversation_id", "state", "action", "is_sale", "duration"]
        rows.append(subset)
    return pd.concat(rows, ignore_index=True), d


def prob_yes_from_logprobs(top_logprobs):
    lp = {e.token.strip().lower(): e.logprob for e in top_logprobs}
    p_yes = math.exp(lp.get("yes", -100.0))
    p_no  = math.exp(lp.get("no",  -100.0))
    total = p_yes + p_no
    return p_yes / total if total > 0 else 0.5


def run_openai_inference(prompts, label=""):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    out = []
    for prompt in tqdm(prompts, desc=label):
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


def simulate_threshold(thr_m1, thr_m2, preds, m1, m2, cost, benefit):
    quit_m1  = preds[preds[f"prob_yes_{m1}"] < thr_m1]
    ended_m1 = preds[(preds[f"prob_yes_{m1}"] >= thr_m1) & (preds["duration"] < m2)]
    quit_m2  = preds[(preds[f"prob_yes_{m1}"] >= thr_m1) &
                     (preds[f"prob_yes_{m2}"] < thr_m2) & (preds["duration"] >= m2)]
    cont_m2  = preds[(preds[f"prob_yes_{m1}"] >= thr_m1) &
                     (preds[f"prob_yes_{m2}"] >= thr_m2) & (preds["duration"] >= m2)]
    assert len(quit_m1)+len(ended_m1)+len(quit_m2)+len(cont_m2) == len(preds)

    total_sales  = ended_m1["is_sale"].sum() + cont_m2["is_sale"].sum()
    total_time   = (len(quit_m1)*m1 + len(quit_m2)*m2
                    + ended_m1["duration"].sum() + cont_m2["duration"].sum())
    total_reward   = total_sales * benefit - total_time * cost
    average_reward = total_reward / len(preds)
    return total_reward, average_reward, int(total_sales), total_time


# ── Grid loop ──────────────────────────────────────────────────────────────────
inference_cache = {}
records = []

for dp_name, windows in DECISION_PAIRS.items():
    m1, m2 = sorted(windows)
    print(f"\n{'='*60}")
    print(f"Window pair: {dp_name}  ({m1}s, {m2}s)")
    print(f"{'='*60}")

    data = {k: build_transcripts(v, m1, m2) for k, v in splits.items()}
    for split_df in data.values():
        for m in [m1, m2]:
            split_df[f"s{m}"] = split_df[f"transcript_{m}"].apply(make_prompt, t=m)

    if dp_name not in inference_cache:
        print(f"\nRunning GPT-4o inference for {dp_name}...")
        cache = {}
        for split_name in ["val", "test"]:
            split_df = data[split_name]
            cache[split_name] = {}
            for m in [m1, m2]:
                prompts  = split_df[f"s{m}"].tolist()
                conv_ids = split_df["conversation_id"].tolist()
                probs = run_openai_inference(
                    prompts, label=f"{dp_name} {split_name} {m}s")
                cache[split_name][m] = dict(zip(conv_ids, probs))
        inference_cache[dp_name] = cache

    cache = inference_cache[dp_name]
    for split_name in ["val", "test"]:
        split_df = data[split_name]
        for m in [m1, m2]:
            split_df[f"prob_yes_{m}"] = split_df["conversation_id"].map(
                cache[split_name][m]).fillna(0)

    print("\n  ROC-AUC:")
    for split_name in ["val", "test"]:
        for m in [m1, m2]:
            auc = roc_auc_score(data[split_name]["is_sale"],
                                data[split_name][f"prob_yes_{m}"])
            print(f"    {split_name} {m}s: {auc:.3f}")

    for rf_name, rf in REWARD_FUNCTIONS.items():
        cost, benefit = rf["cost"], rf["benefit"]
        print(f"\n  -- {rf_name} (cost={cost}, benefit={benefit}) --")

        osap_train, _ = compute_optimal_actions(data["train"], m1, m2, cost, benefit)
        print(f"     Train action dist: {osap_train['action'].value_counts().to_dict()}")

        val_preds  = data["val"].copy()
        test_preds = data["test"].copy()

        tr_base, ar_base, ts_base, tt_base = simulate_threshold(
            0, 0, test_preds, m1, m2, cost, benefit)

        N = 10000
        best_thr = {}

        best_r = -1e9
        for thr in np.linspace(val_preds[f"prob_yes_{m2}"].min()-1e-12,
                               val_preds[f"prob_yes_{m2}"].max()+1e-12, N):
            _, ar, _, _ = simulate_threshold(0, thr, val_preds, m1, m2, cost, benefit)
            if ar > best_r:
                best_r, best_thr[m2] = ar, thr

        best_r = -1e9
        for thr in np.linspace(val_preds[f"prob_yes_{m1}"].min()-1e-12,
                               val_preds[f"prob_yes_{m1}"].max()+1e-12, N):
            _, ar, _, _ = simulate_threshold(thr, best_thr[m2], val_preds, m1, m2, cost, benefit)
            if ar > best_r:
                best_r, best_thr[m1] = ar, thr

        tr_agent, ar_agent, ts_agent, tt_agent = simulate_threshold(
            best_thr[m1], best_thr[m2], test_preds, m1, m2, cost, benefit)

        auc_m1 = roc_auc_score(test_preds["is_sale"], test_preds[f"prob_yes_{m1}"])
        auc_m2 = roc_auc_score(test_preds["is_sale"], test_preds[f"prob_yes_{m2}"])

        print(f"     Baseline  — reward: {tr_base:.2f}, sales: {ts_base}, time: {tt_base:.1f}s")
        print(f"     Agent     — reward: {tr_agent:.2f}, sales: {ts_agent}, time: {tt_agent:.1f}s")
        print(f"     Thresholds: {m1}s={best_thr[m1]:.4f}, {m2}s={best_thr[m2]:.4f}")
        print(f"     Time saved: {tt_base-tt_agent:.1f}s ({(tt_base-tt_agent)/tt_base*100:.1f}%)")
        print(f"     Sales lost: {ts_base - ts_agent}")

        records.append({
            "dataset": "dealerships",
            "run": f"{rf_name} x {dp_name}",
            "reward_fn": rf_name,
            "windows": dp_name,
            "m1_s": m1, "m2_s": m2,
            "cost_per_s": cost,
            "benefit_per_sale": benefit,
            "roc_auc_test_m1": round(auc_m1, 3),
            "roc_auc_test_m2": round(auc_m2, 3),
            "threshold_m1": round(best_thr[m1], 4),
            "threshold_m2": round(best_thr[m2], 4),
            "baseline_total_reward": round(tr_base, 2),
            "baseline_avg_reward": round(ar_base, 2),
            "baseline_sales": ts_base,
            "baseline_time_s": round(tt_base, 1),
            "agent_total_reward": round(tr_agent, 2),
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
out_csv = os.path.join(os.path.dirname(__file__), "data", "results_dealerships_grid_ben1000.csv")
results_df.to_csv(out_csv, index=False)
print(f"\nResults saved to {out_csv}")

print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
cols = ["run", "roc_auc_test_m1", "roc_auc_test_m2",
        "threshold_m1", "threshold_m2",
        "baseline_avg_reward", "agent_avg_reward",
        "sales_lost", "time_saved_pct", "reward_gained"]
print(results_df[cols].to_string(index=False))
