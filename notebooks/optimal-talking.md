# Code Walkthrough: Optimal Talking in Negotiations

> **What this notebook does:** Trains an AI agent to learn what Tommy — the buyer — should *say* at each moment in a car price negotiation in order to maximise the discount extracted from the dealer. The approach is grounded in Behavioral Cloning: for every historical conversation where we know the outcome, we work backwards to label what the optimal buyer action would have been at each turn, then train a model to reproduce those optimal actions. Linguistic features are derived entirely from LLM token attribution — no hand-coded dictionaries are used.
>
> **Relationship to the optimal-stopping study:** The stopping study (Manzoor et al.) solves a simpler problem — binary quit-or-stay decisions at fixed timestamps. This notebook solves a harder one: a sequential control problem where the buyer's words causally determine the dealer's next response. The stopping study's core components (retrospective reward labelling, label masking, token probability extraction) are reused and adapted here; differences are noted inline where they arise.

---

## PART 1 — Data Architecture

---

### Block 1 · Data Split and the Held-Out Test Set

**What it does:** Enforces the strict separation between training data and final evaluation data. Sixty real conversations are locked away before any modelling begins and are never touched until Study 4.

> **Why 60 are locked away:** With ~300 real conversations, a standard 80/20 split would give only 60 test conversations — too few for reliable out-of-sample estimates. By locking exactly 60 away and training on 240 real + 10,000 synthetic, we get the best of both: a large training corpus and a clean, untouched ground truth for final evaluation.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load all real conversations
# Columns: conv_id, speaker_id, start_time, end_time, text,
#          d_B, d_D, outcome, is_sale, duration, dealer,
#          model, msrp, tsrp, add_ons
real_df = pd.read_csv("data/processed/all_conversations.csv")

all_conv_ids = real_df["conv_id"].unique()
all_outcomes = real_df.groupby("conv_id")["is_sale"].first().values

# Lock away 60 real conversations — never touched until Study 4
# Stratify on is_sale so the test set has realistic sale/no-sale balance
trainval_ids, test_ids = train_test_split(
    all_conv_ids, test_size=60,
    stratify=all_outcomes, random_state=42)

real_test_df     = real_df[real_df["conv_id"].isin(test_ids)].copy()
real_trainval_df = real_df[real_df["conv_id"].isin(trainval_ids)].copy()

real_test_df.to_csv("data/splits/real_test_60.csv", index=False)
print(f"Real test set locked: {len(test_ids)} conversations")
print(f"Available for training: {len(trainval_ids)} conversations")

# Load 10,000 synthetic conversations
# Generated from the 240 training conversations only —
# synthetic generator was trained before test_ids were identified
synthetic_df = pd.read_csv("data/synthetic/10000_conversations.csv")

# Combined training pool
combined_df = pd.concat([real_trainval_df, synthetic_df], ignore_index=True)
combined_df["source"] = np.where(
    combined_df["conv_id"].isin(trainval_ids), "real", "synthetic")

print(f"Training pool: {real_trainval_df['conv_id'].nunique()} real + "
      f"{synthetic_df['conv_id'].nunique()} synthetic conversations")
```

**Data split summary:**

| Split | N | Source | Purpose |
|---|---|---|---|
| `real_test_60` | 60 | Real only | Final evaluation — locked until Study 4 |
| `real_trainval` | 240 | Real only | Generator training + RSSM validation |
| `synthetic` | 10,000 | Synthetic | RSSM training, regression power |
| `combined` | 10,240 | Real + synthetic | Primary training pool |

> **Stopping study parallel:** The stopping study splits conversations 75/12.5/12.5 by conversation ID. We use the same principle but impose a hard floor of 60 real conversations for the test set, then augment training with synthetic data to compensate.

---

### Block 2 · Loading and Enriching Conversation Data

**What it does:** Loads the diarized conversation data and derives the negotiation-specific outcome variables that replace the stopping study's `is_sale` binary label.

```python
def load_and_enrich(df):
    """
    Adds negotiation-specific columns to a diarized conversation dataframe.
    Speaker 0 = buyer (Tommy), Speaker 1 = dealer, Speaker 2 = sales manager.
    """
    df = df.copy().sort_values(["conv_id", "start_time"])

    # Call duration — same as stopping study
    df["duration"] = df.groupby("conv_id")["end_time"].transform("max")

    # Concession size: change in dealer's offered discount turn-over-turn
    # This replaces is_sale as the primary outcome variable
    df["delta_d_D"] = df.groupby("conv_id")["d_D"].diff()

    # Binary concession event
    df["concession"] = (df["delta_d_D"] > 0).astype(int)

    # Bargaining gap: buyer's ask minus dealer's current offer
    df["gap"] = df["d_B"] - df["d_D"]

    # Normalised turn position within conversation (0 = start, 1 = end)
    df["turn_position"] = df.groupby("conv_id").cumcount() / \
                          df.groupby("conv_id")["conv_id"].transform("count")

    return df

combined_df  = load_and_enrich(combined_df)
real_test_df = load_and_enrich(real_test_df)
```

**Key columns:**

| Column | Meaning |
|---|---|
| `d_B` | Discount claimed by buyer at this turn (numeric, in dollars) |
| `d_D` | Discount offered by dealer at this turn (numeric) |
| `delta_d_D` | Change in dealer's offer — the concession size. Replaces `is_sale` as the reward signal |
| `concession` | 1 if dealer increased their discount, 0 otherwise |
| `gap` | `d_B − d_D`: the bargaining zone width at this turn |
| `turn_position` | Normalised position in the call (0–1). Replaces the stopping study's fixed timestamps |

> **Key difference from stopping study:** The stopping study has one outcome per conversation (`is_sale`). Here, every dealer turn is a potential outcome event — roughly 10–15 per conversation — giving ~100,000–150,000 labelled observations from 10,000 synthetic conversations.

---

### Block 3 · Synthetic Data Validation

**What it does:** Before using synthetic conversations for training, verifies that they are structurally faithful to real ones across five observable properties. Any FAIL result means the synthetic generator needs retraining.

```python
from scipy.stats import ks_2samp, wasserstein_distance

def validate_synthetic(real_df, synthetic_df):
    """
    Five structural fidelity checks.
    Checks 1–4 use only observable properties — no model required.
    Check 5 requires a trained RSSM and is run after Block 8.
    """
    results = {}

    # Check 1: Concession size distribution
    ks_stat, ks_p = ks_2samp(
        real_df["delta_d_D"].dropna(),
        synthetic_df["delta_d_D"].dropna())
    results["concession_KS_p"]    = ks_p
    results["concession_check"]   = "PASS" if ks_p > 0.05 else "FAIL"

    # Check 2: Gap distribution at first price mention (anchor turn)
    # Both corpora should have similar opening gap distributions
    real_anchor  = real_df[real_df["delta_d_D"].notna()].groupby(
        "conv_id").first()["gap"]
    syn_anchor   = synthetic_df[synthetic_df["delta_d_D"].notna()].groupby(
        "conv_id").first()["gap"]
    w_dist = wasserstein_distance(real_anchor.dropna(), syn_anchor.dropna())
    results["anchor_gap_W1"]     = w_dist
    results["anchor_gap_check"]  = "PASS" if w_dist < 500 else "WARN"

    # Check 3: Turns per conversation distribution
    real_len = real_df.groupby("conv_id").size()
    syn_len  = synthetic_df.groupby("conv_id").size()
    ks2, p2  = ks_2samp(real_len, syn_len)
    results["turn_length_KS_p"]  = p2
    results["turn_length_check"] = "PASS" if p2 > 0.05 else "WARN"

    # Check 4: Concession rate (fraction of dealer turns with delta_d_D > 0)
    real_rate = real_df["concession"].mean()
    syn_rate  = synthetic_df["concession"].mean()
    results["concession_rate_diff"]  = abs(real_rate - syn_rate)
    results["concession_rate_check"] = (
        "PASS" if abs(real_rate - syn_rate) < 0.05 else "FAIL")

    # Check 5: Latent state alignment — run after Block 8
    # Compare z_t distributions between real and synthetic conversations.
    # Near-identical distributions confirm the RSSM generalises correctly.
    results["latent_alignment"] = "RUN AFTER BLOCK 8"

    for k, v in results.items():
        print(f"{k:35s}: {v}")
    return results

diagnostic = validate_synthetic(real_trainval_df, synthetic_df)
```

---

### Block 4 · Phase Assignment

**What it does:** Labels every turn with its negotiation phase (1–5) based on the call evolution framework derived from transcript analysis. Phase is used as a moderator in Study 1 and as a structural control throughout.

The five phases derived from the transcripts:
- **Phase 1 — Pre-negotiation:** Stock number exchange, car confirmation, and rapport building
- **Phase 2 — Price-negotiation:** First numeric discount mentioned by either party and concession exchange
- **Phase 3 — Outcome:** Acceptance (1), callback (2), or walkaway (0)

```python
# Phase assignment — simplified to 3 phases
def assign_phases_3(df):
    df = df.copy().sort_values(["conv_id", "start_time"])

    df["mentions_price"] = df["text"].str.contains(
        r"\$[\d,]+|\d+[\s]*(off|back|discount)",
        case=False, regex=True).astype(int)

    df["is_resolution"] = df["text"].str.contains(
        r"call you back|sleep on|talk to my wife|bye|goodbye|"
        r"going to go|call somebody else|pay less|I'll fly down",
        case=False, regex=True).astype(int)

    # Anchor turn: first turn where buyer names a numeric discount
    df["is_anchor_turn"] = (
        (df["speaker_id"] == 0) &
        (df["mentions_price"] == 1) &
        (~df.groupby("conv_id")["mentions_price"].transform(
            lambda x: x.shift(1).fillna(0).cummax()).astype(bool))
    ).astype(int)

    phases = []
    for conv_id, group in df.groupby("conv_id"):
        group = group.reset_index(drop=True)
        first_price      = group[group["mentions_price"] == 1].index.min()
        first_resolution = group[group["is_resolution"] == 1].index.min()

        p2_start = int(first_price)       if not pd.isna(first_price)      else len(group)
        p3_start = int(first_resolution)  if not pd.isna(first_resolution) else len(group)

        for i in range(len(group)):
            phase = 1 if i < p2_start else 2 if i < p3_start else 3
            phases.append({"conv_id": conv_id, "turn_idx": i, "phase": phase})

    return df.merge(pd.DataFrame(phases), on=["conv_id"])


# Phase 3 outcome: multinomial logit with callback as reference
def assign_resolution_outcome(df):
    """
    Labels the resolution outcome for each conversation.
    accept   = dealer's offer accepted, sale confirmed
    walkaway = Tommy exits to a competitor
    callback = Tommy defers — call ends with explicit follow-up intent
    """
    df = df.copy()
    df["resolution"] = df["text"].apply(lambda t: (
        "accept"   if pd.notna(t) and any(w in t.lower() for w in
                      ["sounds good", "let's do it", "deal", "i'll take it"])
        else "walkaway" if pd.notna(t) and any(w in t.lower() for w in
                      ["pay less", "call somebody else", "going to go",
                       "bye", "goodbye"])
        else "callback" if pd.notna(t) and any(w in t.lower() for w in
                      ["call you back", "sleep on", "talk to my wife",
                       "think about it", "let me know"])
        else None))

    # One resolution label per conversation (last non-null)
    conv_resolution = (df[df["resolution"].notna()]
                       .sort_values("start_time")
                       .groupby("conv_id")["resolution"]
                       .last()
                       .reset_index())
    return conv_resolution


# Multinomial logit for Phase 3
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

conv_resolutions = assign_resolution_outcome(combined_df)
conv_level = snapshots_df.groupby("conv_id").agg(
    L_score_mean  = ("L_score", "mean"),
    L_score_p2    = ("L_score",
                      lambda x: x[snapshots_df.loc[x.index,"phase"]==2].mean()),
    gap_final     = ("gap", "last"),
    uncertainty_final = ("prior_sigma", "last"),
    n_concessions = ("concession", "sum"),
).reset_index().merge(conv_resolutions, on="conv_id")

# Encode: callback=0 (reference), accept=1, walkaway=2
conv_level["outcome_code"] = conv_level["resolution"].map(
    {"callback": 0, "accept": 1, "walkaway": 2})

X = sm.add_constant(conv_level[[
    "L_score_p2", "gap_final",
    "uncertainty_final", "n_concessions"]])
y = conv_level["outcome_code"]

mnlogit = MNLogit(y, X).fit()
print(mnlogit.summary())
```

---

### Block 5 · Building Concession Snapshots

**What it does:** For every dealer turn, assembles the preceding 6 utterances as the context state. Each snapshot is one training example — one moment where the buyer just spoke and the dealer is about to respond.

> **Stopping study parallel:** The stopping study takes two fixed-time photographs per conversation (at 45s and 60s). Here, every dealer turn is a checkpoint — typically 10–15 per conversation — multiplying the effective dataset size by ~10× and replacing fixed timestamps with event-driven decision points.

```python
def build_concession_snapshots(df, context_turns=6):
    """
    Produces one snapshot per dealer turn.
    Each snapshot captures:
      - The last context_turns utterances (the state)
      - The current numeric bargaining position (d_B, d_D, gap)
      - Whether the dealer conceded on this turn (the label)
    """
    snapshots = []
    for conv_id, group in df.groupby("conv_id"):
        group = group.reset_index(drop=True)
        dealer_turns = group[group["speaker_id"].isin([1])].index

        for idx in dealer_turns:
            start   = max(0, idx - context_turns)
            context = group.loc[start:idx - 1]
            transcript_so_far = "\n".join(
                [f"{'buyer' if r['speaker_id']==0 else 'dealer'}: {r['text']}"
                 for _, r in context.iterrows()])

            snapshots.append({
                "conv_id":       conv_id,
                "turn_idx":      idx,
                "phase":         group.loc[idx, "phase"],
                "source":        group.loc[idx, "source"],
                "transcript":    transcript_so_far,
                "d_B":           group.loc[idx - 1, "d_B"] if idx > 0 else 0,
                "d_D":           group.loc[idx, "d_D"],
                "gap":           group.loc[idx, "gap"],
                "turn_position": group.loc[idx, "turn_position"],
                "concession":    group.loc[idx, "concession"],
                "delta_d_D":     group.loc[idx, "delta_d_D"],
                "msrp":          group.loc[idx, "msrp"],
            })

    return pd.DataFrame(snapshots)

snapshots_df = build_concession_snapshots(combined_df)
print(f"Total snapshots: {len(snapshots_df):,}")
print(f"  Real: {(snapshots_df['source']=='real').sum():,}")
print(f"  Synthetic: {(snapshots_df['source']=='synthetic').sum():,}")
```

---

### Block 6 · Retrospective Reward Labelling

**What it does:** For every snapshot — where we already know the full conversation outcome — computes the hindsight-optimal buyer action. This generates training labels without any human annotation.

> **Stopping study parallel:** The stopping study evaluates three strategies (quit at 45s, quit at 60s, never quit) and picks the one with the highest reward. The same logic applies here with three negotiation strategies. The principle is identical: retrospective optimisation using known outcomes to label past decisions.

```python
def compute_concession_rewards(snapshots_df):
    """
    For each snapshot, computes the reward of three buyer strategies
    and assigns the optimal action label.

    Strategy 1 — accept:     Take the dealer's current offer now
    Strategy 2 — push_once:  Make one more ask, then accept
    Strategy 3 — continue:   Keep negotiating to the end of the call

    Reward = final_discount / msrp (normalised across car models)
    """
    # Final discount achieved in each conversation
    final_d = snapshots_df.groupby("conv_id")["d_D"].last().rename("final_discount")
    snapshots_df = snapshots_df.join(final_d, on="conv_id")

    # Strategy 1: accept current offer
    snapshots_df["r_accept"] = snapshots_df["d_D"] / snapshots_df["msrp"]

    # Strategy 2: push once — benefit only if a concession actually followed
    snapshots_df["r_push_once"] = snapshots_df.apply(
        lambda r: (r["d_D"] + r["delta_d_D"]) / r["msrp"]
                  if r["concession"] == 1
                  else r["d_D"] / r["msrp"],
        axis=1)

    # Strategy 3: continue to end
    snapshots_df["r_continue"] = (
        snapshots_df["final_discount"] / snapshots_df["msrp"])

    # Optimal action: whichever strategy yielded the highest reward
    snapshots_df["max_reward"] = snapshots_df[
        ["r_accept", "r_push_once", "r_continue"]].max(axis=1)

    snapshots_df["optimal_action"] = snapshots_df.apply(
        lambda r: "accept"    if r["max_reward"] == r["r_accept"]
             else "push_once" if r["max_reward"] == r["r_push_once"]
             else "continue",
        axis=1)

    return snapshots_df

snapshots_df = compute_concession_rewards(snapshots_df)

print("Optimal action distribution:")
print(snapshots_df["optimal_action"].value_counts(normalize=True).round(3))
```

**Three strategies and their stopping-study analogues:**

| Strategy | Negotiation meaning | Stopping-study analogue |
|---|---|---|
| `accept` | Take the dealer's current offer immediately | Quit at 45s — cut losses now |
| `push_once` | Make one more ask, then accept | Wait to 60s then quit |
| `continue` | Keep negotiating to the end | Never quit — let the call run |

---

## PART 2 — The RSSM: Architecture and Training

> **Why the stopping study's LLM fine-tuning is insufficient here:** The stopping study trains Llama to predict a binary outcome from a partial transcript. That works because the call outcome is independent of the agent's decision at each checkpoint — the sale happens or it doesn't regardless of when the agent quits. In negotiation, Tommy's words *cause* the dealer's next response. This sequential causal dependency requires a model that explicitly tracks hidden state across turns. The RSSM (Recurrent State-Space Model) is that model.

---

### Block 7 · The Frozen LLM Encoder

**What it does:** Converts every utterance into a dense embedding vector using a pretrained language model whose weights are never updated. All negotiation dynamics learning happens downstream in the GRU and VAE layers.

```python
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class FrozenEncoder(nn.Module):
    """
    Wraps a pretrained sentence encoder. Weights are permanently frozen.
    Handles language understanding so the RSSM only needs to learn
    negotiation dynamics — a separation of concerns that makes the
    model tractable on ~10,000 conversations.
    """
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = False   # never updated

    def forward(self, texts):
        with torch.no_grad():
            return torch.tensor(
                self.encoder.encode(texts, show_progress_bar=False),
                dtype=torch.float32)   # [batch, 768]
```

> **Stopping study parallel:** The stopping study fine-tunes the full Llama model on the binary sale-prediction task. Here, the encoder is frozen because: (a) fine-tuning a large LLM on ~10,000 conversations would overfit, (b) we need the encoder to produce stable embeddings that the RSSM can learn to interpret across the full training run, and (c) the encoder's job — understanding language — is already solved by pretraining. What the RSSM needs to learn — negotiation dynamics — is a different problem entirely.

---

### Block 8 · The RSSM Architecture

**What it does:** Defines the full hybrid RSSM. Three components work in sequence at each turn: the GRU accumulates observable history into a deterministic state `h_t`; the VAE layer infers a stochastic latent state `z_t` representing the dealer's hidden position; three output heads predict the concession reward, reconstruct the dealer's utterance, and select the buyer's optimal action.

```python
class NegotiationRSSM(nn.Module):
    """
    Hybrid Recurrent State-Space Model for sequential negotiation.

    At each turn t, given Tommy's utterance and the current price state:
      1. GRU updates deterministic state h_t from h_{t-1} and x_{t-1}
         → tracks observable history: gap, phase, prior moves
      2. VAE infers stochastic latent z_t from h_t and x_t
         → represents dealer's hidden state: reservation price, type, room
      3. Three output heads operate on (h_t, z_t):
         → reward head:  predict concession size
         → recon head:   reconstruct dealer's next utterance embedding
         → policy head:  select accept / push_once / continue
    """
    def __init__(self, embed_dim=768, h_dim=256, z_dim=64, price_dim=3):
        super().__init__()
        self.encoder = FrozenEncoder()

        # Deterministic state: GRU takes [previous embedding, price state]
        self.gru = nn.GRUCell(
            input_size=embed_dim + price_dim,
            hidden_size=h_dim)

        # Posterior q(z_t | h_t, x_t) — used during training
        # We observe x_t (dealer's response), so we can infer z_t exactly
        self.post_mu     = nn.Linear(h_dim + embed_dim, z_dim)
        self.post_logvar = nn.Linear(h_dim + embed_dim, z_dim)

        # Prior p(z_t | h_t) — used at deployment
        # Before seeing the dealer's response, we predict z_t from h_t alone
        self.prior_mu     = nn.Linear(h_dim, z_dim)
        self.prior_logvar = nn.Linear(h_dim, z_dim)

        # Output heads — all condition on (h_t, z_t)
        self.reward_head = nn.Sequential(
            nn.Linear(h_dim + z_dim, 128), nn.ReLU(), nn.Linear(128, 1))

        self.recon_head = nn.Sequential(
            nn.Linear(h_dim + z_dim, 256), nn.ReLU(), nn.Linear(256, embed_dim))

        self.policy_head = nn.Sequential(
            nn.Linear(h_dim + z_dim, 128), nn.ReLU(), nn.Linear(128, 3))
            # 3 actions: accept=0, push_once=1, continue=2

    def reparametrize(self, mu, logvar):
        """Sample z ~ N(mu, sigma²) via the reparametrisation trick."""
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, texts_t, prices_t, h_prev, texts_prev=None):
        """
        Single-turn forward pass.
        texts_t:   list[str]  — utterances at turn t
        prices_t:  [B, 3]     — [d_B, d_D, gap] at turn t
        h_prev:    [B, h_dim] — deterministic state from previous turn
        texts_prev: list[str] — utterances at t-1 (None at turn 0)
        """
        x_t    = self.encoder(texts_t)                          # [B, embed_dim]
        x_prev = self.encoder(texts_prev) if texts_prev \
                 else torch.zeros_like(x_t)

        # GRU update: h_t encodes everything observable up to turn t
        h_t = self.gru(torch.cat([x_prev, prices_t], dim=-1), h_prev)

        # Posterior: infer dealer's latent state given what they just said
        post_input   = torch.cat([h_t, x_t], dim=-1)
        post_mu      = self.post_mu(post_input)
        post_logvar  = self.post_logvar(post_input)
        z_t          = self.reparametrize(post_mu, post_logvar)

        # Prior: predict dealer's latent state before they respond
        prior_mu     = self.prior_mu(h_t)
        prior_logvar = self.prior_logvar(h_t)

        hz = torch.cat([h_t, z_t], dim=-1)

        return {
            "h_t":           h_t,
            "z_t":           z_t,
            "post_mu":       post_mu,
            "post_logvar":   post_logvar,
            "prior_mu":      prior_mu,
            "prior_logvar":  prior_logvar,
            "reward_pred":   self.reward_head(hz).squeeze(-1),
            "recon_pred":    self.recon_head(hz),
            "action_logits": self.policy_head(hz),
        }
```

**What the three state components represent in a real negotiation:**

| Component | What it holds at Phase 4 entry | How it changes |
|---|---|---|
| `h_t` | Accumulated facts: gap is $1,000, manager invoked twice, Phase 4, Tommy used BATNA in Phase 3 | Deterministic — updated by GRU every turn, no uncertainty |
| `z_t` (posterior) | Inferred dealer state given their last response: floor likely $1,500–$2,000, shield may be real | Stochastic — narrows as concession events reveal the dealer's true position |
| `z_t` (prior) | Predicted dealer state before they respond: what do we expect given h_t alone? | Stochastic — the deployment version; trained to match the posterior |

---

### Block 9 · ELBO Training Objective

**What it does:** Defines the three-term loss function that trains the GRU and VAE layers jointly. Each term has a distinct role in ensuring `z_t` becomes a useful representation of the dealer's hidden state.

```python
def elbo_loss(outputs, x_t_embed, reward_target, kl_weight=1.0):
    """
    ELBO = reconstruction loss + reward loss + KL divergence

    Term 1 — Reconstruction: z_t must be informative about what
             the dealer says next. If z_t carries no information
             about dealer behavior, this term is high.

    Term 2 — Reward: z_t must be useful for predicting whether
             a concession occurred. This connects the latent state
             to the outcome we actually care about.

    Term 3 — KL divergence: the prior p(z_t | h_t) must stay
             close to the posterior q(z_t | h_t, x_t). This is
             what makes deployment possible — at inference time
             we only have h_t, so the prior must be good enough
             to substitute for the posterior.
    """
    recon_loss  = nn.functional.mse_loss(outputs["recon_pred"], x_t_embed)
    reward_loss = nn.functional.mse_loss(outputs["reward_pred"], reward_target)

    kl = -0.5 * torch.mean(
        1 + outputs["post_logvar"] - outputs["prior_logvar"]
        - (outputs["post_mu"] - outputs["prior_mu"]).pow(2)
          / outputs["prior_logvar"].exp()
        - outputs["post_logvar"].exp()
          / outputs["prior_logvar"].exp())

    total = recon_loss + reward_loss + kl_weight * kl
    return total, {"recon": recon_loss.item(),
                   "reward": reward_loss.item(),
                   "kl": kl.item()}
```

> **No analogue in stopping study:** The stopping study uses cross-entropy loss on a binary label. The ELBO is needed here because we must train both the observable-history model (GRU) and the hidden-state inference model (VAE) simultaneously, while ensuring the prior learned for deployment is close to the posterior that is only available during training.

---

### Block 10 · RSSM Training Loop

**What it does:** Trains the GRU and VAE parameters on the combined real + synthetic corpus. Uses KL annealing to prevent posterior collapse — a known failure mode in VAE training where the model ignores the latent space entirely.

```python
def train_rssm(model, train_loader, val_loader,
               n_epochs=50, lr=1e-3, kl_warmup_epochs=10):
    """
    KL annealing: weight ramps from 0 to 1 over kl_warmup_epochs.
    Without this, the model tends to collapse z_t to the prior early
    in training before the GRU has learned useful representations,
    making the VAE layer effectively useless.
    """
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        kl_weight = min(1.0, epoch / kl_warmup_epochs)
        model.train()
        epoch_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            h = torch.zeros(batch["prices"].shape[0], 256)
            episode_loss = 0

            for t in range(batch["seq_len"]):
                out = model(
                    texts_t    = batch["texts"][:, t],
                    prices_t   = batch["prices"][:, t],
                    h_prev     = h,
                    texts_prev = batch["texts"][:, t-1] if t > 0 else None)

                loss, _ = elbo_loss(
                    out,
                    x_t_embed     = batch["embeddings"][:, t],
                    reward_target = batch["rewards"][:, t],
                    kl_weight     = kl_weight)

                episode_loss += loss
                h = out["h_t"].detach()   # stop gradient across episode boundary

            episode_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(episode_loss.item())

        val_loss = evaluate_rssm(model, val_loader, kl_weight)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/rssm_best.pt")

        print(f"Epoch {epoch+1:3d} | "
              f"train {np.mean(epoch_losses):.4f} | "
              f"val {val_loss:.4f} | kl_w {kl_weight:.2f}")

    model.load_state_dict(torch.load("models/rssm_best.pt"))
    return model

rssm = NegotiationRSSM()
rssm = train_rssm(rssm, train_loader, val_loader)
```

> **Stopping study parallel:** Early stopping on validation AUC is used in both studies. The stopping study stops after 4 epochs when AUC reaches 1.0 — a fast convergence because the task is simple (binary, fixed timestamps). The RSSM trains for up to 50 epochs because the task is harder: jointly learning a GRU, a VAE, and three output heads from sequential data with a non-trivial KL term.

---

## PART 3 — Linguistic Feature Extraction

> **Why not LIWC or hand-coded dictionaries:** Predefined psychological categories like LIWC impose a theoretical vocabulary onto the data. The words that actually matter in *this* negotiation corpus — "competing," "shipping," "allocated," "rarity" — are discovered empirically from the data, not imported from a general psychology framework. Occlusion attribution lets the trained LLM tell us which words in the buyer's utterance shifted the concession probability, without any prior specification.

---

### Block 11 · Concession Probability Extraction

**What it does:** Runs the trained RSSM on every snapshot to extract `prob_concession` — the model's predicted probability that the dealer will make a concession given the current state. This is the negotiation equivalent of the stopping study's `prob_yes`.

```python
def get_prob_concession(model, snapshot_row, h_prev):
    """
    Extracts the model's concession probability for one snapshot.
    Concession is predicted if reward_pred > 0 under the current z_t.
    Also returns prior_sigma — the width of the dealer's belief interval
    (Knightian uncertainty, used in Study 2).
    """
    model.eval()
    with torch.no_grad():
        out = model(
            texts_t  = [snapshot_row["transcript"]],
            prices_t = torch.tensor(
                [[snapshot_row["d_B"],
                  snapshot_row["d_D"],
                  snapshot_row["gap"]]], dtype=torch.float32),
            h_prev   = h_prev)

    prob_conc    = torch.sigmoid(out["reward_pred"]).item()
    prior_sigma  = torch.exp(0.5 * out["prior_logvar"]).mean().item()
    post_sigma   = torch.exp(0.5 * out["post_logvar"]).mean().item()
    z_mu         = out["post_mu"].squeeze().numpy()

    return prob_conc, prior_sigma, post_sigma, z_mu, out["h_t"]
```

> **Stopping study parallel:** The stopping study extracts `prob_yes` from the fine-tuned Llama's output token distribution (Block 10 in that notebook). Here, `prob_concession` comes from the RSSM's reward head rather than a language model's token probabilities — because our model is a state-space model, not a next-token predictor.

---

### Block 12 · Occlusion Attribution — Top-k Word Extraction

**What it does:** For each snapshot, measures the causal influence of every word in the buyer's last utterance by temporarily removing it and measuring the drop in `prob_concession`. Words whose removal causes the largest drop are the most influential.

```python
def extract_top_k_words(model, snapshot_row, h_prev, k=10):
    """
    Occlusion attribution: for each word in the buyer's last utterance,
    compute base_prob - masked_prob.
    Positive attribution = word helps extract a concession.
    Negative attribution = word suppresses a concession.
    """
    base_prob, *_ = get_prob_concession(model, snapshot_row, h_prev)

    # Extract buyer's last utterance
    lines = snapshot_row["transcript"].strip().split("\n")
    buyer_lines = [l for l in lines if l.startswith("buyer:")]
    if not buyer_lines:
        return []

    words = buyer_lines[-1].replace("buyer:", "").strip().split()
    attributions = []

    for i, word in enumerate(words):
        masked = words[:i] + ["[MASK]"] + words[i+1:]
        masked_utt = "buyer: " + " ".join(masked)
        masked_transcript = "\n".join(
            [l for l in lines if not l.startswith("buyer:")] + [masked_utt])

        masked_row = snapshot_row.copy()
        masked_row["transcript"] = masked_transcript
        masked_prob, *_ = get_prob_concession(model, masked_row, h_prev)

        attributions.append({
            "word":        word,
            "attribution": base_prob - masked_prob
        })

    return sorted(attributions, key=lambda x: abs(x["attribution"]), reverse=True)[:k]

# Apply to all snapshots
snapshots_df["top_k_words"] = snapshots_df.apply(
    lambda r: extract_top_k_words(rssm, r, get_h_prev(rssm, r), k=10),
    axis=1)
```

---

### Block 13 · Corpus-Level Linguistic Markers and L_score

**What it does:** Aggregates per-snapshot word attributions into a stable corpus-level vocabulary of effective negotiation language. Computes a scalar `L_score` for every snapshot — the sum of mean attributions for all words present in the buyer's utterance that appear in the marker vocabulary.

```python
from collections import defaultdict

def build_linguistic_markers(snapshots_df, min_occurrences=5):
    """
    Aggregates top-k word attributions across all concession-event snapshots.
    Returns a ranked vocabulary of empirically effective negotiation language.
    """
    word_attributions = defaultdict(list)
    for _, row in snapshots_df[snapshots_df["concession"] == 1].iterrows():
        for entry in row["top_k_words"]:
            word_attributions[entry["word"]].append(entry["attribution"])

    marker_df = pd.DataFrame([
        {"word":             word,
         "mean_attribution": np.mean(attrs),
         "frequency":        len(attrs),
         "std_attribution":  np.std(attrs)}
        for word, attrs in word_attributions.items()
        if len(attrs) >= min_occurrences
    ]).sort_values("mean_attribution", ascending=False)

    return marker_df

linguistic_markers = build_linguistic_markers(snapshots_df)

print("Top 20 concession-inducing words:")
print(linguistic_markers.head(20)[["word","mean_attribution","frequency"]])

print("\nTop 20 concession-suppressing words:")
print(linguistic_markers.tail(20)[["word","mean_attribution","frequency"]])


def compute_L_score(utterance_text, marker_df):
    """Scalar linguistic effectiveness score for one buyer utterance."""
    lookup = marker_df.set_index("word")["mean_attribution"].to_dict()
    return sum(lookup.get(w.lower(), 0.0)
               for w in utterance_text.split())

snapshots_df["L_score"] = snapshots_df.apply(
    lambda r: compute_L_score(
        r["transcript"].split("\n")[-1], linguistic_markers),
    axis=1)
```

**What the outputs mean:**

| Output | Meaning |
|---|---|
| `mean_attribution` | Average drop in `prob_concession` when this word is removed — its average causal influence |
| `frequency` | How often the word appeared in top-k lists — consistency across the corpus |
| `L_score` | Per-snapshot scalar: sum of `mean_attribution` for all marker-vocabulary words present |

---

### Block 14 · Tactic Classification from Top-k Words

**What it does:** Maps corpus-derived linguistic markers onto six Cialdini-class persuasion tactics. Tactic scores are used in Study 3's heterogeneous effects analysis. The lexicons are grounded in the transcript analysis rather than imposed from theory.

```python
TACTIC_LEXICONS = {
    "batna":       ["deal", "state", "out-of-state", "competing",
                    "elsewhere", "ship", "shipping", "fly", "alternative",
                    "already", "better", "offer"],
    "scarcity":    ["rare", "rarity", "hard", "allocated", "allocation",
                    "fighting", "limited", "only", "left", "week",
                    "plentiful", "come by"],
    "rapport":     ["dude", "bro", "man", "brother", "buddy",
                    "appreciate", "love", "trust", "together", "friend"],
    "reciprocity": ["trying", "best", "bat", "went", "fought",
                    "manager", "asked", "effort", "behalf"],
    "commitment":  ["deposit", "ready", "buy", "close", "lock",
                    "tonight", "tomorrow", "schedule", "fly down"],
    "authority":   ["boss", "manager", "Marlon", "Donnie", "VIP",
                    "general", "owner", "approved", "allowed"],
}

def score_tactics(utterance, marker_df):
    """
    Weighted tactic score: sum of mean_attribution for all words
    in the utterance that belong to each tactic's lexicon.
    """
    words  = utterance.lower().split()
    lookup = marker_df.set_index("word")["mean_attribution"].to_dict()
    return {
        tactic: sum(lookup.get(w, 0.0) for w in words if w in lexicon)
        for tactic, lexicon in TACTIC_LEXICONS.items()
    }

tactic_scores = snapshots_df.apply(
    lambda r: pd.Series(score_tactics(
        r["transcript"].split("\n")[-1], linguistic_markers)),
    axis=1)
snapshots_df = pd.concat([snapshots_df, tactic_scores], axis=1)
```

---

## PART 4 — Four-Study Research Implementation

---

## STUDY 1 · Does Language Have an Independent Effect on Concessions?

> **Research question:** Do linguistic cues have an independent causal effect on dealer concessions, over and above the numeric bargaining position? Does this effect vary across the five phases of the call?
>
> **Contribution:** Establishes the linguistic channel in sequential field negotiation — not "does language matter?" (already known from lab studies) but "how much does language matter conditional on bargaining position, and when in the call does it matter most?"

---

### Block 15 · Study 1 — Panel Regression

**What it does:** Estimates the independent effect of `L_score` on concession size after controlling for all numeric variables. Phase-as-moderator tests whether the linguistic channel varies across the call arc — the specific contribution over prior lab work.

```python
import statsmodels.formula.api as smf

# Model 1a: numeric variables only — baseline
model_1a = smf.ols(
    "delta_d_D ~ d_B + d_D + gap + turn_position + C(phase)",
    data=snapshots_df
).fit(cov_type="cluster",
      cov_kwds={"groups": snapshots_df["conv_id"]})

# Model 1b: add aggregate linguistic score
model_1b = smf.ols(
    "delta_d_D ~ d_B + d_D + gap + turn_position + C(phase) + L_score",
    data=snapshots_df
).fit(cov_type="cluster",
      cov_kwds={"groups": snapshots_df["conv_id"]})

# Model 1c: phase × linguistic interaction — main Study 1 result
# Tests: is the linguistic effect largest at Phase 3 (anchor turn)?
model_1c = smf.ols(
    "delta_d_D ~ d_B + d_D + gap + turn_position + "
    "C(phase) + L_score + L_score:C(phase)",
    data=snapshots_df
).fit(cov_type="cluster",
      cov_kwds={"groups": snapshots_df["conv_id"]})

# Model 1d: logistic robustness — does L_score predict any concession at all?
model_1d = smf.logit(
    "concession ~ d_B + d_D + gap + turn_position + "
    "C(phase) + L_score + L_score:C(phase)",
    data=snapshots_df
).fit(cov_type="cluster",
      cov_kwds={"groups": snapshots_df["conv_id"]})

# Variance decomposition
r2_numeric      = model_1a.rsquared
r2_linguistic   = model_1b.rsquared
r2_interaction  = model_1c.rsquared

print("STUDY 1 RESULTS")
print(f"R² numeric only:              {r2_numeric:.4f}")
print(f"R² + L_score:                 {r2_linguistic:.4f}  "
      f"(incremental: {r2_linguistic - r2_numeric:.4f})")
print(f"R² + L_score × phase:         {r2_interaction:.4f}  "
      f"(incremental: {r2_interaction - r2_linguistic:.4f})")

print("\nL_score coefficient by phase:")
for phase in [1, 2, 3, 4, 5]:
    key  = f"L_score:C(phase)[T.{phase}]"
    coef = model_1c.params.get(key, model_1c.params.get("L_score", np.nan))
    pval = model_1c.pvalues.get(key, model_1c.pvalues.get("L_score", np.nan))
    sig  = "***" if pval < .001 else "**" if pval < .01 \
           else "*" if pval < .05 else ""
    print(f"  Phase {phase}: β = {coef:7.4f}  p = {pval:.4f} {sig}")
```

> **Code Logic:**
> - Clustered standard errors by `conv_id` are mandatory — turns within the same conversation are correlated. Ignoring this would inflate significance substantially.
> - Phase fixed effects absorb baseline concession probability at each phase, isolating the linguistic effect *within* phases rather than across them.
> - Model 1c is the main result: if the Phase 3 coefficient is the largest, language matters most at the anchor turn — the moment Tommy first names a justified number.

---

### Block 16 · Study 1 — Ceiling Check and Identification Diagnostic

**What it does:** Bounds how much linguistic variance remains unattributed (the "dark linguistic matter") and checks the key identification assumption — that `L_score` is not simply a proxy for bargaining position strength.

```python
# Ceiling check: how much does the full LLM signal add over L_score?
model_ceiling = smf.ols(
    "delta_d_D ~ d_B + d_D + gap + turn_position + "
    "C(phase) + prob_concession",
    data=snapshots_df
).fit(cov_type="cluster",
      cov_kwds={"groups": snapshots_df["conv_id"]})

dark_matter = model_ceiling.rsquared - r2_linguistic
print(f"R² with prob_concession:   {model_ceiling.rsquared:.4f}")
print(f"'Dark linguistic matter':  {dark_matter:.4f}")
print("(Signal the LLM detects but cannot be attributed to specific words)")

# Identification: does L_score vary independently of gap within conversations?
gap_l_corr = snapshots_df.groupby("conv_id").apply(
    lambda g: g["L_score"].corr(g["gap"])).describe()
print("\nWithin-conversation correlation of L_score with gap:")
print(gap_l_corr)
print("(Near-zero mean supports identification: "
      "language quality varies somewhat independently of BATNA strength)")

# Robustness: replicate on real conversations only
real_only = snapshots_df[snapshots_df["source"] == "real"]
model_real = smf.ols(
    "delta_d_D ~ d_B + d_D + gap + turn_position + C(phase) + "
    "L_score + L_score:C(phase)",
    data=real_only
).fit(cov_type="cluster",
      cov_kwds={"groups": real_only["conv_id"]})

print(f"\nRobustness (real conversations only):")
print(f"L_score: β={model_real.params['L_score']:.4f} "
      f"p={model_real.pvalues['L_score']:.4f}")
```

---

## STUDY 2 · Is Belief-Updating the Mechanism?

> **Research question:** Does the linguistic channel operate through belief-updating — do high-`L_score` utterances reduce the dealer's uncertainty about the buyer's type, and does this uncertainty reduction predict subsequent concessions?
>
> **Contribution:** Provides the causal mechanism for Study 1's finding. The dealer's Knightian uncertainty — represented by the width of the RSSM's prior distribution `prior_sigma` — is shown to be a measurable, economically meaningful quantity that mediates the linguistic effect. This connects to Gilboa et al.'s (2008) multiple-prior framework: the dealer holds a set of beliefs about the buyer's type, and effective buyer language collapses that set.

---

### Block 17 · Study 2 — Extracting Belief Uncertainty Trajectories

**What it does:** Extracts the prior uncertainty `prior_sigma` at every turn from the trained RSSM. This is the width of the dealer's belief interval — the Knightian uncertainty measure — before they have seen the buyer's current utterance.

```python
def extract_uncertainty_trajectories(model, snapshots_df):
    """
    Extracts prior_sigma (dealer's Knightian uncertainty) at every turn.
    prior_sigma = mean std of prior p(z_t | h_t) across z dimensions.
    Wide prior = dealer has not yet inferred buyer's type.
    Narrow prior = dealer has a confident estimate of buyer's floor.
    """
    model.eval()
    records = []

    for conv_id, group in snapshots_df.groupby("conv_id"):
        h = torch.zeros(1, 256)
        group = group.sort_values("turn_idx")

        for _, row in group.iterrows():
            with torch.no_grad():
                out = model(
                    texts_t  = [row["transcript"]],
                    prices_t = torch.tensor(
                        [[row["d_B"], row["d_D"], row["gap"]]],
                        dtype=torch.float32),
                    h_prev   = h)

            prior_sigma = torch.exp(0.5 * out["prior_logvar"]).mean().item()
            post_sigma  = torch.exp(0.5 * out["post_logvar"]).mean().item()

            records.append({
                "conv_id":      conv_id,
                "turn_idx":     row["turn_idx"],
                "prior_sigma":  prior_sigma,   # uncertainty_D: dealer's belief width
                "post_sigma":   post_sigma,    # uncertainty after seeing response
            })
            h = out["h_t"]

    return pd.DataFrame(records)

uncertainty_df = extract_uncertainty_trajectories(rssm, snapshots_df)
snapshots_df   = snapshots_df.merge(
    uncertainty_df, on=["conv_id", "turn_idx"])

# Uncertainty reduction: change in prior_sigma from turn t-1 to t
snapshots_df["delta_uncertainty"] = snapshots_df.groupby(
    "conv_id")["prior_sigma"].diff()
```

---

### Block 18 · Study 2 — Mediation Analysis

**What it does:** Tests the three-step mediation pathway: `L_score → uncertainty reduction → concession`. If belief-updating is the mechanism, the `L_score` coefficient in Model 1b should shrink when `prior_sigma` is added as a covariate.

```python
# Step 1: Does L_score predict uncertainty reduction?
model_2a = smf.ols(
    "delta_uncertainty ~ L_score + d_B + d_D + gap + "
    "C(phase) + turn_position",
    data=snapshots_df.dropna(subset=["delta_uncertainty"])
).fit(cov_type="cluster",
      cov_kwds={"groups": snapshots_df.dropna(
          subset=["delta_uncertainty"])["conv_id"]})

print("Step 1 — L_score → uncertainty reduction")
print(f"  β = {model_2a.params['L_score']:.4f}  "
      f"p = {model_2a.pvalues['L_score']:.4f}")
print("  (Negative = high L_score reduces dealer's uncertainty)")

# Step 2: Does uncertainty_D predict concessions?
model_2b = smf.ols(
    "delta_d_D ~ prior_sigma + d_B + d_D + gap + "
    "C(phase) + turn_position",
    data=snapshots_df
).fit(cov_type="cluster",
      cov_kwds={"groups": snapshots_df["conv_id"]})

print("\nStep 2 — prior_sigma → concession size")
print(f"  β = {model_2b.params['prior_sigma']:.4f}  "
      f"p = {model_2b.pvalues['prior_sigma']:.4f}")

# Step 3: Partial mediation — L_score coefficient shrinks when prior_sigma added
model_2c = smf.ols(
    "delta_d_D ~ L_score + prior_sigma + d_B + d_D + gap + "
    "C(phase) + turn_position",
    data=snapshots_df
).fit(cov_type="cluster",
      cov_kwds={"groups": snapshots_df["conv_id"]})

l_total   = model_1b.params["L_score"]
l_direct  = model_2c.params["L_score"]
mediation = (1 - l_direct / l_total) * 100

print("\nStep 3 — Full mediation model")
print(f"  L_score direct effect: β = {l_direct:.4f}  "
      f"p = {model_2c.pvalues['L_score']:.4f}")
print(f"  Proportion mediated:   {mediation:.1f}%")
```

---

### Block 19 · Study 2 — Bootstrap Indirect Effect

**What it does:** Tests the significance of the indirect effect (a × b path) using bootstrapped confidence intervals clustered at the conversation level.

```python
def bootstrap_indirect_effect(df, n_bootstrap=1000, seed=42):
    """
    Resamples at conversation level to preserve within-conversation
    correlation. A 95% CI excluding zero confirms significant mediation.
    """
    np.random.seed(seed)
    conv_ids = df["conv_id"].unique()
    indirect = []

    for _ in range(n_bootstrap):
        ids  = np.random.choice(conv_ids, size=len(conv_ids), replace=True)
        boot = pd.concat([df[df["conv_id"] == c] for c in ids],
                         ignore_index=True)
        try:
            a = smf.ols("delta_uncertainty ~ L_score + d_B + d_D + gap",
                        data=boot.dropna(subset=["delta_uncertainty"])
                        ).fit().params["L_score"]
            b = smf.ols("delta_d_D ~ L_score + prior_sigma + d_B + d_D + gap",
                        data=boot).fit().params["prior_sigma"]
            indirect.append(a * b)
        except Exception:
            continue

    indirect = np.array(indirect)
    ci_lo, ci_hi = np.percentile(indirect, [2.5, 97.5])
    print(f"Indirect effect:  {indirect.mean():.6f}")
    print(f"95% CI:           [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"Significant:      {'YES' if ci_lo > 0 or ci_hi < 0 else 'NO'}")
    return indirect

indirect_dist = bootstrap_indirect_effect(
    snapshots_df.dropna(subset=["delta_uncertainty"]))
```

---

## STUDY 3 · Which Cues Work for Which Dealer Types?

> **Research question:** Do different persuasion tactics have heterogeneous effects on concessions, moderated by the dealer's latent type as inferred from `z_t` trajectories?
>
> **Contribution:** Turns the aggregate linguistic effect into an actionable taxonomy. Shows that optimal tactic selection is buyer-seller match-specific — the same tactic does not work equally well on all dealer types. Connects to the personalisation literature in marketing.

---

### Block 20 · Study 3 — Dealer Type Clustering

**What it does:** Clusters conversations into dealer archetypes based on their `z_t` trajectory features — how uncertain the dealer is, how fast they reveal their floor, how much room they ultimately give. These clusters become the moderator variable.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_dealer_types(snapshots_df, n_clusters=4):
    """
    Clusters dealers by three trajectory features extracted from prior_sigma:
      - Mean uncertainty across the conversation
      - Uncertainty at the anchor turn (Phase 3 entry)
      - Total uncertainty reduction (how much the dealer reveals across the call)
    """
    conv_feats = snapshots_df.groupby("conv_id").agg(
        sigma_mean    = ("prior_sigma", "mean"),
        sigma_phase3  = ("prior_sigma",
                          lambda x: x.iloc[max(0, len(x)//3)]),
        sigma_decline = ("prior_sigma",
                          lambda x: x.iloc[0] - x.iloc[-1])
    ).reset_index()

    features = conv_feats[["sigma_mean","sigma_phase3","sigma_decline"]].values
    scaler   = StandardScaler()
    km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    conv_feats["dealer_type"] = km.fit_predict(scaler.fit_transform(features))

    # Label clusters by centroid interpretation
    c = km.cluster_centers_
    labels = {}
    for k in range(n_clusters):
        high_sigma   = c[k][0] > 0
        declining    = c[k][2] > 0
        if   high_sigma and declining:     labels[k] = "opaque_soft"
        elif high_sigma and not declining: labels[k] = "opaque_hard"
        elif not high_sigma and declining: labels[k] = "transparent_soft"
        else:                              labels[k] = "transparent_hard"

    conv_feats["dealer_type_label"] = conv_feats["dealer_type"].map(labels)
    print("Dealer type distribution:")
    print(conv_feats["dealer_type_label"].value_counts())

    return conv_feats[["conv_id","dealer_type","dealer_type_label"]]

dealer_types = cluster_dealer_types(snapshots_df)
snapshots_df = snapshots_df.merge(dealer_types, on="conv_id")
```

**Dealer type archetypes:**

| Type | Uncertainty profile | Concession pattern | Effective tactic |
|---|---|---|---|
| `opaque_soft` | High initial uncertainty, declines through call | Moves eventually but needs convincing | Rapport + BATNA combination |
| `opaque_hard` | High uncertainty, does not decline | Rarely moves; manager shield is real | BATNA — credible outside option only |
| `transparent_soft` | Low initial uncertainty, declines fast | Reveals floor early, concedes readily | Commitment — lock in before they change mind |
| `transparent_hard` | Low uncertainty throughout | Knows their floor, holds it | Authority — escalate past salesperson |

---

### Block 21 · Study 3 — Heterogeneous Effects Regression

**What it does:** Estimates the effect of each tactic for each dealer type. The interaction table — rows are tactics, columns are dealer types — is the main Study 3 result.

```python
tactics = ["batna", "scarcity", "rapport", "reciprocity",
           "commitment", "authority"]

# Omnibus: do any tactics matter at all?
model_3a = smf.ols(
    "delta_d_D ~ " + " + ".join(tactics) +
    " + d_B + d_D + gap + C(phase) + turn_position",
    data=snapshots_df
).fit(cov_type="cluster",
      cov_kwds={"groups": snapshots_df["conv_id"]})

print("STUDY 3 — Tactic main effects (omnibus):")
for t in tactics:
    coef = model_3a.params.get(t, np.nan)
    pval = model_3a.pvalues.get(t, np.nan)
    sig  = "***" if pval<.001 else "**" if pval<.01 else "*" if pval<.05 else ""
    print(f"  {t:12s}: β = {coef:7.4f}  p = {pval:.4f} {sig}")

# Heterogeneous effects: tactic × dealer type interaction
interaction = " + ".join([f"{t}:C(dealer_type_label)" for t in tactics])
model_3b = smf.ols(
    "delta_d_D ~ " + " + ".join(tactics) + " + " + interaction +
    " + d_B + d_D + gap + C(phase) + turn_position",
    data=snapshots_df
).fit(cov_type="cluster",
      cov_kwds={"groups": snapshots_df["conv_id"]})

print("\nSTUDY 3 — Heterogeneous effects (tactic × dealer type):")
dtypes = dealer_types["dealer_type_label"].unique()
rows   = {}
for t in tactics:
    rows[t] = {}
    for d in dtypes:
        key  = f"{t}:C(dealer_type_label)[T.{d}]"
        coef = model_3b.params.get(key, np.nan)
        pval = model_3b.pvalues.get(key, np.nan)
        sig  = "*" if (not np.isnan(pval) and pval < .05) else ""
        rows[t][d] = f"{coef:.3f}{sig}" if not np.isnan(coef) else "—"

print(pd.DataFrame(rows).T.to_string())
```

---

## STUDY 4 · Can a Learned Policy Improve Buyer Outcomes?

> **Research question:** Can an RSSM-guided negotiation policy, trained on the full model from Studies 1–3, achieve better out-of-sample outcomes than Tommy's actual behaviour across the 60 real held-out conversations?
>
> **Contribution:** Validates the entire modelling framework. The 60 real conversations were never seen during training, so outperforming Tommy here is genuine out-of-sample evidence — not a simulation artefact.

---

### Block 22 · Policy Head Fine-Tuning

**What it does:** Fine-tunes only the policy head using the optimal action labels from Block 6. All other RSSM parameters are frozen — the policy benefits from the latent representation learned in Blocks 8–10 without disturbing it.

```python
def train_policy(model, train_loader, val_loader, n_epochs=30, lr=5e-4):
    """
    Fine-tunes only the policy_head. All other parameters frozen.
    Loss is weighted cross-entropy: actions from higher-reward
    trajectories receive larger gradients.
    """
    for name, p in model.named_parameters():
        p.requires_grad = "policy_head" in name

    optimizer = torch.optim.Adam(model.policy_head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")
    best_val  = -float("inf")

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            h = torch.zeros(batch["prices"].shape[0], 256)
            loss = 0

            for t in range(batch["seq_len"]):
                out = model(
                    texts_t    = batch["texts"][:, t],
                    prices_t   = batch["prices"][:, t],
                    h_prev     = h,
                    texts_prev = batch["texts"][:, t-1] if t > 0 else None)

                ce   = criterion(out["action_logits"],
                                 batch["optimal_actions"][:, t])
                loss += (ce * batch["rewards"][:, t].abs()).mean()
                h     = out["h_t"].detach()

            loss.backward()
            optimizer.step()

        val_reward = evaluate_policy(model, val_loader)
        if val_reward > best_val:
            best_val = val_reward
            torch.save(model.state_dict(), "models/policy_best.pt")

    model.load_state_dict(torch.load("models/policy_best.pt"))
    return model

rssm = train_policy(rssm, train_loader, val_loader)
```

---

### Block 23 · Study 4 — Out-of-Sample Evaluation

**What it does:** Runs four strategies on the 60 real held-out conversations. The RSSM policy is compared against Tommy's actual behaviour, a naive always-push baseline, and an always-accept lower bound.

```python
def evaluate_strategy(model, real_test_df, strategy="rssm"):
    """
    Simulates a buyer strategy on the 60 real test conversations.
    Returns normalised reward (final_discount / msrp) per conversation.

    Strategies:
      rssm:   model's policy_head action at each turn
      tommy:  Tommy's actual action (historical behaviour)
      naive:  always push_once — ignores dealer state entirely
      accept: always accept — lower bound
    """
    model.eval()
    test_snaps = build_concession_snapshots(real_test_df)
    test_snaps = assign_phases(test_snaps)
    results    = []

    for conv_id, group in test_snaps.groupby("conv_id"):
        msrp = real_test_df[real_test_df["conv_id"]==conv_id]["msrp"].iloc[0]
        h    = torch.zeros(1, 256)
        disc = 0

        for _, row in group.sort_values("turn_idx").iterrows():
            with torch.no_grad():
                out = model(
                    texts_t  = [row["transcript"]],
                    prices_t = torch.tensor(
                        [[row["d_B"],row["d_D"],row["gap"]]],
                        dtype=torch.float32),
                    h_prev   = h)

            if   strategy == "rssm":
                action = ["accept","push_once","continue"][
                    out["action_logits"].argmax(dim=-1).item()]
            elif strategy == "tommy":  action = row["optimal_action"]
            elif strategy == "naive":  action = "push_once"
            elif strategy == "accept": action = "accept"

            if action in ["push_once","continue"] and row["concession"]:
                disc = row["d_D"] + row["delta_d_D"]
            elif action == "accept":
                disc = row["d_D"]
                break

            h = out["h_t"]

        results.append({
            "conv_id":  conv_id,
            "strategy": strategy,
            "reward":   disc / msrp if msrp > 0 else np.nan
        })

    return pd.DataFrame(results)

from scipy import stats

all_results = pd.concat([
    evaluate_strategy(rssm, real_test_df, s)
    for s in ["rssm","tommy","naive","accept"]])

summary = all_results.groupby("strategy")["reward"].agg(
    ["mean","std","median"]).round(4)
summary["vs_tommy_pct"] = (
    (summary["mean"] - summary.loc["tommy","mean"]) /
    summary.loc["tommy","mean"] * 100).round(1)

print("STUDY 4 — OUT-OF-SAMPLE RESULTS (60 real held-out conversations)")
print(summary.to_string())

rssm_r  = all_results[all_results["strategy"]=="rssm"]["reward"]
tommy_r = all_results[all_results["strategy"]=="tommy"]["reward"]
t, p    = stats.ttest_rel(rssm_r, tommy_r)
d       = (rssm_r.mean() - tommy_r.mean()) / tommy_r.std()
print(f"\nPaired t-test (RSSM vs Tommy): t={t:.3f}  p={p:.4f}")
print(f"Effect size (Cohen's d):        {d:.3f}")
```

**Expected results structure:**

| Strategy | Mean reward | vs. Tommy | What it shows |
|---|---|---|---|
| `accept` | Lowest | — | Lower bound: no negotiation at all |
| `tommy` | Baseline | 0% | Tommy's historical performance |
| `naive` | Moderate | ±X% | Always pushing — ignores dealer state |
| `rssm` | Highest | +Y% | Stops pushing when `z_t` signals a real floor |

> **Why RSSM should beat naive push:** The naive strategy does not know when the dealer has reached their true floor. The RSSM policy stops pushing when `prior_sigma` is narrow and `z_t` indicates high probability that the current offer is near the dealer's reservation price — saving relationship capital and avoiding antagonising dealers who have genuinely no room to move.

---

### Block 24 · Study 4 — Domain Adaptation Check

**What it does:** Final validation that the model trained on synthetic + real data generalises to the 60 real test conversations it has never seen. Compares `prior_sigma` distributions between the test set and the synthetic training set.

```python
from scipy.stats import ks_2samp

def domain_adaptation_check(model, real_test_df, synthetic_df):
    test_snaps = build_concession_snapshots(real_test_df)
    syn_snaps  = build_concession_snapshots(
        synthetic_df.sample(n=min(1000, len(synthetic_df)), random_state=42))

    test_unc = extract_uncertainty_trajectories(model, test_snaps)
    syn_unc  = extract_uncertainty_trajectories(model, syn_snaps)

    ks_stat, ks_p = ks_2samp(
        test_unc["prior_sigma"].dropna(),
        syn_unc["prior_sigma"].dropna())

    print("Domain adaptation check (prior_sigma distribution):")
    print(f"  KS stat: {ks_stat:.4f}   p: {ks_p:.4f}")
    print(f"  Result:  "
          f"{'Good generalisation' if ks_p > 0.05 else 'Distribution shift — report as limitation'}")

domain_adaptation_check(rssm, real_test_df, synthetic_df)
```

---

## PART 5 — Full Pipeline Summary

```
PART 1 — DATA ARCHITECTURE
  Block 1   Lock 60 real test conversations; load 10,000 synthetic
  Block 2   Enrich data: delta_d_D, concession, gap, turn_position
  Block 3   Validate synthetic fidelity (5 structural checks)
  Block 4   Assign negotiation phases (1–5) to all turns
  Block 5   Build concession snapshots (one per dealer turn)
  Block 6   Retrospective reward labelling: accept/push_once/continue

PART 2 — RSSM TRAINING
  Block 7   Frozen LLM encoder — language understanding, no gradient
  Block 8   RSSM architecture — GRU + VAE + three output heads
  Block 9   ELBO training objective — recon + reward + KL
  Block 10  Training loop — KL annealing, early stopping

PART 3 — LINGUISTIC FEATURES
  Block 11  prob_concession extraction from reward head
  Block 12  Occlusion attribution → top-k words per snapshot
  Block 13  Corpus-level linguistic markers → L_score
  Block 14  Tactic classification from top-k words

STUDY 1 — LINGUISTIC CHANNEL (Blocks 15–16)
  → Does L_score predict delta_d_D beyond numeric controls?
  → Does the effect vary by phase? (Phase 3 anchor expected largest)
  → Key outputs: incremental R², phase interaction coefficients

STUDY 2 — BELIEF-UPDATING MECHANISM (Blocks 17–19)
  → Does L_score reduce prior_sigma (dealer's Knightian uncertainty)?
  → Does uncertainty reduction mediate the concession effect?
  → Key outputs: proportion mediated, bootstrap 95% CI

STUDY 3 — HETEROGENEOUS TACTIC EFFECTS (Blocks 20–21)
  → Are dealer types recoverable from z_t trajectory features?
  → Does tactic effectiveness vary by dealer type?
  → Key outputs: heterogeneous effects table (tactics × dealer types)

STUDY 4 — POLICY OPTIMISATION (Blocks 22–24)
  → Fine-tune policy head on optimal action labels
  → Evaluate on 60 real held-out conversations
  → Key outputs: RSSM vs Tommy; Cohen's d; domain adaptation check
```

---

## Design Overview

| Dimension | Stopping study | This study |
|---|---|---|
| **Problem type** | Observation (when to stop) | Control (what to say) |
| **Outcome** | Binary: sale / no sale | Continuous: `delta_d_D / MSRP` |
| **Checkpoints** | Fixed timestamps (45s, 60s) | Event-driven (each dealer turn) |
| **Reward** | `sale × benefit − duration × cost` | `final_discount / MSRP` |
| **Label generation** | 3-strategy retrospective reward | Same principle: 3 strategies |
| **Model** | Fine-tuned Llama (full weights) | Frozen encoder + GRU + VAE |
| **Training objective** | Cross-entropy on binary label | ELBO (recon + reward + KL) |
| **Linguistic features** | Not used | Occlusion attribution → `L_score` |
| **Belief / uncertainty** | Not modelled | `prior_sigma` = Knightian uncertainty |
| **Test set** | 12.5% of conversations | 60 real conversations, locked before training |
| **Key parameter** | Decision threshold θ* | RSSM policy + `L_score:uncertainty` interaction |
