# Code Walkthrough: Optimal Talking in Negotiations

> **What this notebook does:** It trains an AI agent to decide, during a live sales call, whether to keep the salesperson on the line or end the call early — because continuing a doomed call wastes money. The notebook uses a technique called *Behavioral Cloning*: it first figures out, from historical data, what the *correct* decision would have been at each moment, and then teaches a large language model (LLM) to replicate those correct decisions.

---

## PART 1: Setting Up and Preparing the Data

---

### Block 1 · Setting Business Rules

```python
COST_PER_UNIT_TIME = 0.1
BENEFIT_PER_POSITIVE_OUTCOME = 10.0
DECISION_OPPORTUNITIES = [45, 60]
```

**What it does:** This block defines three critical business parameters.

| Code | Meaning |
|---|---|
| `COST_PER_UNIT_TIME = 0.1` | **Business rule:** Every second a salesperson stays on a call costs $0.10 (or whatever currency unit). This is the "price of time." |
| `BENEFIT_PER_POSITIVE_OUTCOME = 10.0` | **Business rule:** A completed sale is worth $10. So a sale must offset the time cost — creating a real financial trade-off. |
| `DECISION_OPPORTUNITIES = [45, 60]` | **Business rule:** The agent is allowed to make a "quit or stay?" decision at exactly the 45-second mark and the 60-second mark of every call. These are the two checkpoints. |

> **Code Logic:**
> - The ratio of benefit to cost (10 ÷ 0.1 = 100) means a sale is worth 100 seconds of call time.
> - If a call is heading nowhere past 100 seconds, it is already losing money.
> - In the paper, the cost c is derived more rigorously as the expected value of initiating a new call: 1 / (average_duration) × success_rate × time_per_period. The notebook uses a simplified fixed ratio.

---

### Block 2 · Loading Conversation Data

```python
dataset_url = "https://raw.githubusercontent.com/.../synthetic_sales_conversations.csv"
diarized_conversations = pd.read_csv(dataset_url)

diarized_conversations["is_sale"] = diarized_conversations["outcome"].apply(
    lambda x: 1 if x == "sale" else 0 if x == "no sale" else np.nan)

diarized_conversations["duration"] = diarized_conversations.groupby(
    "conversation_id")["end_time"].transform("max")
```

**What it does:** Downloads a dataset of synthetic sales calls and creates two new summary columns.

| Code | Meaning |
|---|---|
| `diarized_conversations["is_sale"] = ...` | Adds a new column: 1 if the call resulted in a sale, 0 if not. This is the ultimate **outcome label** the model needs to learn from. |
| `lambda x: 1 if x == "sale" else 0 ...` | A one-line rule applied to every row: "if the outcome says 'sale', write 1; if 'no sale', write 0; otherwise leave blank." |
| `groupby("conversation_id")["end_time"].transform("max")` | For every conversation, finds the timestamp of the very last utterance — i.e., how long the entire call lasted. Adds this as a `duration` column on every row of that conversation. |

> **Code Logic:**
>  - The dataset is in *diarized* format — each row is a separate speech turn labelled by speaker ("Speaker 0" = salesperson, "Speaker 1" = customer).
>  - Every speech turn is timestamped and labelled by the final outcome, so that the model can learn from each partial transcript prefix whether the reward-maximizing action is to "wait" or "quit", given the call’s eventual outcome and timing.

#### **Outcome Labels in Paper:**

The outcome label (is_sale or not) comes from the CRM / call logging system at the firm, not from any model or annotation process. The paper describes this process partially in `Section 4.1`:

> In addition, we observe the salesperson’s identifier, the call outcome (i.e., whether the call resulted in a sale based on the consumer
> confirming the energy contract, hereafter call success), and metadata such as the call start and end times.

So is_sale = 1 was recorded when the consumer verbally confirmed the energy contract during the call. This was captured automatically by the firm's operational system alongside the transcript. The dataset also includes the salesperson identifier and call start/end times as co-recorded metadata.

---

### Block 3 · Splitting Data into Train, Validation, and Test Sets

```python
train_conversation_ids, test_conversation_ids, ... = train_test_split(
    all_conversation_ids, all_outcomes, test_size=0.25, stratify=all_outcomes)

train_conversation_ids, val_conversation_ids, ... = train_test_split(
    train_conversation_ids, train_outcomes, test_size=0.25, stratify=train_outcomes)
```

**What it does:** Divides all conversations into three separate groups so the model is tested fairly.

| Group | Size | Purpose |
|---|---|---|
| **Training set** | ~75% | The AI learns from these conversations |
| **Validation set** | ~12.5% | Used during training to check progress and stop early if needed |
| **Test set** | ~12.5% | Held back completely — used only at the very end to report final results |

**Key detail — `stratify=all_outcomes`:** This ensures each group has the same *proportion* of sales vs. non-sales. Without this, one group could accidentally have mostly easy cases. This ensures the model trains proportionately on both "sale" and "no sale" outcomes.

---

### Block 4 · Building Transcript Snapshots at Each Checkpoint

```python
m1, m2 = sorted(DECISION_OPPORTUNITIES)   # m1=45, m2=60

for m in [m1, m2]:
    transcripts[m] = data_transcripts[dftype].loc[
        (data_transcripts[dftype]["end_time"] >= 0) &
        (data_transcripts[dftype]["end_time"] < m)
    ].groupby("conversation_id")["transcript"].apply(lambda x: '\n'.join(x))
```

**What it does:** For every conversation, takes a "photograph" of the transcript at the 45-second mark and another at the 60-second mark — capturing only what has been said *so far*.

| Code | Meaning |
|---|---|
| `m1, m2 = sorted(DECISION_OPPORTUNITIES)` | Unpacks the two checkpoints into `m1 = 45` and `m2 = 60` in sorted order. |
| `data_transcripts[dftype]["end_time"] < m` | Filters to only include utterances that *finished before* the checkpoint. We cannot use words spoken after the decision point. |
| `groupby("conversation_id")["transcript"].apply(lambda x: '\n'.join(x))` | Collects all speech turns for each conversation and joins them into one block of text, separated by new lines — like assembling a readable transcript. |

> **Code Logic:**
>  - This is equivalent of saying "at 45 seconds, what do we *actually know*?"
>  - The agent must make its decision with only the information available at that moment.

---

### Block 5 · Wrapping Transcripts into Prompts (States)

```python
def convert_to_state(transcript, t):
    state = "Below is the first " + str(t) +
            " seconds of the sales call between the sales agent Speaker 0 and"
            " the customer Speaker 1:\n" + transcript + "\n" +
            "Will this call end in a sale (respond with 'yes' or 'no'):  "
    return state
```

**What it does:** Formats each transcript snapshot into a natural-language question (prompt) the LLM can read and answer.

| Code | Meaning |
|---|---|
| `"Below is the first " + str(t) + " seconds..."` | Provides the LLM with context — it knows it is reading only the first 45 or 60 seconds of a call. |
| `transcript` | The actual conversation text is inserted here. |
| `"Will this call end in a sale (respond with 'yes' or 'no'):"` | The question the LLM must answer. The model is being asked to predict the final outcome from partial information. |

> **Code Logic:**
>  - The code builds the *prompt* that is subsequently fed to the LLM. This is the *feedstock* for training and testing the model subsequently.
>  - Each of these *prompt* becomes the *state* for which the model is asked to choose an *act* — quit or wait — in the subsequent code block based on a reward function.

---

### Block 6 · Computing the Optimal Decision at Each Checkpoint or *state-act* pairs *(most important block)*

```python
df["rq" + str(m1)] = -m1 * COST_PER_UNIT_TIME                    # reward if salesperson quits at 45s

df["rq" + str(m2)] = df["is_sale"] * BENEFIT_PER_POSITIVE_OUTCOME \ #rew
                   * (df["duration"] <= m2) \
                   - df["duration"].apply(lambda x: min(m2, x)) * COST_PER_UNIT_TIME

df["rc" + str(m2)] = df["is_sale"] * BENEFIT_PER_POSITIVE_OUTCOME \
                   - df["duration"] * COST_PER_UNIT_TIME

df["max_reward"] = df[["rq45", "rq60", "rc60"]].max(axis=1)
```

**What it does:** For every historical call — where we *already know* the outcome — calculates what the financially optimal decision would have been. This is how the "correct answer" labels are generated without any human expert.

**Three strategies evaluated:**

| Strategy Column | Meaning | Reward Formula |
|---|---|---|
| `rq45` | Quit at 45 seconds | −45 × 0.1 = **−$4.50** always (fixed cost, no sales benefit) |
| `rq60` | Wait to 60s, then quit | Sale benefit (if call ≤ 60s long) minus cost of time = `sale × 10 × (duration ≤ 60) − min(60, duration) × 0.1` |
| `rc60` | Never quit — let it run | Full sale benefit minus total call duration cost = sale × 10 − duration × 0.1|

| Code | Meaning |
|---|---|
| `-m1 * COST_PER_UNIT_TIME` | Quitting at 45 seconds always costs exactly −$4.50. There is no benefit because you end before a sale can be confirmed. |
| `df["is_sale"] * BENEFIT_PER_POSITIVE_OUTCOME` | If the call *would have* ended in a sale, credit the full $10 benefit. If not, this term is zero. |
| `* (df["duration"] <= m2)` | The sale benefit only applies if the call actually finished by the 60s mark — you cannot claim a sale from a call you hung up on. |
| `df[["rq45", "rq60", "rc60"]].max(axis=1)` | For each call, picks the *highest reward* among the three strategies. That becomes the label: whatever strategy maximised reward is what the LLM should learn to do. |

> **Code Logic:**
>  - This step answers: *"With the benefit of hindsight, what should have been done?"*
>  - It is retrospective optimisation — using known outcomes to label past decisions, creating the training data without any costly human annotation.

### Optimal Action Assignment: Three Cases

#### Case 1 — Quit Early (`rq45` is max)

```python
df.loc[df["max_reward"]==df["rq45"], "a45"] = "no"
df.loc[df["max_reward"]==df["rq45"], "a60"] = "no"
```

**Call:** 120s duration, no sale

| Strategy | Calculation | Reward |
|---|---|---:|
| Quit at 45s (`rq45`) | 0 − 45 × 0.1 | **−4.50** ✓ |
| Quit at 60s (`rq60`) | 0 − 60 × 0.1 | −6.00 |
| Never quit (`rc60`) | 0 − 120 × 0.1 | −12.00 |

> No sale possible — cut losses at the earliest checkpoint. Both `a45` and `a60` = `"no"`.

#### Case 2 — Wait Then Quit (`rq60` is max) → a45 = `"yes"`, a60 = `"no"`

```python
df.loc[df["max_reward"]==df["rq45"], "a45"] = "no"
df.loc[df["max_reward"]==df["rq45"], "a60"] = "no"
```

**Call:** 55s duration, sale

| Strategy | Calculation | Reward |
|---|---|---:|
| Quit at 45s (`rq45`) | 0 − 45 × 0.1 | −4.50 |
| Quit at 60s (`rq60`) | 10 × (55 ≤ 60) − 55 × 0.1 | **+4.50** ✓ |
| Never quit (`rc60`) | 10 − 55 × 0.1 | +4.50 |

> Sale completes at 55s — before 60s checkpoint. Stay through 45s to capture it, then `"no"` at 60s (call already over). `a45` = `"yes"`, `a60` = `"no"`.

#### Case 3 — Never Quit (`rc60` is max) → a45 = `"yes"`, a60 = `"yes"`
**Call:** 90s duration, sale

| Strategy | Calculation | Reward |
|---|---|---:|
| Quit at 45s (`rq45`) | 0 − 45 × 0.1 | −4.50 |
| Quit at 60s (`rq60`) | 10 × (90 ≤ 60) − 60 × 0.1 | −6.00 |
| Never quit (`rc60`) | 10 − 90 × 0.1 | **+1.00** ✓ |

> Sale completes at 90s — past both checkpoints. Quitting at either point destroys the sale. Stay through both. `a45` = `"yes"`, `a60` = `"yes"`.

---

## PART 2 — Training the LLM and Evaluating Results

---

### Block 7 · Loading the Llama Language Model

```python
huggingface_hub.login(token=HF_TOKEN)
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
```

**What it does:** Downloads and loads Meta's Llama 3.2 language model (3 billion parameters) — a powerful, open-source AI that reads and generates text.

| Code | Meaning |
|---|---|
| `"meta-llama/Llama-3.2-3B"` | Specifies the base (un-customised) version of Llama — the "blank slate" that will be trained on sales call decisions. |
| `AutoTokenizer.from_pretrained(...)` | Loads the *tokeniser* — the system that converts human text into the numerical codes the AI processes internally. |
| `AutoModelForCausalLM.from_pretrained(...)` | Loads the actual neural network weights. "CausalLM" means it is a text-generation model that reads text left to right and predicts what comes next. |
| `tokenizer.padding_side = 'left'` | A technical setting: when processing multiple calls in a batch, shorter transcripts are padded on the *left* side. This ensures the model's final computation always reflects the actual last word of the prompt, not a blank filler character. |

---

### Block 8 · Tokenising the Training Data with Label Masking

```python
def tokenize_fn(example, add_label):
    encoded_prompt = tokenizer.encode(tokenizer.bos_token + example["prompt"])
    if add_label:
        encoded_label = tokenizer.encode(example["completion"] + tokenizer.eos_token)
        return {
            "input_ids":      encoded_prompt + encoded_label,
            "attention_mask": [1] * (len(encoded_prompt) + len(encoded_label)),
            "labels":         [-100] * len(encoded_prompt) + encoded_label
        }
```

**What it does:** Converts each (prompt, action) pair into the number sequences the model trains on, and crucially *masks* the prompt so the model only learns from the action word.

| Code | Meaning |
|---|---|
| `tokenizer.bos_token + example["prompt"]` | Adds a special "beginning of sequence" marker before the transcript prompt — like a formal salutation that signals the start of a document. |
| `tokenizer.encode(example["completion"] + tokenizer.eos_token)` | Converts the action word ("yes" or "no") plus an "end of sequence" marker into numbers. |
| `"labels": [-100] * len(encoded_prompt) + encoded_label` | **The critical masking step.** The value −100 tells the training algorithm: "ignore this token — do not calculate error for these positions." Only the action tokens ("yes"/"no" + end marker) contribute to the learning signal, not the input prompt. |

> **Code Logic:**
>  - Without label masking, the model would waste effort learning to predict the *question* rather than the *answer*.
>  - Masking focuses 100% of learning on the decision: "given this transcript, should we quit?" This is equivalent to marking an exam where only the answer box is graded, not the working shown in the margin.

---

### Block 9 · Fine-Tuning the Model

```python
training_args = transformers.TrainingArguments(
    output_dir="./llama-3.2-3B/",
    num_train_epochs=10,
    learning_rate=1e-4,
    per_device_train_batch_size=12,
    metric_for_best_model="auc",
    load_best_model_at_end=True,
    bf16=True,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset.shuffle(),
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=1)]
)

trainer.train()
```

**What it does:** Runs the actual model training — showing the model thousands of (transcript → decision) examples so it learns to predict the right action.

**Key settings explained:**

| Setting | Value | Meaning |
|---|---|---|
| `num_train_epochs=10` | Up to 10 | Maximum number of complete passes through the training data. Training may stop earlier if the model converges. |
| `learning_rate=1e-4` | 0.0001 | How large a step the model takes when adjusting its internal parameters after each error. Too large = unstable learning; too small = very slow. |
| `per_device_train_batch_size=12` | 12 | How many calls are processed simultaneously in one learning step. Larger batches need more GPU memory. |
| `metric_for_best_model="auc"` | AUC | The model saves its best version based on AUC score (a measure of ranking accuracy) not raw loss — because AUC better predicts whether the threshold tuning step will work. |
| `load_best_model_at_end=True` | — | At the end of training, automatically reverts to whichever checkpoint had the highest AUC on the validation set. |
| `bf16=True` | — | Uses a memory-efficient number format (bfloat16) that halves GPU memory use with minimal accuracy loss. Requires a modern GPU. |
| `EarlyStoppingCallback(patience=1)` | — | If validation AUC does not improve for 1 full epoch, training stops automatically — preventing over-fitting and saving compute time. |

> **Code Logic:** In practice, training stopped at Epoch 4 when validation AUC reached 1.00 — perfect separation of "quit" vs. "wait" calls. The model effectively mastered the decision problem.

For more refer to the [Model Learning](model_learning.md) document.

---

### Block 10 · Generating Predictions on New Calls

```python
outputs = trainer.model.generate(
    **batch,
    max_new_tokens=2,
    do_sample=False,
    temperature=None, top_p=None, top_k=None,
    return_dict_in_generate=True, output_scores=True
)
logprobs = torch.log_softmax(scores, dim=-1)
```

**What it does:** Runs the trained model on unseen calls and records both its verbal answer ("yes"/"no") and its *confidence score* for that answer.

| Code | Meaning |
|---|---|
| `max_new_tokens=2` | The model is only allowed to generate 2 new words — we expect just "yes" or "no" followed by an end marker. |
| `do_sample=False` | Uses *greedy decoding*: always picks the single highest-probability word. This makes the output deterministic and consistent — the same call always gets the same answer. |
| `temperature=None, top_p=None, top_k=None` | Disables all randomness-injection mechanisms. The model gives its most confident, unambiguous answer each time. *Temperature* rescales the logits before softmax. *Top-k sampling* truncates the distribution to only the k highest-probability tokens, then resamples from those k options. *Top-p (nucleus) sampling* keeps the smallest set of tokens whose cumulative probability exceeds p.|
| `output_scores=True` | Returns the raw unnormalised probability score for the output `yes` or `no` at the decision point. |
| `torch.log_softmax(scores, dim=-1)` | Normalizes the probability scores (logarithmic scale). From these, we extract the probability assigned specifically to "yes" — the model's confidence that the call will end in a sale. |

> **Key insight:**
>  - The model never truly outputs just `yes` or `no` — it outputs a full probability distribution over its entire vocabulary at every step.
>  - These four settings together ensure we capture and correctly interpret that distribution, turning a language model into a calibrated confidence estimator.
---

### Block 11 · Computing Confidence Scores (`prob_yes`)

```python
predictions["prob"] = predictions["logprob"].apply(lambda x: math.exp(x))
predictions.loc[predictions["response"] == "yes", "prob_yes"] = predictions["prob"]
predictions.loc[predictions["response"] != "yes", "prob_yes"] = 1.0 - predictions["prob"]
```

**What it does:** Converts the model's raw output into a single number between 0 and 1 — the probability that this call will result in a sale.

| Logic | Meaning |
|---|---|
| If the model said **"yes"**: `prob_yes = prob` | The model's confidence in "yes" is taken directly. |
| If the model said **"no"**: `prob_yes = 1 − prob` | The model was confident in "no" — so its confidence in "yes" is the complement. |

> **Code Logic:** This gives us a continuous score (e.g., 0.03 = very likely to fail, 0.91 = very likely to succeed) rather than just a binary word. This score is what the threshold system uses to make the final quit/wait decision.

---

### Block 12 · Finding Optimal Decision Thresholds: Backward Induction

See the [Backward Induction](backward_induction.md) document.
```

### Block 13 · Final Results

```python
print("Test set reward WITH the stopping agent:")
print("Total reward:", total_reward_agent)
print("Sales lost by stopping agent:", total_sales_noagent - total_sales_agent)
print("Time saved (%):", (total_time_noagent - total_time_agent) / total_time_noagent * 100)
```

**What it reports:** The head-to-head comparison between letting every call run to completion versus using the AI stopping agent.

| Metric | No Agent | With Agent | Change |
|---|---|---|---|
| Avg. reward per call | −$1.56 | −$1.21 | **+23% improvement** |
| Total time spent (s) | 3,279 s | 3,002 s | **−8.5% time saved** |
| Total sales closed | 25 | 24 | −1 sale lost |

> **Code Logic:** The agent sacrifices one sale out of 25 to save 8.5% of total call time. That freed-up time can be reallocated to additional calls, which in expectation generates *more* sales than the one lost. This is the core business case: intelligent early stopping is better than letting every call run regardless of signal.

---
