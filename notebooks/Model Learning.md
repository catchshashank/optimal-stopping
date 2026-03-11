# How the Model Learns: A Sequential Explanation

---

## The Core Idea

The model starts knowing nothing about sales calls. It is a general language model
that has read most of the internet — it understands English, but has no opinion about
whether a sales call will succeed. Training teaches it to develop that opinion.

The way it learns is through **repeated mistake correction**. Show it a transcript,
ask it to predict the outcome, check if it was right, measure how wrong it was, and
nudge its internal settings slightly in the direction of being more right. Repeat
thousands of times.

---

## Step 1 — What the Model Looks Like Inside

The model is a neural network with **3 billion parameters** — think of these as
3 billion individual dials, each set to some number. Together, these dials determine
how the model responds to any input.

Before training, these dials are set to Meta's pre-trained values — good for general
language tasks but not calibrated for sales call decisions. Training adjusts these
dials, very slightly, over many examples.

The goal is to end up with a set of dial positions where the model consistently
assigns high probability to `"yes"` for transcripts that lead to sales, and high
probability to `"no"` for transcripts that do not.

---

## Step 2 — One Training Example

Take a single training example:

- **Input (prompt):** *"Below is the first 45 seconds of the sales call… Will this
  call end in a sale (yes/no):"*
- **Correct answer (label):** `"yes"`

The model reads the prompt and produces a probability distribution over all 32,000
words in its vocabulary. At this point in training it might say:
```
P("yes") = 0.31
P("no")  = 0.44
P("the") = 0.02
... (32,000 other tiny probabilities)
```

The correct answer is `"yes"` but the model gave it only 31% probability.
It was wrong — and measurably so.

---

## Step 3 — Measuring How Wrong: The Loss Function

The training algorithm measures wrongness using **cross-entropy loss**:

$$\mathcal{L} = -\log P(\text{correct token})$$

In this example:

$$\mathcal{L} = -\log(0.31) = 1.17$$

The higher this number, the more wrong the model was. If the model had been
perfectly confident — P("yes") = 1.0 — the loss would be:

$$\mathcal{L} = -\log(1.0) = 0$$

**The masking connection:** Prompt tokens received label = −100 during tokenisation.
The loss function skips any position with label = −100. So the loss is computed
*only* at the `"yes"`/`"no"` token position — the model is not graded on predicting
words in the transcript, only on whether it predicted the right action.

---

## Step 4 — Backpropagation: Tracing the Blame

Once the loss is computed, the algorithm asks: *which of the 3 billion dials
contributed most to this mistake?*

This is done through **backpropagation** — working backwards through the network
from the loss, calculating how much each dial's current value contributed to the
error. This produces a **gradient** for every single parameter:

$$\frac{\partial \mathcal{L}}{\partial w_i}$$

Think of the gradient as a signed receipt:

> *"Parameter 47,823,901 made the error 0.0003 worse.
> Parameter 2,109,442 made it 0.0001 better."*

A **positive gradient** means increasing that dial made things worse.
A **negative gradient** means increasing it made things better.

---

## Step 5 — Gradient Descent: Adjusting the Dials

The optimizer (**AdamW**) uses the gradients to update every parameter:

$$w_i \leftarrow w_i - \eta \cdot \frac{\partial \mathcal{L}}{\partial w_i}$$

Where η (eta) is the **learning rate** = 0.0001 in this notebook.

In plain English: *nudge every dial slightly in the direction that reduces the loss.*

| Learning Rate | Risk |
|---|---|
| Too large | Model overshoots — like overcorrecting a steering wheel |
| Too small | Learning takes forever |
| 0.0001 ✓ | Stable, controlled convergence |

**AdamW specifically** keeps a running memory of recent gradients and adapts the
nudge size individually per parameter — parameters receiving large gradients get
smaller nudges, and vice versa. This makes training more stable than standard
gradient descent.

---

## Step 6 — Batching: Learning from 12 Examples at Once

The notebook sets `batch_size = 12`, meaning the model processes 12 training
examples simultaneously before updating its parameters.

The losses from all 12 examples are averaged:

$$\mathcal{L}_{\text{batch}} = \frac{1}{12} \sum_{i=1}^{12} \mathcal{L}_i$$

Then one combined parameter update is made. This is more efficient than updating
after every single example, and the averaging makes the gradient estimate more
stable — one unusual call does not disproportionately jerk the model in the
wrong direction.

---

## Step 7 — One Epoch: Seeing All the Data Once

After processing all training examples in batches of 12, the model has completed
**one epoch**. The notebook allows up to 10 epochs.

| Epoch | What the Model Has Learned | Val AUC |
|---|---|:---:|
| 1 | Almost nothing — still effectively random | 0.08 |
| 2 | Starting to pick up some linguistic signals | 0.56 |
| 3 | Has learned most of the pattern | 0.95 |
| 4 | Has fully separated the two classes | **1.00** |
| — | Early stopping triggers — training ends | — |

The rapid jump from 0.08 to 1.00 in just 4 epochs happens because the base model
already understands English deeply. It does not need to learn what words mean —
it only needs to learn *which patterns in sales conversations correlate with success*.

---

## Step 8 — Early Stopping: Knowing When to Quit

After every epoch, the model is evaluated on the **validation set** — calls it has
never trained on. If the AUC does not improve for 1 full epoch (`patience = 1`),
training stops automatically and the best checkpoint is restored.

This prevents **overfitting** — the failure mode where a model memorises the
training examples perfectly but performs poorly on new data.

> Think of a student who memorises past exam papers but cannot handle new questions.

---

## The Full Learning Loop
```
Training data (transcript → correct action)
        ↓
Model reads transcript
        ↓
Produces probability distribution over all 32,000 words
        ↓
Loss = −log P(correct action)       ← only graded on action token
        ↓
Backpropagation traces blame to each of 3B parameters
        ↓
AdamW nudges all 3B dials slightly toward lower loss
        ↓
Repeat for next batch of 12 examples
        ↓
Repeat for all batches → one epoch complete
        ↓
Check validation AUC → improved?  →  continue
                     → no improvement?  →  stop, restore best model
```

---

## The Intuition in One Sentence

> Every training step is the model being shown a sales transcript, making a
> prediction, being told how wrong it was, and adjusting 3 billion internal
> settings by a tiny amount so it would be slightly less wrong next time —
> repeated thousands of times until it can reliably tell a promising call
> from a lost cause.
