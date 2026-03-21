# ERROR_ANALYSIS.md
## ArvyaX Emotion Guidance System — Failure Case Analysis

This document analyses 10 representative failure cases, explaining what went wrong, why, and how to improve.

---

## Case 1 — SHORT INPUT (most common failure type)

**Input:**
```
journal_text: "ok session"
stress_level: 3  |  energy_level: 2  |  sleep_hours: 7
reflection_quality: conflicted
```

**True label:** `restless` (intensity 3)
**Predicted:** `neutral` (intensity 2)
**Confidence:** 0.42 → `uncertain_flag = 1`

**What went wrong:**
"ok session" has 2 content tokens — both are vague filler words with no emotional signal. The model defaulted to the majority-class-adjacent prediction (`neutral`) because it had nothing else to go on. The metadata (energy=2, stress=3) suggested mild restlessness but wasn't strong enough to override.

**Why the model failed:**
TF-IDF on 2 tokens produces a near-zero feature vector. The model essentially predicts from metadata alone for these rows, and metadata signals are weak (all states cluster at ~3.0 for stress/energy).

**How to improve:**
When `text_quality_flag=1`, switch to a metadata-primary inference path that upweights stress/energy/sleep signals by 2×. Alternatively, prompt the user: *"Tell us a little more about how you're feeling"* before accepting a 2-word reflection.

---

## Case 2 — CONFLICTING SIGNALS (text says positive, metadata says stressed)

**Input:**
```
journal_text: "lowkey felt pretty grounded. i had to restart once."
stress_level: 5  |  energy_level: 1  |  sleep_hours: 8.5
reflection_quality: vague
```

**True label:** `overwhelmed` (intensity 4)
**Predicted:** `calm` (intensity 2)
**Confidence:** 0.61

**What went wrong:**
The word "grounded" has a strong positive TF-IDF weight in the `calm` class (top word for calm state). The model followed the text signal. But `stress=5`, `energy=1` is a classic exhaustion/overwhelm pattern that the metadata should have flagged.

**Why the model failed:**
The model averages text and metadata signals rather than detecting their conflict. When text says "grounded" and metadata says "stress=5, energy=1", averaging gives a misleading result.

**How to improve:**
Add explicit conflict detection: when `text_quality_flag=0` AND `stress≥4` AND text sentiment is positive, set `contradiction_flag=1` and apply a rule override. The decision engine already handles `contradiction_flag=1` — the gap is in detecting it from text+metadata mismatch, not just numeric mismatch.

---

## Case 3 — SIMILAR CLASS CONFUSION (calm vs neutral)

**Input:**
```
journal_text: "The rain ambience was pleasant, though I can't say it shifted my mood much."
stress_level: 2  |  energy_level: 3  |  reflection_quality: conflicted
```

**True label:** `neutral` (intensity 2)
**Predicted:** `calm` (intensity 2)
**Confidence:** 0.58

**What went wrong:**
"Pleasant" and low stress point toward calm. "Can't say it shifted my mood" points toward neutral. Both classes share overlapping vocabulary. The model picked the slightly more probable class.

**Why the model failed:**
`calm` and `neutral` are semantically adjacent — users themselves may struggle to distinguish them. The decision boundary between these two classes is inherently fuzzy. This is not really a model failure; it's a label taxonomy problem.

**How to improve:**
Consider merging `calm` and `neutral` into a single `settled` state. Both classes lead to similar decision engine outputs (deep_work or light_planning) anyway, so the practical impact of confusing them is low. Alternatively, use a 2D valence × arousal representation where calm = (positive, low arousal) and neutral = (neutral, low arousal) — the separation becomes easier.

---

## Case 4 — NOISY LABEL

**Input:**
```
journal_text: "ended up like everything piled up. then my mind wandered again."
stress_level: 4  |  energy_level: 2  |  reflection_quality: clear
```

**True label:** `mixed` (intensity 2)
**Predicted:** `overwhelmed` (intensity 3)
**Confidence:** 0.69

**What went wrong:**
"Everything piled up" is a strong overwhelmed signal. "Mind wandered" is a restless signal. The model predicted overwhelmed at intensity 3, which is arguably more correct than the label `mixed` at intensity 2. This appears to be a mislabelled training example.

**Why the model failed:**
The ground-truth label may itself be wrong. When the labeller wrote `mixed` they may have been in a different mental state than the journal text suggests. With subjective wellness data, ~10–15% label noise is expected.

**How to improve:**
Apply label smoothing: treat each label as 90% confident, 10% distributed across adjacent classes. Identify probable mislabels using cross-validation consensus — when all 5 CV folds predict `overwhelmed` but the label is `mixed`, flag for review. Downweight these samples during training.

---

## Case 5 — INTENSITY MISMATCH (state correct, intensity wrong)

**Input:**
```
journal_text: "so so tired. nothing feels real. completely drained after the session."
emotional_state: tired / overwhelmed  |  True intensity: 5
```

**True label:** `overwhelmed` (intensity 5)
**Predicted:** `overwhelmed` (intensity 3)
**Confidence:** 0.72

**What went wrong:**
Emotional state was correctly identified but intensity was off by 2. The repetition ("so so tired") and extreme language ("nothing feels real", "completely drained") are strong intensity-5 markers that the model underweighted.

**Why the model failed:**
The ordinal model doesn't have explicit features for linguistic intensity markers. TF-IDF treats "so so tired" the same as "so tired" — it doesn't count repetitions. Extreme adjectives like "completely" are in the vocabulary but their intensity signal may be diluted across many contexts.

**How to improve:**
Add explicit intensity features: repetition count (count of repeated words), exclamation marks, all-caps word count, extreme modifier count ("completely", "utterly", "nothing", "everything"). These are simple regex features that don't require any model.

---

## Case 6 — SARCASM / IRONY

**Input:**
```
journal_text: "oh great, another amazing day of doing absolutely nothing useful."
stress_level: 3  |  energy_level: 2  |  reflection_quality: vague
```

**True label:** `restless` (intensity 3)
**Predicted:** `focused` (intensity 2)
**Confidence:** 0.55

**What went wrong:**
"Great" and "amazing" are strong positive TF-IDF signals in the `focused`/`calm` vocabulary. The model took them at face value. The sarcastic framing ("oh great", "absolutely nothing useful") was not detected.

**Why the model failed:**
TF-IDF has no concept of tone or irony. "Great" in "oh great" means the opposite of "great" in "had a great session" — but TF-IDF treats both identically. Sarcasm detection is genuinely hard without a pre-trained language model.

**How to improve:**
Add a simple sarcasm heuristic: when positive words (great, amazing, wonderful) co-occur with irony markers (oh, sure, yeah right, absolutely nothing), apply a tone-flip flag. Not perfect but catches common patterns. Alternatively, use a small fine-tuned DistilBERT — it handles negation and irony natively.

---

## Case 7 — TEMPORAL REFERENCE WITHOUT CONTEXT

**Input:**
```
journal_text: "same as yesterday. feeling much better now though."
previous_day_mood: overwhelmed
```

**True label:** `calm` (intensity 2)
**Predicted:** `mixed` (intensity 3)
**Confidence:** 0.53  →  `uncertain_flag = 1`

**What went wrong:**
"Feeling much better now" implies improvement from a previous negative state. The model predicted `mixed` because it couldn't contextualise "better" relative to "overwhelmed yesterday". "Better than overwhelmed" = calm, but the model saw "better" in isolation.

**Why the model failed:**
The model treats `previous_day_mood` as an encoded categorical with no temporal reasoning. It doesn't compute "mood delta" — the change from yesterday to today.

**How to improve:**
Add a `mood_delta` engineered feature: encode `previous_day_mood` on a scale and subtract from predicted state probability. References to "better than yesterday" in text should trigger upward intensity adjustment. A simple keyword detector for "better than", "more than yesterday", "less than before" could capture this.

---

## Case 8 — TEMPLATE TEXT AMBIGUITY

**Input:**
```
journal_text: "honestly i felt mentally flooded. i couldn't tell if it was helping at first."
stress_level: 5  |  energy_level: 4  |  reflection_quality: clear
```

**True label:** `overwhelmed` (intensity 4)
**Predicted:** `restless` (intensity 4)
**Confidence:** 0.60

**What went wrong:**
"Mentally flooded" is a strong overwhelmed signal. However, "high stress AND high energy" (5, 4) pattern is equally associated with `restless` in the training data. The model split its vote between the two states and landed on restless.

**Why the model failed:**
"Mentally flooded" is a phrase that appears in the training data with mixed labels (sometimes `overwhelmed`, sometimes `restless`). The dataset contains many templated phrases ("honestly i felt...", "i couldn't tell if...") used across multiple emotional states. These templates reduce the discriminative power of TF-IDF bigrams.

**How to improve:**
The templated phrases in this dataset are a form of label noise. Identify template-heavy rows and downweight their bigrams in TF-IDF (increase `min_df` for phrases that appear across ≥3 different state classes). Focus importance on the non-template words that carry actual signal.

---

## Case 9 — RARE CLASS (focused)

**Input:**
```
journal_text: "woke up feeling able to prioritize. mountain visuals made it easier to pause."
stress_level: 1  |  energy_level: 4  |  time_of_day: morning
```

**True label:** `focused` (intensity 3)
**Predicted:** `calm` (intensity 2)
**Confidence:** 0.64

**What went wrong:**
`focused` (193 samples) is the least common class. "Able to prioritize" is a focused-state phrase but also appears in calm contexts. The model slightly underweights focused states because it has seen fewer examples.

**Why the model failed:**
Despite balanced class weights, the model may still underperform on the least common class because less diverse training examples were seen. The decision boundary for `focused` is less well-defined.

**How to improve:**
Oversample `focused` class slightly using SMOTE on the feature vectors (not raw text). Generate 10–15 synthetic examples by interpolating between existing focused samples. Also ensure the decision engine treats `focused` predictions as high-confidence action triggers (→ deep_work immediately) regardless of uncertainty.

---

## Case 10 — CONFIDENT WRONG PREDICTION

**Input:**
```
journal_text: "by the end i was locked in for a bit. i kept thinking about emails."
stress_level: 5  |  energy_level: 1  |  reflection_quality: clear
```

**True label:** `overwhelmed` (intensity 4)
**Predicted:** `restless` (intensity 3)
**Confidence:** 0.77  ←  HIGH CONFIDENCE, WRONG

**What went wrong:**
"Locked in" is a focus signal. "Kept thinking about emails" is a restless signal. The model was very confident in `restless` but the true state was `overwhelmed`. The metadata (stress=5, energy=1) strongly signals overwhelmed, but the model overweighted the text.

**Why the model failed:**
When the model is highly confident but wrong, it usually means a training distribution gap — the model has seen many "kept thinking about emails → restless" examples but few where this phrase co-occurs with stress=5, energy=1. It didn't learn that this phrase pattern can mean different things under different metadata conditions.

**How to improve:**
Apply **temperature scaling** to reduce overconfidence. After training, fit a single temperature parameter T on a validation set: `calibrated_prob = softmax(logits / T)`. This makes the 0.77 confidence more like 0.60, which correctly flags it as uncertain. This is one of the most impactful post-training calibration improvements.

---

## Cross-Cutting Insights

### 1. Short text is the #1 failure cause
115 training texts (9.6%) have ≤3 tokens. These give almost zero signal. The model should either prompt for more text or explicitly switch to a metadata-primary inference path for these rows.

### 2. Sarcasm and irony are unsolvable with TF-IDF alone
A minimum viable fix: co-occurrence heuristic for irony markers + positive words. Full fix: a small fine-tuned DistilBERT on the reflection text.

### 3. Confident wrong predictions are the most dangerous
When `confidence > 0.7` and the prediction is wrong, it's a training distribution gap. Temperature scaling after training reduces these systematic overconfidence errors at zero training cost.

### 4. Semantically similar states confuse each other
`calm` vs `neutral`, `restless` vs `overwhelmed` — these are inherently fuzzy. Recommend: merge similar states or use a 2D (valence × arousal) representation in a future version.

### 5. The templated text in this dataset hurts TF-IDF
Many journal entries follow templates ("honestly i felt...", "by the end i was..."). These templates appear across multiple state classes. Identifying and downweighting template bigrams would improve TF-IDF discrimination.

### 6. Label noise is real and unavoidable
Subjective wellness labels are inconsistent. Label smoothing (ε=0.1) and cross-validation consensus flagging of suspected mislabels are the most practical remedies without collecting new data.
