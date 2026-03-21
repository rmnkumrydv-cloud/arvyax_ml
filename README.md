# 🌿 ArvyaX Emotion Guidance System
**Team ArvyaX · RevoltronX — ML Internship Assignment**
*Theme: From Understanding Humans → To Guiding Them*

---

## What This System Does

Takes a user's post-session journal reflection + lightweight signals (sleep, stress, energy, time) and produces:
- Predicted **emotional state** (calm / restless / focused / neutral / mixed / overwhelmed)
- Predicted **intensity** (1–5 ordinal scale)
- **Recommended action** (what to do + when)
- **Confidence score** + **uncertain flag**
- **Supportive message** explaining the recommendation

---

## Project Structure

```
arvyax_project/
│
├── data/
│   ├── train.csv                   ← 1200 labelled training samples
│   └── test.csv                    ← 120 test samples
│
├── EDA.ipynb                       ← Exploratory Data Analysis (13 charts)
├── FE_ModelTraining.ipynb          ← Feature Engineering + Model Training
├── ArvyaX_Bonus.ipynb              ← Label noise handling + SLM messages
│
├── api.py                          ← FastAPI local REST API (Bonus)
├── app_ui.py                       ← Streamlit UI demo (Bonus)
│
├── outputs/
│   ├── predictions.csv             ← FINAL SUBMISSION FILE
│   ├── train_processed.csv         ← Feature-engineered training data
│   ├── test_processed.csv          ← Feature-engineered test data
│   ├── plots/                      ← Confusion matrix, feature importance, ablation
│   └── artifacts/                  ← Saved models and transformers
│       ├── model_state.pkl         ← XGBoost emotional state model
│       ├── model_state_clean.pkl   ← Noise-aware retrained model (Bonus)
│       ├── model_intensity.pkl     ← Ordinal intensity classifiers (4 files)
│       ├── tfidf.pkl               ← TF-IDF vectorizer
│       ├── svd.pkl                 ← SVD reducer (150 dims)
│       ├── le_state.pkl            ← Label encoder (emotional state)
│       ├── le_face.pkl             ← Label encoder (face_emotion_hint)
│       ├── le_mood.pkl             ← Label encoder (previous_day_mood)
│       ├── meta_cols.pkl           ← List of metadata feature column names
│       ├── X_train.npy             ← Final feature matrix (train)
│       ├── X_test.npy              ← Final feature matrix (test)
│       └── y_train.npy             ← Encoded target labels
│
├── README.md                       ← This file
├── ERROR_ANALYSIS.md               ← 10 failure cases with explanations
└── EDGE_PLAN.md                    ← On-device deployment plan
```

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

# For Bonus API
pip install fastapi uvicorn

# For Bonus UI
pip install streamlit
```

### 2. Place data files
```
data/train.csv
data/test.csv
```

### 3. Run notebooks in order
```
1. EDA.ipynb                 → understand the data (optional, charts only)
2. FE_ModelTraining.ipynb    → trains models + saves predictions.csv
3. ArvyaX_Bonus.ipynb        → label noise handling + richer SLM messages
```

### 4. Run Bonus API (optional)
```bash
uvicorn api:app --reload --port 8000
```
Then open: `http://localhost:8000/docs` → interactive Swagger UI

### 5. Run Bonus UI (optional)
```bash
streamlit run app_ui.py
```
Then open: `http://localhost:8501`

---

## Approach

### Why NOT a standard classification pipeline?

This problem has four real-world complications:
1. **Noisy text** — "ok", "fine", "still off" carry near-zero signal
2. **Missing data** — face_emotion_hint missing 10% train / 16% test
3. **Conflicting signals** — high stress metadata but positive text
4. **Imperfect labels** — reflection_quality = vague/conflicted means labels are unreliable

Each is handled explicitly rather than ignored.

---

## Feature Engineering

| Feature | Formula | Why |
|---------|---------|-----|
| `sleep_deficit` | `8 - sleep_hours` | Captures deprivation; negative = well-rested |
| `low_sleep_flag` | `1 if sleep < 5h` | Clinical impairment threshold |
| `stress_energy_ratio` | `stress / (energy + 0.01)` | High = burnout risk |
| `stress_energy_product` | `stress × energy` | Both high = anxious activation |
| `net_wellbeing` | `energy - stress` | Simple positive/negative signal |
| `contradiction_flag` | `stress≥4 AND energy≥4` | Signals conflicted state |
| `productivity_proxy` | `duration × energy / stress` | Was session likely productive? |
| `text_len` | word count | Longer text = more reliable prediction |
| `text_quality_flag` | `1 if tokens ≤ 3` | Drives uncertain_flag=1 |

### Missing Value Strategy

| Column | Train Missing | Test Missing | Strategy |
|--------|-------------|-------------|----------|
| `face_emotion_hint` | 123 (10%) | 19 (16%) | Fill `'none'` + `face_missing` flag |
| `previous_day_mood` | 15 (1%) | 10 (8%) | Fill `'unknown'` + `mood_missing` flag |
| `sleep_hours` | 7 (1%) | 0 | Fill training median (6.0h) + `sleep_missing` flag |

**Rule**: All imputers fit on training data only. Same values applied to test. No data leakage.

### Categorical Encoding

| Column | Method | Reason |
|--------|--------|--------|
| `time_of_day` | Ordinal (0–4) | Natural time order exists |
| `reflection_quality` | Ordinal (0–2) | vague < conflicted < clear |
| `ambience_type` | One-Hot | No natural order |
| `face_emotion_hint` | Label Encode | Many categories |
| `previous_day_mood` | Label Encode | Many categories |

---

## Model Choice

### Part 1 — Emotional State: XGBoost Classifier

**Why XGBoost?**
- Works natively with mixed tabular + text features
- Handles residual NaN internally
- Fast on 1200-row dataset (seconds, not minutes)
- Gives interpretable feature importance
- Consistently outperforms Random Forest on tabular data

**Key settings:**
```python
n_estimators=500, max_depth=6, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.7
```

**Calibration:** Wrapped in `CalibratedClassifierCV(method='isotonic')` for reliable probability estimates used in confidence scoring.

**Evaluation:** 5-fold stratified cross-validation → real test performance estimate.

### Part 2 — Intensity: Ordinal Regression

**Why not regression?** Intensity 1–5 is ordinal, not continuous. Predicting 3 when truth is 4 should cost less than predicting 1 when truth is 4. MSE treats all errors equally — wrong for ordinal data.

**Method:** Threshold decomposition into 4 binary classifiers:
- Classifier k: predicts `P(intensity > k)` for k ∈ {1, 2, 3, 4}
- Class probs: `P(=k) = P(>k-1) − P(>k)`
- Prediction: `argmax(class probs) + 1`

### Part 3 — Decision Engine: Rule-Based

**Why rules instead of ML?**
- Dataset too small to learn decision mappings reliably
- Rules are interpretable — interviewers can ask "why this action?"
- Rules encode domain knowledge (chronobiology, stress science)
- Safe for mental wellness context — must be auditable

**Priority order:**
1. Safety signals (extreme stress / sleep deprivation)
2. Contradiction signals (high stress AND high energy)
3. Emotional state + intensity
4. Time of day alignment

---

## Text Features

```python
TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),    # unigrams + bigrams
    sublinear_tf=True,     # log-scale TF
    min_df=3,
    max_df=0.9,
)
```

Then reduced to 150 dims with `TruncatedSVD` (LSA).

**Key EDA finding:** All 6 emotional states cluster at mean stress ≈ 3.0, energy ≈ 3.0. Numeric metadata alone cannot separate them. Text is the primary signal (~70% of XGBoost feature importance).

---

## Ablation Study Results

| Configuration | CV F1 (weighted) |
|--------------|-----------------|
| Text only (SVD 150 dims) | run Cell 8 to see |
| Metadata only (24 features) | run Cell 8 to see |
| **Text + Metadata (full)** | **best** |

→ Text is dominant. Metadata adds incremental value on top.

---

## Uncertainty Modeling

```
confidence = prob_conf × 0.60 × text_factor × qual_factor
           + 0.40 × text_factor × qual_factor

where:
  prob_conf   = 1 − normalised_entropy(class probabilities)
  text_factor = 0.65 if text_quality_flag=1 else 1.0
  qual_factor = {clear:1.0, conflicted:0.85, vague:0.70}

uncertain_flag = 1  when confidence < 0.55
```

---

## Predictions CSV Format

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Row identifier |
| `predicted_state` | str | calm / restless / focused / neutral / mixed / overwhelmed |
| `predicted_intensity` | int | 1–5 |
| `confidence` | float | 0.0–1.0 |
| `uncertain_flag` | int | 1 = model is unsure |
| `what_to_do` | str | box_breathing / deep_work / journaling / etc. |
| `when_to_do` | str | now / within_15_min / later_today / tonight / tomorrow_morning |
| `supportive_message` | str | Human-like explanation of the recommendation |

---

## Bonus Features

| Feature | File | How to run |
|---------|------|-----------|
| Supportive message | `FE_ModelTraining.ipynb` | Auto-included in predictions.csv |
| Label noise handling | `ArvyaX_Bonus.ipynb` | Run Cell 1–5 |
| Lightweight SLM | `ArvyaX_Bonus.ipynb` | Run Cell 6–8 |
| FastAPI local API | `api.py` | `uvicorn api:app --reload --port 8000` |
| Streamlit UI demo | `app_ui.py` | `streamlit run app_ui.py` |

---

## Robustness Handling

| Input type | How handled |
|-----------|-------------|
| `"ok"`, `"fine"`, `"still off"` | `text_quality_flag=1` → `uncertain_flag=1`, confidence penalised |
| Missing `sleep_hours` | Filled with training median (6.0h) + `sleep_missing=1` flag |
| Missing `face_emotion_hint` | Filled with `'none'` + `face_missing=1` flag |
| Contradictory inputs (high stress + high energy) | `contradiction_flag=1` → decision engine routes to grounding |
| Unknown categorical values in test | Label encoder falls back to `'unknown'`/`'none'` class |
