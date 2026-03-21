"""
api.py — ArvyaX FastAPI Local Server
Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
No external LLM APIs used — runs 100% locally.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import joblib, re, os

# ── Load all artifacts once at startup ──────────────────────────────
BASE = "outputs/artifacts"

try:
    tfidf       = joblib.load(f"{BASE}/tfidf.pkl")
    svd         = joblib.load(f"{BASE}/svd.pkl")
    le_state    = joblib.load(f"{BASE}/le_state.pkl")
    le_face     = joblib.load(f"{BASE}/le_face.pkl")
    le_mood     = joblib.load(f"{BASE}/le_mood.pkl")
    model_state = joblib.load(f"{BASE}/model_state.pkl")
    model_intens= joblib.load(f"{BASE}/model_intensity.pkl")
    meta_cols   = joblib.load(f"{BASE}/meta_cols.pkl")
    LOADED = True
except Exception as e:
    LOADED = False
    LOAD_ERROR = str(e)

app = FastAPI(
    title="🌿 ArvyaX Emotion Guidance API",
    description="Predict emotional state and receive a personalised action recommendation. Runs 100% locally.",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Constants ────────────────────────────────────────────────────────
TIME_MAP  = {"early_morning":0,"morning":1,"afternoon":2,"evening":3,"night":4}
QUAL_MAP  = {"vague":0,"conflicted":1,"clear":2}
THRESHOLDS= [1,2,3,4]

ACTION_DESC = {
    "box_breathing"  : "Try box breathing: 4 counts in, hold, out, hold. Calms the nervous system fast.",
    "journaling"     : "Free-write for 10 minutes — no structure. Get thoughts out of your head.",
    "grounding"      : "5-4-3-2-1 grounding: name 5 things you see, 4 you touch, 3 you hear.",
    "deep_work"      : "Channel this energy into a focused 25-minute Pomodoro work block.",
    "yoga"           : "10 minutes of gentle yoga or stretching to reconnect body and mind.",
    "light_planning" : "Write 3 intentions for the day — no more, no less. 5 minutes max.",
    "rest"           : "Give yourself permission to rest. No agenda, no screens.",
    "movement"       : "Take a short walk or shake out your body for 5 minutes.",
    "pause"          : "Sit quietly for 5 minutes. Just observe — don't try to fix anything.",
    "sleep_prep"     : "Dim the lights, step away from screens, and begin winding down.",
    "gratitude"      : "Write 3 things you genuinely appreciated about today.",
    "sound_therapy"  : "Put on soft music or nature sounds and simply listen.",
}
TIMING_PHRASE = {
    "now"             : "right now, before anything else",
    "within_15_min"   : "in the next 15 minutes",
    "later_today"     : "later today when things settle",
    "tonight"         : "tonight as part of your wind-down",
    "tomorrow_morning": "tomorrow morning as a fresh start",
}

# ── Helper functions ─────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return "neutral experience"
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "neutral experience"

def compute_confidence(proba, text_quality_flag, reflection_quality):
    proba     = np.array(proba, dtype=np.float64)
    proba     = proba + 1e-10
    proba     = proba / proba.sum()
    n         = len(proba)
    H         = -np.sum(proba * np.log(proba))
    prob_conf = 1.0 - H / np.log(n)
    text_f    = 0.65 if int(text_quality_flag) == 1 else 1.0
    qual_f    = {"clear":1.0,"conflicted":0.85,"vague":0.70}.get(
                  str(reflection_quality).lower(), 0.80)
    conf = prob_conf * 0.60 * text_f * qual_f + 0.40 * text_f * qual_f
    return float(np.clip(conf, 0, 1))

def predict_intensity(X, models):
    exceed = np.column_stack([models[k].predict_proba(X)[:,1] for k in THRESHOLDS])
    probs  = np.zeros((X.shape[0], 5))
    probs[:,0] = 1.0 - exceed[:,0]
    for j in range(1,4):
        probs[:,j] = exceed[:,j-1] - exceed[:,j]
    probs[:,4] = exceed[:,3]
    probs = np.clip(probs, 0, 1)
    return int(probs.argmax(axis=1)[0] + 1), probs[0]

def decide(state, intensity, stress, energy, time_of_day, sleep_hours):
    state = str(state).lower(); tod = str(time_of_day).lower()
    if stress >= 5 and intensity >= 4: return "box_breathing", "now"
    if sleep_hours < 5.0:
        return ("sleep_prep","tonight") if tod in ("evening","night") else ("rest","within_15_min")
    if stress >= 4 and energy >= 4:    return "grounding", "now"
    if state in ("overwhelmed","anxious","stressed"):
        if intensity >= 4: return "box_breathing","now"
        if intensity == 3: return "grounding","within_15_min"
        return "journaling","later_today"
    if state == "restless":
        return ("movement","now") if energy >= 3 else ("grounding","within_15_min")
    if state == "calm":
        if tod in ("morning","afternoon") and energy >= 3: return "deep_work","within_15_min"
        if tod == "evening": return "gratitude","later_today"
        return "rest","tonight"
    if state == "focused":  return "deep_work","now"
    if state == "mixed":    return "journaling","within_15_min"
    if state == "neutral":
        if tod == "morning":   return "light_planning","within_15_min"
        if tod == "afternoon": return "deep_work","later_today"
        if tod == "evening":   return "yoga","later_today"
        return "rest","tonight"
    return "pause","within_15_min"

def build_features(req):
    """Convert API request → feature vector for models."""
    text_clean = clean_text(req.journal_text)
    text_len   = len(text_clean.split())
    tq_flag    = int(text_len <= 3)

    X_text = svd.transform(tfidf.transform([text_clean]))

    face_val = req.face_emotion_hint or "none"
    face_enc = le_face.transform([face_val])[0] if face_val in le_face.classes_ \
               else le_face.transform(["none"])[0]
    mood_val = req.previous_day_mood or "unknown"
    mood_enc = le_mood.transform([mood_val])[0] if mood_val in le_mood.classes_ \
               else le_mood.transform(["unknown"])[0]

    sleep    = float(req.sleep_hours or 6.0)
    energy   = float(req.energy_level or 3)
    stress   = float(req.stress_level or 3)
    duration = float(req.duration_min or 15)

    meta = {
        "sleep_hours"          : sleep,
        "energy_level"         : energy,
        "stress_level"         : stress,
        "duration_min"         : duration,
        "sleep_deficit"        : 8.0 - sleep,
        "low_sleep_flag"       : int(sleep < 5),
        "stress_energy_ratio"  : stress / (energy + 0.01),
        "stress_energy_product": stress * energy,
        "net_wellbeing"        : energy - stress,
        "contradiction_flag"   : int(stress >= 4 and energy >= 4),
        "productivity_proxy"   : duration * energy / (stress + 0.01),
        "text_len"             : text_len,
        "text_quality_flag"    : tq_flag,
        "time_enc"             : TIME_MAP.get(req.time_of_day or "morning", 1),
        "quality_enc"          : QUAL_MAP.get(req.reflection_quality or "clear", 2),
        "face_enc"             : face_enc,
        "mood_enc"             : mood_enc,
        "face_missing"         : int(req.face_emotion_hint is None),
        "mood_missing"         : int(req.previous_day_mood is None),
        "sleep_missing"        : int(req.sleep_hours is None),
    }
    amb_cols = [c for c in meta_cols if c.startswith("amb_")]
    for c in amb_cols:
        meta[c] = int(c == f"amb_{req.ambience_type}")

    X_meta = np.array([meta.get(c, 0) for c in meta_cols],
                       dtype=np.float32).reshape(1, -1)
    return np.hstack([X_text, X_meta]), tq_flag, text_clean

# ── Request / Response schemas ───────────────────────────────────────
class PredictRequest(BaseModel):
    journal_text      : str   = Field(..., example="Felt a bit scattered but pushed through.")
    ambience_type     : Optional[str]  = Field("forest", example="ocean")
    duration_min      : Optional[float]= Field(15)
    sleep_hours       : Optional[float]= Field(7.0)
    energy_level      : Optional[float]= Field(3)
    stress_level      : Optional[float]= Field(3)
    time_of_day       : Optional[str]  = Field("morning")
    previous_day_mood : Optional[str]  = Field(None)
    face_emotion_hint : Optional[str]  = Field(None)
    reflection_quality: Optional[str]  = Field("clear")

class PredictResponse(BaseModel):
    predicted_state    : str
    predicted_intensity: int
    confidence         : float
    uncertain_flag     : int
    what_to_do         : str
    when_to_do         : str
    action_description : str
    supportive_message : str

# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    if not LOADED:
        return {"status": "error", "message": LOAD_ERROR}
    return {"status": "ok", "model": "ArvyaX v1.0", "classes": le_state.classes_.tolist()}

@app.get("/")
def root():
    return {"message": "🌿 ArvyaX API running. Visit /docs for interactive API docs."}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not LOADED:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {LOAD_ERROR}")
    try:
        X, tq_flag, text_clean = build_features(req)

        # State prediction
        state_proba   = model_state.predict_proba(X)[0]
        pred_state    = le_state.inverse_transform([state_proba.argmax()])[0]

        # Intensity prediction
        pred_intensity, _ = predict_intensity(X, model_intens)

        # Confidence
        conf = compute_confidence(state_proba, tq_flag,
                                  req.reflection_quality or "clear")
        flag = int(conf < 0.55)

        # Decision
        what, when = decide(pred_state, pred_intensity,
                            float(req.stress_level or 3),
                            float(req.energy_level or 3),
                            req.time_of_day or "morning",
                            float(req.sleep_hours or 6.0))

        # Message
        adv  = {1:"mildly",2:"somewhat",3:"moderately",4:"quite",5:"very"}.get(pred_intensity,"moderately")
        hedge= "Hard to read exactly, but " if conf < 0.5 else ""
        msg  = (f"{hedge}You seem {adv} {pred_state} right now. "
                f"I suggest you {ACTION_DESC.get(what,'take a moment')} — "
                f"{TIMING_PHRASE.get(when,'when you can')}.")

        return PredictResponse(
            predicted_state    = pred_state,
            predicted_intensity= pred_intensity,
            confidence         = round(conf, 4),
            uncertain_flag     = flag,
            what_to_do         = what,
            when_to_do         = when,
            action_description = ACTION_DESC.get(what, ""),
            supportive_message = msg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
def get_classes():
    return {"emotional_states": le_state.classes_.tolist(),
            "actions": list(ACTION_DESC.keys())}
