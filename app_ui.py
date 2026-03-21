"""
app_ui.py — ArvyaX Streamlit UI Demo
Run: streamlit run app_ui.py
No external APIs — calls the local FastAPI or runs inference directly.
"""

import streamlit as st
import numpy as np
import pandas as pd
import re, os, sys

st.set_page_config(
    page_title="🌿 ArvyaX — Emotion Guidance",
    page_icon="🌿",
    layout="wide",
)

# ── Load models ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_models():
    import joblib
    BASE = "outputs/artifacts"
    return {
        "tfidf"       : joblib.load(f"{BASE}/tfidf.pkl"),
        "svd"         : joblib.load(f"{BASE}/svd.pkl"),
        "le_state"    : joblib.load(f"{BASE}/le_state.pkl"),
        "le_face"     : joblib.load(f"{BASE}/le_face.pkl"),
        "le_mood"     : joblib.load(f"{BASE}/le_mood.pkl"),
        "model_state" : joblib.load(f"{BASE}/model_state.pkl"),
        "model_intens": joblib.load(f"{BASE}/model_intensity.pkl"),
        "meta_cols"   : joblib.load(f"{BASE}/meta_cols.pkl"),
    }

# ── Constants ────────────────────────────────────────────────────────
TIME_MAP  = {"early_morning":0,"morning":1,"afternoon":2,"evening":3,"night":4}
QUAL_MAP  = {"vague":0,"conflicted":1,"clear":2}
THRESHOLDS= [1,2,3,4]
STATE_EMOJI = {
    "calm":"😌","restless":"😤","overwhelmed":"😰",
    "focused":"🎯","neutral":"😐","mixed":"😕"
}
ACTION_EMOJI = {
    "box_breathing":"🫁","journaling":"📓","grounding":"🌱",
    "deep_work":"💻","yoga":"🧘","light_planning":"📋",
    "rest":"🛋️","movement":"🚶","pause":"⏸️",
    "sleep_prep":"🌙","gratitude":"🙏","sound_therapy":"🎵",
}
ACTION_DESC = {
    "box_breathing"  :"4 counts in → hold → out → hold. Calms nervous system fast.",
    "journaling"     :"Free-write 10 min — no structure needed.",
    "grounding"      :"5-4-3-2-1: name 5 things you see, 4 touch, 3 hear.",
    "deep_work"      :"25-min Pomodoro focus block.",
    "yoga"           :"10 min gentle yoga or stretching.",
    "light_planning" :"Write 3 intentions for today — no more.",
    "rest"           :"No agenda, no screens. Just recharge.",
    "movement"       :"Short walk or body shake for 5 min.",
    "pause"          :"Sit quietly 5 min. Just observe.",
    "sleep_prep"     :"Dim lights, step away from screens.",
    "gratitude"      :"Write 3 things you appreciated today.",
    "sound_therapy"  :"Soft music or nature sounds.",
}

def clean_text(text):
    if not isinstance(text,str) or not text.strip():
        return "neutral experience"
    text = re.sub(r"[^a-z\s]"," ",text.lower())
    return re.sub(r"\s+"," ",text).strip() or "neutral experience"

def compute_confidence(proba, tq_flag, rq):
    p = np.array(proba,dtype=np.float64)+1e-10; p/=p.sum()
    H = -np.sum(p*np.log(p))
    pc= 1.0-H/np.log(len(p))
    tf= 0.65 if tq_flag else 1.0
    qf= {"clear":1.0,"conflicted":0.85,"vague":0.70}.get(str(rq).lower(),0.80)
    return float(np.clip(pc*0.60*tf*qf+0.40*tf*qf,0,1))

def predict_intensity(X, models):
    ex= np.column_stack([models[k].predict_proba(X)[:,1] for k in THRESHOLDS])
    p = np.zeros((1,5))
    p[0,0]=1-ex[0,0]
    for j in range(1,4): p[0,j]=ex[0,j-1]-ex[0,j]
    p[0,4]=ex[0,3]; p=np.clip(p,0,1)
    return int(p.argmax()+1)

def decide(state,intensity,stress,energy,tod,sleep):
    s=str(state).lower(); t=str(tod).lower()
    if stress>=5 and intensity>=4: return "box_breathing","now"
    if sleep<5:
        return ("sleep_prep","tonight") if t in ("evening","night") else ("rest","within_15_min")
    if stress>=4 and energy>=4: return "grounding","now"
    if s in ("overwhelmed","anxious","stressed"):
        if intensity>=4: return "box_breathing","now"
        if intensity==3: return "grounding","within_15_min"
        return "journaling","later_today"
    if s=="restless": return ("movement","now") if energy>=3 else ("grounding","within_15_min")
    if s=="calm":
        if t in ("morning","afternoon") and energy>=3: return "deep_work","within_15_min"
        if t=="evening": return "gratitude","later_today"
        return "rest","tonight"
    if s=="focused": return "deep_work","now"
    if s=="mixed":   return "journaling","within_15_min"
    if s=="neutral":
        if t=="morning":   return "light_planning","within_15_min"
        if t=="afternoon": return "deep_work","later_today"
        if t=="evening":   return "yoga","later_today"
        return "rest","tonight"
    return "pause","within_15_min"

# ── UI Layout ────────────────────────────────────────────────────────
st.title("🌿 ArvyaX — Emotion Guidance System")
st.caption("Understand your emotional state → Receive a meaningful action. Runs 100% locally.")
st.divider()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📝 Your Reflection")
    journal_text = st.text_area(
        "Write how you're feeling after your session...",
        placeholder="E.g. Felt a bit scattered but the ocean sounds helped. Mind still jumping around.",
        height=140,
    )
    st.caption("Even a few words work. Short entries will be flagged as uncertain.")

with col_right:
    st.subheader("📊 Context Signals")
    sleep_hours   = st.slider("Sleep hours last night", 0.0, 10.0, 7.0, 0.5)
    stress_level  = st.slider("Stress level (1=low, 5=high)", 1, 5, 3)
    energy_level  = st.slider("Energy level (1=low, 5=high)", 1, 5, 3)
    time_of_day   = st.selectbox("Time of day", ["morning","afternoon","evening","night","early_morning"])
    ambience_type = st.selectbox("Session ambience", ["forest","ocean","rain","mountain","cafe"])
    refl_quality  = st.selectbox("Reflection quality", ["clear","conflicted","vague"])

st.divider()
run_btn = st.button("✨  Get Guidance", type="primary", use_container_width=True)

if run_btn:
    if not journal_text.strip():
        st.warning("Please write at least a few words in your reflection.")
    else:
        try:
            m = load_models()

            # Build features
            tc  = clean_text(journal_text)
            tl  = len(tc.split())
            tqf = int(tl <= 3)
            X_text = m["svd"].transform(m["tfidf"].transform([tc]))

            fe_val  = "none"
            face_enc= m["le_face"].transform([fe_val])[0] if fe_val in m["le_face"].classes_ \
                      else m["le_face"].transform(["none"])[0]
            me_val  = "unknown"
            mood_enc= m["le_mood"].transform([me_val])[0] if me_val in m["le_mood"].classes_ \
                      else m["le_mood"].transform(["unknown"])[0]

            meta_d = {
                "sleep_hours":sleep_hours,"energy_level":energy_level,"stress_level":stress_level,
                "duration_min":15.0,"sleep_deficit":8-sleep_hours,
                "low_sleep_flag":int(sleep_hours<5),
                "stress_energy_ratio":stress_level/(energy_level+0.01),
                "stress_energy_product":stress_level*energy_level,
                "net_wellbeing":energy_level-stress_level,
                "contradiction_flag":int(stress_level>=4 and energy_level>=4),
                "productivity_proxy":15*energy_level/(stress_level+0.01),
                "text_len":tl,"text_quality_flag":tqf,
                "time_enc":TIME_MAP.get(time_of_day,1),
                "quality_enc":QUAL_MAP.get(refl_quality,2),
                "face_enc":face_enc,"mood_enc":mood_enc,
                "face_missing":1,"mood_missing":1,"sleep_missing":0,
            }
            for c in m["meta_cols"]:
                if c.startswith("amb_"):
                    meta_d[c] = int(c==f"amb_{ambience_type}")

            X_meta = np.array([meta_d.get(c,0) for c in m["meta_cols"]],
                               dtype=np.float32).reshape(1,-1)
            X = np.hstack([X_text, X_meta])

            # Predict
            sp    = m["model_state"].predict_proba(X)[0]
            state = m["le_state"].inverse_transform([sp.argmax()])[0]
            intens= predict_intensity(X, m["model_intens"])
            conf  = compute_confidence(sp, tqf, refl_quality)
            flag  = int(conf < 0.55)
            what, when = decide(state, intens, stress_level, energy_level, time_of_day, sleep_hours)

            # Message
            adv  = {1:"mildly",2:"somewhat",3:"moderately",4:"quite",5:"very"}.get(intens,"moderately")
            hedge= "Hard to read exactly, but " if conf < 0.5 else ""
            msg  = f"{hedge}You seem {adv} {state} right now. {ACTION_DESC.get(what,'Take a moment for yourself.')}"

            # ── Display results ─────────────────────────────────────
            st.divider()
            r1, r2, r3 = st.columns(3)

            emoji = STATE_EMOJI.get(state,"🌀")
            r1.metric(f"{emoji} Emotional State", state.title(), f"Intensity {intens}/5")

            bar_color = "green" if conf > 0.65 else "orange" if conf > 0.5 else "red"
            r2.metric("🎯 Confidence", f"{conf:.0%}",
                      "✅ Reliable" if conf>0.55 else "⚠️ Uncertain")

            ae = ACTION_EMOJI.get(what,"▶️")
            when_display = when.replace("_"," ").title()
            r3.metric(f"{ae} Action", what.replace("_"," ").title(), when_display)

            st.divider()
            if flag:
                st.warning("⚠️ Low confidence — signals are ambiguous. "
                           "This guidance is a suggestion, not a diagnosis.")

            st.info(f"**💬 Your Guidance**\n\n{msg}")

            with st.expander("🔍 Full details"):
                st.json({
                    "predicted_state"     : state,
                    "predicted_intensity" : intens,
                    "confidence"          : round(conf,4),
                    "uncertain_flag"      : flag,
                    "what_to_do"          : what,
                    "when_to_do"          : when,
                    "action_description"  : ACTION_DESC.get(what,""),
                    "text_quality_flag"   : tqf,
                    "text_length_tokens"  : tl,
                })

        except FileNotFoundError:
            st.error("❌ Model files not found. Run FE_ModelTraining.ipynb first to generate outputs/artifacts/")
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.divider()
st.caption("🔒 ArvyaX · All processing is local — no data leaves your device.")
