"""
BeaGrace Health — Streamlit Web App
=====================================
Live demo for Technovation judges.
Deploys to Hugging Face Spaces or Streamlit Community Cloud.
"""

import json
import os
import re
import sys
import tempfile
import unicodedata
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BeaGrace Health",
    page_icon="🏥",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp { background: #f7f4ef; }

.hero {
    background: #1a3a2a;
    border-radius: 16px;
    padding: 40px 36px 32px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    margin: 0 0 6px 0;
    color: #e8f5ee;
    letter-spacing: -0.5px;
}
.hero p { font-size: 1rem; color: #9ec4ad; margin: 0; font-weight: 300; }
.hero .tag {
    display: inline-block;
    background: #2d5c3f;
    color: #7ecf97;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 20px;
    margin-bottom: 14px;
}

.pipeline {
    display: flex;
    gap: 0;
    margin-bottom: 28px;
    background: white;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e0dbd3;
}
.step {
    flex: 1;
    padding: 14px 10px;
    text-align: center;
    font-size: 0.72rem;
    font-weight: 500;
    color: #7a7060;
    border-right: 1px solid #e0dbd3;
}
.step:last-child { border-right: none; }
.step .num { display: block; font-size: 1.1rem; margin-bottom: 4px; }
.step.active { background: #1a3a2a; color: #9ec4ad; }
.step.active .num { color: #7ecf97; }

.card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    border: 1px solid #e0dbd3;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #9a8f7e;
    margin-bottom: 12px;
}

.finding {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px solid #f0ece6;
}
.finding:last-child { border-bottom: none; }
.finding-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    background: #e8f5ee;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}
.finding-icon.warn { background: #fff4e0; }
.finding-term { font-weight: 600; font-size: 0.95rem; color: #1a3a2a; }
.finding-meta { font-size: 0.8rem; color: #9a8f7e; margin-top: 2px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }
.badge-green { background: #e8f5ee; color: #1a7a3a; }
.badge-amber { background: #fff4e0; color: #b56a00; }

.transcript-box {
    background: #f7f4ef;
    border: 1px solid #e0dbd3;
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.95rem;
    color: #3a3028;
    margin-bottom: 4px;
    font-style: italic;
}

.handoff {
    background: #1a3a2a;
    border-radius: 12px;
    padding: 24px;
    margin-top: 8px;
}
.handoff-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: #7ecf97; margin-bottom: 10px;
}
.handoff-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem; color: #e8f5ee; line-height: 1.5;
}

.no-match {
    background: #fff8f0; border: 1px solid #f0d9b0;
    border-radius: 12px; padding: 20px;
    color: #8a6020; font-size: 0.9rem;
}
.info-box {
    background: #e8f5ee; border: 1px solid #b0d9be;
    border-radius: 10px; padding: 12px 16px;
    color: #1a4a2a; font-size: 0.85rem; margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Glossary ─────────────────────────────────────────────────────────────────
DEFAULT_GLOSSARY = [
    {"dialect_expression": "Ori mi n dun mi",               "english_medical_term": "Headache",                                           "literal_meaning": "my head hurts"},
    {"dialect_expression": "Ara mi n jo",                   "english_medical_term": "Burning sensation",                                  "literal_meaning": "my body is burning"},
    {"dialect_expression": "Ala mi jo",                     "english_medical_term": "Burning sensation",                                  "literal_meaning": "my body is burning"},
    {"dialect_expression": "Inu mi n ru",                   "english_medical_term": "Nausea / Gastrointestinal distress",                 "literal_meaning": "my stomach is rumbling"},
    {"dialect_expression": "Ori n fo mi",                   "english_medical_term": "Headache",                                           "literal_meaning": "my head is aching"},
    {"dialect_expression": "kpaja kpaja",                   "english_medical_term": "Numbness / Tingling sensation",                      "literal_meaning": "tingling sensation"},
    {"dialect_expression": "O dun mi",                      "english_medical_term": "Acute pain",                                         "literal_meaning": "it is paining me"},
    {"dialect_expression": "Ori mi ti gbono",               "english_medical_term": "Fever / Hyperthermia",                               "literal_meaning": "my head is hot"},
    {"dialect_expression": "Ara mi gbona",                  "english_medical_term": "Fever / Hyperthermia",                               "literal_meaning": "my body is hot"},
    {"dialect_expression": "Ara ro mi",                     "english_medical_term": "Generalized body aches",                             "literal_meaning": "my body is paining me"},
    {"dialect_expression": "Ara mi dun mi",                 "english_medical_term": "Angina pectoris / Chest pain",                       "literal_meaning": "i have sharp pain in my chest"},
    {"dialect_expression": "Aya me dunmi gon gidi",         "english_medical_term": "Acute / Severe chest pain",                          "literal_meaning": "the pain is getting stronger"},
    {"dialect_expression": "Oyji ko mi",                    "english_medical_term": "Dizziness / Vertigo",                                "literal_meaning": "my eyes are turning"},
    {"dialect_expression": "E mimi mi ko Kan le",           "english_medical_term": "Dyspnea / Shortness of breath",                      "literal_meaning": "i am having trouble breathing"},
    {"dialect_expression": "Es se mi ro mi",                "english_medical_term": "Arthralgia / Joint pain",                            "literal_meaning": "my joint feels swollen"},
    {"dialect_expression": "Ore mi mi ko le de",            "english_medical_term": "Asthenia / Generalized muscle weakness",             "literal_meaning": "i feel very weak and cannot stand"},
    {"dialect_expression": "Kpajakpaja rum mi le se",       "english_medical_term": "Peripheral paresthesia (Numbness in extremities)",   "literal_meaning": "tingling sensation in my feet"},
    {"dialect_expression": "Ono ofun mi nuy me",            "english_medical_term": "Pharyngitis / Sore throat",                          "literal_meaning": "my throat is scratching me"},
    {"dialect_expression": "Oruru mu mi, o da bi mun i ba","english_medical_term": "Diaphoresis and pyrexia (Sweating and feverishness)", "literal_meaning": "i have been sweating and feel feverish"},
    {"dialect_expression": "Eti mi kpar o woo",             "english_medical_term": "Tinnitus / Ringing ears",                            "literal_meaning": "my ear is ringing"},
    {"dialect_expression": "Ono funmi dunmi tim baru ko",   "english_medical_term": "Tussis-induced chest pain (Pain upon coughing)",     "literal_meaning": "it pains me when i cough"},
]

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_glossary():
    try:
        with open("glossary.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_GLOSSARY


def normalize(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", text.lower().strip())


def run_matcher(transcript, glossary, threshold=75):
    from rapidfuzz import fuzz, process
    index = {normalize(e["dialect_expression"]): e for e in glossary}
    norm_keys = list(index.keys())
    tokens = transcript.split()
    norm_tokens = normalize(transcript).split()
    if not tokens:
        return [], []

    candidates = []
    for size in range(1, 9):
        for i in range(len(norm_tokens) - size + 1):
            candidates.append((i, i + size, " ".join(norm_tokens[i:i+size])))

    scored = []
    for start, end, chunk in candidates:
        best_key, score, _ = process.extractOne(chunk, norm_keys, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            scored.append((score, start, end, best_key))

    scored.sort(key=lambda x: -x[0])
    used: set[int] = set()
    matched = []
    for score, start, end, best_key in scored:
        positions = set(range(start, end))
        if positions & used:
            continue
        used |= positions
        entry = index[best_key]
        matched.append({
            "patient_phrase": " ".join(tokens[start:end]),
            "clinical_term": entry["english_medical_term"],
            "literal_meaning": entry.get("literal_meaning", ""),
            "match_score": round(score, 1),
            "needs_review": score < 85,
            "token_span": (start, end),
        })

    matched.sort(key=lambda x: x["token_span"][0])
    seen: set[str] = set()
    summary = []
    for m in matched:
        if m["clinical_term"] not in seen:
            seen.add(m["clinical_term"])
            summary.append(m)
    return matched, summary


def get_unmatched(transcript, matched):
    tokens = transcript.split()
    used = set()
    for m in matched:
        used.update(range(*m["token_span"]))
    return " ".join(t for i, t in enumerate(tokens) if i not in used).strip()


def translate_google(text):
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="yo", target="en").translate(text)
    except Exception:
        return None


def transcribe_audio(audio_bytes: bytes) -> str | None:
    import requests
    import time

    api_key = "b2732dcd7b3d488780c686a21aaf136e"  # paste your key here

    # Step 1: Upload audio
    upload_response = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers={"authorization": api_key},
        data=audio_bytes,
    )
    upload_url = upload_response.json()["upload_url"]

    # Step 2: Request transcription in Yoruba
    transcript_response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers={"authorization": api_key, "content-type": "application/json"},
        json={"audio_url": upload_url, "language_code": "yo"},
    )
    transcript_id = transcript_response.json()["id"]

    # Step 3: Poll until complete
    while True:
        polling = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
            headers={"authorization": api_key},
        )
        status = polling.json()["status"]
        if status == "completed":
            return polling.json()["text"]
        elif status == "error":
            return None
        time.sleep(2)


# ── Load glossary ─────────────────────────────────────────────────────────────
glossary = load_glossary()

# ── UI ────────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <div class="tag">Technovation 2026 · BeaGrace Foundation</div>
    <h1>🏥 BeaGrace Health</h1>
    <p>Yoruba patient speech → clean clinical English for nurses.<br>
    Bridging the language gap in underserved communities.</p>
</div>
""", unsafe_allow_html=True)

# Pipeline steps
st.markdown("""
<div class="pipeline">
    <div class="step active"><span class="num">🎙️</span>Patient speaks Yoruba</div>
    <div class="step active"><span class="num">📝</span>Whisper transcribes</div>
    <div class="step active"><span class="num">📖</span>Glossary matches</div>
    <div class="step active"><span class="num">🌐</span>Google translates</div>
    <div class="step active"><span class="num">👩‍⚕️</span>Nurse reads English</div>
</div>
""", unsafe_allow_html=True)

# Info box
st.markdown("""
<div class="info-box">
    🎤 Upload an <strong>.mp3</strong> recording of a patient speaking in Yoruba.
    Whisper will transcribe it, the glossary will identify medical terms,
    and Google Translate will handle anything else.
</div>
""", unsafe_allow_html=True)

# Audio uploader
audio_file = st.file_uploader(
    "Upload patient audio (.mp3)",
    type=["mp3", "wav", "m4a"],
    label_visibility="collapsed",
)

use_google = st.checkbox("Translate unmatched text with Google Translate", value=True)

run = st.button("▶  Run Pipeline", type="primary", use_container_width=True)

# ── Pipeline execution ────────────────────────────────────────────────────────
if run:
    if not audio_file:
        st.warning("Please upload an audio file first.")
    else:
        # Step 1 & 2: Transcribe
        with st.spinner("Transcribing audio with Whisper..."):
            transcript = transcribe_audio(audio_file.read())

        if not transcript:
            st.error("Transcription failed. Please try again.")
        else:
            # Show transcript
            st.markdown('<div class="card-title" style="margin-top:8px">Yoruba transcript</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="transcript-box">"{transcript}"</div>', unsafe_allow_html=True)

            # Step 3: Glossary match
            matched, summary = run_matcher(transcript, glossary)
            unmatched = get_unmatched(transcript, matched)

            # Step 4: Translate unmatched
            translated = None
            if unmatched and use_google:
                with st.spinner("Translating unmatched text..."):
                    translated = translate_google(unmatched)

            # Step 5: Glossary findings
            st.markdown("---")
            if summary:
                st.markdown('<div class="card-title">Glossary findings</div>', unsafe_allow_html=True)
                for f in summary:
                    icon = "⚠️" if f["needs_review"] else "✅"
                    icon_class = "warn" if f["needs_review"] else ""
                    badge_class = "badge-amber" if f["needs_review"] else "badge-green"
                    badge_text = "Verify" if f["needs_review"] else f"{f['match_score']}% match"
                    st.markdown(f"""
                    <div class="finding">
                        <div class="finding-icon {icon_class}">{icon}</div>
                        <div>
                            <div class="finding-term">{f['clinical_term']}</div>
                            <div class="finding-meta">
                                Patient said: <em>"{f['patient_phrase']}"</em> &nbsp;·&nbsp;
                                Meaning: {f['literal_meaning']} &nbsp;·&nbsp;
                                <span class="badge {badge_class}">{badge_text}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="no-match">
                    ⚠️ No glossary matches found — full transcript sent to Google Translate.
                </div>
                """, unsafe_allow_html=True)

            # Unmatched translation
            if unmatched:
                st.markdown('<div class="card-title" style="margin-top:16px">Unmatched text</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.markdown(f"**Yoruba:** {unmatched}")
                if translated:
                    col2.markdown(f"**English:** {translated}")
                elif use_google:
                    col2.markdown("⚠️ Translation unavailable — check internet connection")
                else:
                    col2.markdown("*Google Translate disabled*")

            # Nurse handoff
            terms = [f["clinical_term"] for f in summary]
            if translated and translated != unmatched:
                handoff = f"Patient presents with: {', '.join(terms)}. Also reports: {translated}." if terms else translated
            elif terms:
                handoff = f"Patient presents with: {', '.join(terms)}."
            else:
                handoff = translated or "Unable to process — manual clinician review required."

            st.markdown(f"""
            <div class="handoff">
                <div class="handoff-label">📋 Nurse Handoff</div>
                <div class="handoff-text">{handoff}</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#9a8f7e; font-size:0.8rem;'>"
    "BeaGrace Foundation · Built for Technovation 2026 · "
    "Mentored by Shalini · AIVA Programme</p>",
    unsafe_allow_html=True,
)
