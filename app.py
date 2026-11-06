"""
BeaGrace Health — Gradio Web App
==================================
Live demo for Technovation judges.
Deploys to Hugging Face Spaces (Gradio SDK).
"""

import json
import re
import os
import sys
import tempfile
import unicodedata

import gradio as gr

# ── Glossary ──────────────────────────────────────────────────────────────────
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

def load_glossary():
    try:
        with open("glossary.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_GLOSSARY

glossary = load_glossary()

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def transcribe_audio(audio_path: str) -> str | None:
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "whisper_server.py", audio_path],
            capture_output=True,
            text=True,
            timeout=180,
        )
        stdout = result.stdout.strip()
        if not stdout:
            return None
        data = json.loads(stdout)
        if "error" in data:
            return None
        return data["transcript"]
    except Exception:
        return None

# ── Main pipeline function ────────────────────────────────────────────────────
def run_pipeline(audio_file, use_google):
    if audio_file is None:
        return (
            "❌ Please upload an audio file.",
            "",
            "",
            "",
        )

    # Step 1 & 2: Transcribe
    transcript = transcribe_audio(audio_file)
    if not transcript:
        return (
            "❌ Transcription failed. Please try again.",
            "",
            "",
            "",
        )

    # Step 3: Glossary match
    matched, summary = run_matcher(transcript, glossary)
    unmatched = get_unmatched(transcript, matched)

    # Step 4: Translate unmatched
    translated = None
    if unmatched and use_google:
        translated = translate_google(unmatched)

    # Build findings text
    findings_text = ""
    if summary:
        for f in summary:
            flag = " ⚠️ verify" if f["needs_review"] else " ✅"
            findings_text += f"**{f['clinical_term']}**{flag}\n"
            findings_text += f"- Patient said: *\"{f['patient_phrase']}\"*\n"
            findings_text += f"- Meaning: {f['literal_meaning']}\n"
            findings_text += f"- Match score: {f['match_score']}%\n\n"
    else:
        findings_text = "⚠️ No glossary matches found."

    # Build unmatched text
    unmatched_text = ""
    if unmatched:
        unmatched_text = f"**Yoruba:** {unmatched}\n\n"
        if translated and translated != unmatched:
            unmatched_text += f"**English:** {translated}"
        elif use_google:
            unmatched_text += "⚠️ Translation unavailable"

    # Build handoff
    terms = [f["clinical_term"] for f in summary]
    if translated and translated != unmatched:
        handoff = f"Patient presents with: {', '.join(terms)}. Also reports: {translated}." if terms else translated
    elif terms:
        handoff = f"Patient presents with: {', '.join(terms)}."
    else:
        handoff = translated or "Unable to process — manual clinician review required."

    return transcript, findings_text, unmatched_text, f"📋 {handoff}"

# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
.gradio-container { font-family: 'DM Sans', sans-serif; max-width: 800px; margin: 0 auto; }
.hero { background: #1a3a2a; border-radius: 12px; padding: 24px; margin-bottom: 20px; color: white; }
.handoff-box { background: #1a3a2a !important; border-radius: 12px !important; color: #e8f5ee !important; font-size: 1.1rem !important; }
"""

with gr.Blocks(css=css, title="BeaGrace Health") as demo:

    gr.HTML("""
    <div class="hero">
        <div style="display:inline-block; background:#2d5c3f; color:#7ecf97; font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; padding:3px 10px; border-radius:20px; margin-bottom:12px;">
            Technovation 2026 · BeaGrace Foundation
        </div>
        <h1 style="color:#e8f5ee; margin:0 0 6px; font-size:2rem;">🏥 BeaGrace Health</h1>
        <p style="color:#9ec4ad; margin:0; font-size:0.95rem;">
            Yoruba patient speech → clean clinical English for nurses.<br>
            Bridging the language gap in underserved communities.
        </p>
    </div>
    <div style="display:flex; background:white; border-radius:10px; border:1px solid #e0dbd3; overflow:hidden; margin-bottom:20px; text-align:center; font-size:12px; font-weight:500;">
        <div style="flex:1; padding:12px; background:#1a3a2a; color:#9ec4ad;">🎙️<br>Patient speaks Yoruba</div>
        <div style="flex:1; padding:12px; background:#1a3a2a; color:#9ec4ad; border-left:1px solid #2d5c3f;">📝<br>Whisper transcribes</div>
        <div style="flex:1; padding:12px; background:#1a3a2a; color:#9ec4ad; border-left:1px solid #2d5c3f;">📖<br>Glossary matches</div>
        <div style="flex:1; padding:12px; background:#1a3a2a; color:#9ec4ad; border-left:1px solid #2d5c3f;">🌐<br>Google translates</div>
        <div style="flex:1; padding:12px; background:#1a3a2a; color:#9ec4ad; border-left:1px solid #2d5c3f;">👩‍⚕️<br>Nurse reads English</div>
    </div>
    """)

    with gr.Row():
        audio_input = gr.Audio(
            label="Upload patient audio (.mp3)",
            type="filepath",
            sources=["upload"],
        )

    use_google = gr.Checkbox(
        label="Translate unmatched text with Google Translate",
        value=True,
    )

    run_btn = gr.Button("▶ Run Pipeline", variant="primary", size="lg")

    transcript_out = gr.Textbox(label="Yoruba Transcript", interactive=False)
    findings_out   = gr.Markdown(label="Glossary Findings")
    unmatched_out  = gr.Markdown(label="Unmatched Text")
    handoff_out    = gr.Textbox(
        label="📋 Nurse Handoff",
        interactive=False,
        elem_classes=["handoff-box"],
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[audio_input, use_google],
        outputs=[transcript_out, findings_out, unmatched_out, handoff_out],
    )

    gr.HTML("""
    <div style="text-align:center; color:#9a8f7e; font-size:11px; margin-top:20px; padding-top:16px; border-top:1px solid #e0dbd3;">
        BeaGrace Foundation · Built for Technovation 2026 · Mentored by Shalini · AIVA Programme
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
