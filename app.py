import gradio as gr
import json
import re
import os
import time
import unicodedata
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from clinguard import Lexicon, normalize
import assemblyai as aai
import librosa
import soundfile as sf

from usage_log import log_usage, log_feedback
from safety import check_danger, danger_banner
from learning import submit_suggestion, pending_suggestions, review, learned_entries
from impact import compute_stats, impact_html, TEST_DEPARTMENT

GLOSSARY_FILE = Path('glossary.json')

# Languages the app supports. Add a language by adding an entry to languages.json
# plus its glossary file (see README: "Adding a new language").
try:
    LANGUAGES = json.loads(Path('languages.json').read_text(encoding='utf-8'))
except Exception:
    LANGUAGES = {'yo': {'label': 'Yoruba (Ekiti)', 'name': 'Yoruba', 'dialect_note': 'Ekiti dialect',
                        'speech_code': 'yo', 'glossary_file': 'glossary.json',
                        'danger_keywords': 'yoruba_keywords'}}
DEFAULT_LANG = next(iter(LANGUAGES))


def lang_cfg(lang: str) -> dict:
    return LANGUAGES.get(lang) or LANGUAGES[DEFAULT_LANG]
ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')


def load_glossary(lang: str = DEFAULT_LANG) -> list:
    entries = []
    path = Path(lang_cfg(lang).get('glossary_file', 'glossary.json'))
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    # plus phrases approved by a reviewer through the learning loop
    return entries + learned_entries(lang)


def clean_text(text: str) -> str:
    """Lower-case, no tone marks or punctuation (from the ClinGuard package)."""
    return normalize(text)


def fuzzy_match_glossary(text: str, threshold: int = 75, lang: str = DEFAULT_LANG) -> list:
    """Glossary matching powered by ClinGuard (pip install clinguard): tone-mark
    normalisation, tolerant length-aware fuzzy matching, specific phrases first."""
    lexicon = Lexicon.from_records(load_glossary(lang), language=lang, threshold=threshold / 100.0)
    return [{
        'ratio': m.score,
        'dialect': m.entry.phrase,
        'english': m.entry.clinical_term,
        'meaning': m.entry.meaning,
    } for m in lexicon.matches(text)]


def preprocess_audio(audio_path: str) -> str:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp * 0.95
    cleaned_path = str(Path(audio_path).with_name(Path(audio_path).stem + '_cleaned.wav'))
    sf.write(cleaned_path, y, sr)
    return cleaned_path


def transcribe_audio(audio_path: str, lang: str = DEFAULT_LANG) -> str:
    cleaned_path = preprocess_audio(audio_path)
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    config = aai.TranscriptionConfig(language_code=lang_cfg(lang).get('speech_code', 'yo'))
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(cleaned_path)
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f'Transcription failed: {transcript.error}')
    return transcript.text.strip()


def call_groq(prompt: str, max_tokens: int = 1000) -> str:
    """Send one prompt to Groq, retrying up to 3 times."""
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': 'openai/gpt-oss-120b',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,  # gpt-oss reasons before answering, so leave room
    }
    last_error = None
    for attempt in range(3):
        try:
            print(f'  [Groq] attempt {attempt + 1}')
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            content = (response.json()['choices'][0]['message'].get('content') or '').strip()
            if content:
                return content
            last_error = 'empty response'
        except Exception as e:
            last_error = e
            print(f'  [Groq] attempt {attempt + 1} failed: {e}')
        time.sleep(3)
    raise RuntimeError(f'AI service unavailable after 3 tries ({last_error}).')


def translate_to_english(text: str, lang: str = DEFAULT_LANG):
    """Returns (english, literal_meaning, glossary_confidence).
    Uses the glossary first; if nothing matches, asks the LLM to translate."""
    cfg = lang_cfg(lang)
    matches = fuzzy_match_glossary(text, lang=lang)
    if matches:
        match = matches[0]
        return match['english'], match['meaning'], match['ratio']
    prompt = f"""Translate this {cfg['name']} speech ({cfg.get('dialect_note', 'local dialect')}, transcribed by speech recognition, may contain errors) into plain English.
Reply with the English translation only, one short sentence. If it cannot be understood, reply exactly: UNCLEAR

{cfg['name']}: {text}"""
    translated = call_groq(prompt, max_tokens=400)
    if translated.strip().upper().startswith('UNCLEAR'):
        return 'Unclear - please ask the patient to repeat', '', 0.0
    return translated, '', 0.0


EXAMPLES_FILE = Path('clinical_examples.json')


def load_prompt_examples() -> str:
    """Patient statements with clinical terms supplied by the BeaGrace team.
    Only entries marked use_in_prompt are shown to the model; the rest are kept
    for testing (see eval_notes.py)."""
    if not EXAMPLES_FILE.exists():
        return ''
    with open(EXAMPLES_FILE, 'r', encoding='utf-8') as f:
        examples = [e for e in json.load(f) if e.get('use_in_prompt')]
    return '\n'.join(f'- "{e["patient_says"]}" -> {e["clinical_term"]}' for e in examples)


def generate_clinical_note(english_symptom: str, literal_meaning: str, lang: str = DEFAULT_LANG) -> str:
    cfg = lang_cfg(lang)
    spoken = f"{cfg.get('dialect_note', '')} {cfg['name']}".replace('dialect ', '').strip()
    context = f'Literal meaning from patient in {spoken}: "{literal_meaning}"' if literal_meaning else ''
    examples = load_prompt_examples()
    examples_block = f"""
Examples of how patients describe symptoms and the clinical term a nurse would record:
{examples}
""" if examples else ''
    prompt = f"""You are a clinical documentation assistant supporting nurses in a rural Nigerian healthcare facility.

A patient has reported the following symptom (translated from {spoken}):
"{english_symptom}"
{context}
{examples_block}
Important: interpret the symptom based strictly on the literal meaning provided. Do not assume urinary symptoms unless explicitly stated.
Describe possible conditions as "possible" or "suggestive of"; do not state a diagnosis.
Set Urgency to High for anything that could be life-threatening for the patient or, in pregnancy, the baby (for example: heavy bleeding, reduced or absent fetal movement, difficulty breathing, signs of stroke, seizures, thoughts of self-harm).

Generate a brief structured nurse handoff note with these fields:
- Symptom: (clinical term)
- Patient report: (what the patient said, in plain English)
- Urgency: (Low / Medium / High)
- Recommended action: (what the nurse should do next)

Be concise. Use standard clinical language."""

    return call_groq(prompt, max_tokens=1000)


def status_html(kind: str, text: str) -> str:
    """A coloured message card. kind: success / info / warn / error."""
    icons = {'success': '✅', 'info': 'ℹ️', 'warn': '⚠️', 'error': '⛔'}
    return (f'<div class="status-card status-{kind}"><span>{icons.get(kind, "")}</span>'
            f'<span>{text}</span></div>')


GENERATE_LABEL = '⚕️ Generate Clinical Note'
RATE_FIRST_LABEL = '⭐ Rate the note below to continue'
# After Generate is tapped, bring the note and the rating buttons into view
SCROLL_TO_NOTE_JS = """() => {
    const el = document.getElementById('note-anchor');
    if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
}"""


REVIEWER_TITLE = '🔑 Reviewer: approve new phrases (password required)'


def reviewer_title() -> str:
    try:
        count = len(pending_suggestions())
    except Exception:
        return REVIEWER_TITLE
    waiting = f'{count} waiting' if count else 'nothing waiting'
    return f'{REVIEWER_TITLE} · {waiting}'


DEPARTMENTS = ['Not specified', 'General outpatient (GOPD)', 'Antenatal', 'Postnatal',
               'Accident & Emergency', 'Medical/Surgical ward', 'Other', TEST_DEPARTMENT]


def extract_urgency(note: str):
    match = re.search(r'Urgency\W*\s*(Low|Medium|High)', note or '', re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def generate_or_ask_for_rating(audio_file, department, pending_note_id, language=DEFAULT_LANG):
    """Before making a new note, ask the nurse to rate the previous one."""
    if pending_note_id:
        keep = gr.update()
        warning = status_html('warn', '<b>Please rate the previous note first.</b> Scroll down and tap '
                                      '👍, 👎 or 🤷 Not sure, then tap Generate again.')
        return (keep, keep, keep, keep, keep, pending_note_id, warning, keep, warning,
                gr.update(value=RATE_FIRST_LABEL))
    result = process_audio(audio_file, department, language)
    note_made = bool(result[5])   # a note id means a note was generated and now needs rating
    return (*result, '', gr.update(value=RATE_FIRST_LABEL if note_made else GENERATE_LABEL))


def process_audio(audio_file, department='Not specified', language=DEFAULT_LANG):
    """Returns the 5 output boxes, the note id (for feedback) and a cleared feedback message."""
    if audio_file is None:
        return '', '', '', '', '', '', '', ''
    transcript = english_symptom = literal_meaning = confidence_str = ''
    try:
        transcript = transcribe_audio(audio_file, language)
        if not transcript:
            log_usage(department, 'no_speech')
            return ('No speech detected. Please record again, speaking clearly '
                    'for a few seconds.'), '', '', '', '', '', '', ''
        english_symptom, literal_meaning, confidence = translate_to_english(transcript, language)
        if confidence >= 0.9:
            confidence_str = f'{confidence:.0%} glossary match'
        elif confidence > 0:
            confidence_str = f'{confidence:.0%} partial match - please confirm with the patient'
        else:
            confidence_str = 'No glossary match - AI translation, needs nurse review'
        # Safety check that does not depend on the AI
        reasons = check_danger(glossary_term=english_symptom if confidence > 0 else '',
                               transcript=transcript,
                               keywords_key=lang_cfg(language).get('danger_keywords', ''),
                               english_text=f'{english_symptom} {literal_meaning}')
        clinical_note = generate_clinical_note(english_symptom, literal_meaning, language)
        note_id = log_usage(department, 'note', glossary_match=confidence > 0,
                            confidence=confidence, symptom=english_symptom,
                            urgency=extract_urgency(clinical_note),
                            danger_sign=bool(reasons))
        if confidence == 0:
            clinical_note = ('LOW CONFIDENCE - phrase not in glossary. '
                             'Confirm with the patient before acting.\n\n' + clinical_note)
        if reasons:
            clinical_note = ('URGENT - possible danger sign detected by the safety check. '
                             'Escalate now.\n\n' + clinical_note)
        return (transcript, english_symptom, literal_meaning, confidence_str, clinical_note,
                note_id, '', danger_banner(reasons))
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_usage(department, 'error')
        # Keep whatever already worked so the nurse still sees it
        reasons = check_danger(transcript=transcript, english_text=f'{english_symptom} {literal_meaning}',
                               keywords_key=lang_cfg(language).get('danger_keywords', ''))
        return (transcript or f'Error: {e}', english_symptom, literal_meaning,
                confidence_str, f'Could not finish: {e}\nPlease try again in a moment.', '', '',
                danger_banner(reasons))
    finally:
        # Remove only the cleaned wav temp file — no text files are written
        if audio_file:
            cleaned = Path(audio_file).with_name(Path(audio_file).stem + '_cleaned.wav')
            try:
                if cleaned.exists():
                    cleaned.unlink()
                    print(f'  [Cleanup] Removed {cleaned.name}')
            except Exception:
                pass


CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-deep: #1a3a2a;
    --green-mid: #2d6a4f;
    --green-bright: #40916c;
    --green-light: #74c69d;
    --green-pale: #d8f3dc;
    --cream: #faf7f2;
    --warm-white: #ffffff;
    --text-dark: #1a2e22;
    --text-mid: #3a5c47;
    --text-light: #6b9080;
    --accent-gold: #e9c46a;
    --accent-red: #c1440e;
    --shadow: 0 4px 24px rgba(26,58,42,0.10);
    --radius: 14px;
}

body, .gradio-container {
    background: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Header */
.beagrace-header {
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green-mid) 100%);
    border-radius: var(--radius);
    padding: 20px 28px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.beagrace-header::before {
    content: '🏥';
    font-size: 120px;
    position: absolute;
    right: -10px;
    top: -20px;
    opacity: 0.08;
}
.beagrace-header h1 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--warm-white) !important;
    font-size: 1.8rem !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.5px;
}
.beagrace-header p {
    color: var(--green-light) !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
    font-weight: 300;
}
.beagrace-header .badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: var(--accent-gold);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Panels */
.panel-upload, .panel-results {
    background: var(--warm-white);
    border-radius: var(--radius);
    padding: 28px;
    box-shadow: var(--shadow);
    border: 1px solid rgba(45,106,79,0.08);
}
.panel-label {
    font-family: 'DM Serif Display', serif;
    color: var(--green-deep);
    font-size: 1.1rem;
    font-weight: 400;
    margin-bottom: 4px;
    padding-bottom: 4px;
    border-bottom: 2px solid var(--green-pale);
}

/* Upload area */
.panel-upload .gr-audio {
    border: 2px dashed var(--green-light) !important;
    border-radius: 10px !important;
    background: var(--green-pale) !important;
}

/* Button */
#submit-btn {
    background: linear-gradient(135deg, var(--green-mid), var(--green-bright)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 14px !important;
    margin-top: 16px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 14px rgba(45,106,79,0.35) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
#submit-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(45,106,79,0.45) !important;
}

/* Output textboxes */
.gr-textbox textarea, .gr-textbox input {
    border: 1.5px solid var(--green-pale) !important;
    border-radius: 8px !important;
    background: var(--cream) !important;
    color: var(--text-dark) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
}
.gr-textbox label {
    color: var(--text-mid) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Clinical note special styling */
#clinical-note textarea {
    background: var(--green-deep) !important;
    color: var(--green-light) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

/* Pipeline steps */
.pipeline-steps {
    display: flex;
    gap: 8px;
    margin: 4px 0 10px 0;
    flex-wrap: wrap;
}
.step {
    background: var(--green-pale);
    color: var(--green-mid);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
}
.step-arrow {
    color: var(--text-light);
    font-size: 0.8rem;
}

/* Compact layout: smaller gaps between blocks, empty message slots take no space */
.gradio-container {
    --layout-gap: 10px !important;
    --block-padding: 8px 12px !important;
    --form-gap-width: 1px !important;
}
.slot { padding: 0 !important; margin: 0 !important; min-height: 0 !important;
        border: none !important; background: transparent !important; box-shadow: none !important; }
.slot .prose, .slot > div { min-height: 0 !important; }

/* Impact page */
.impact-wrap { padding: 4px 0; }
.impact-head { margin-bottom: 12px; }
.impact-title { font-family: 'DM Serif Display', serif; font-size: 1.5rem; color: var(--green-deep); }
.impact-sub { font-size: 0.85rem; color: var(--text-light); margin-top: 2px; }
.impact-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.impact-card {
    background: var(--warm-white); border-radius: var(--radius); padding: 16px 18px;
    box-shadow: var(--shadow); border-top: 4px solid var(--green-bright);
}
.impact-number { font-family: 'DM Serif Display', serif; font-size: 2.1rem; color: var(--green-mid); line-height: 1.1; }
.impact-label { font-size: 0.88rem; font-weight: 600; color: var(--text-dark); margin-top: 4px; }
.impact-note { font-size: 0.78rem; color: var(--text-light); margin-top: 2px; }
.impact-depts { margin-top: 14px; background: var(--green-pale); border-radius: 10px; padding: 12px 16px;
                font-size: 0.88rem; color: var(--text-dark); }
.impact-depts ul { margin: 6px 0 0 18px; padding: 0; }

/* Status message cards */
.status-card {
    display: flex; gap: 10px; align-items: flex-start;
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
    font-size: 0.9rem; line-height: 1.45;
}
.status-success { background: #d8f3dc; color: #1a3a2a !important; border-left: 5px solid #40916c; }
.status-info    { background: #e7f0fa; color: #1d3557 !important; border-left: 5px solid #457b9d; }
.status-warn    { background: #fff4d6; color: #5c4400 !important; border-left: 5px solid #e9c46a; }
.status-error   { background: #fde4dc; color: #7a2009 !important; border-left: 5px solid #c1440e; }
.status-card * { color: inherit !important; }

/* Teach-the-app and reviewer sections */
.soft-section, .reviewer-section {
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
    border: 1px solid rgba(45,106,79,0.12) !important;
    margin-top: 16px !important;
}
.reviewer-section { border-top: 4px solid var(--accent-gold) !important; }
.section-intro {
    background: var(--green-pale); color: var(--text-dark);
    border-radius: 10px; padding: 12px 16px; font-size: 0.9rem; line-height: 1.5;
}
.reviewer-intro { background: #fff8e6; border-left: 5px solid var(--accent-gold); }
.section-hint { font-size: 0.8rem; color: var(--text-light); margin-top: 4px; }
.step-title {
    font-family: 'DM Serif Display', serif; color: var(--green-deep);
    font-size: 1rem; margin: 14px 0 4px 0;
    padding-bottom: 4px; border-bottom: 2px solid var(--green-pale);
}
#suggest-btn, #review-load-btn {
    background: var(--warm-white) !important; color: var(--green-mid) !important;
    border: 2px solid var(--green-bright) !important; border-radius: 10px !important; font-weight: 600 !important;
}
#approve-btn {
    background: linear-gradient(135deg, var(--green-mid), var(--green-bright)) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(45,106,79,0.3) !important;
}
#reject-btn {
    background: var(--warm-white) !important; color: var(--accent-red) !important;
    border: 2px solid var(--accent-red) !important; border-radius: 10px !important; font-weight: 600 !important;
}

/* Danger banner: always white text on red */
.danger-banner, .danger-banner * {
    color: #ffffff !important;
}

/* Footer */
.footer-bar {
    background: var(--green-deep);
    border-radius: var(--radius);
    padding: 16px 24px;
    margin-top: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.footer-bar p {
    color: var(--text-light) !important;
    font-size: 0.8rem !important;
    margin: 0 !important;
}
.footer-bar .highlight {
    color: var(--green-light) !important;
    font-weight: 600;
}
"""

with gr.Blocks(css=CSS, title='BeaGrace Health') as demo:

    gr.HTML("""
    <div class="beagrace-header">
        <div class="badge">🌍 Technovation 2026 · UNICEF Generation Unlimited</div>
        <h1>🏥 BeaGrace Health</h1>
        <p>Ekiti Yoruba speech → structured clinical nurse handoff note · Supporting rural healthcare in Nigeria</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab('🩺 Nurse tool'):
            gr.HTML('''<div class="pipeline-steps">
        <div class="step">1️⃣ Record the patient speaking Yoruba</div>
        <div class="step-arrow">→</div>
        <div class="step">2️⃣ Tap Generate Clinical Note</div>
        <div class="step-arrow">→</div>
        <div class="step">3️⃣ Check the note and rate it</div>
    </div>''', elem_classes=['slot'])
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-label">📂 Patient audio</div>', elem_classes=['slot'])
                    audio_input = gr.Audio(
                        label='Upload audio file (Ekiti Yoruba)',
                        type='filepath',
                    )
                    language_input = gr.Dropdown(
                        choices=[(cfg['label'], code) for code, cfg in LANGUAGES.items()],
                        value=DEFAULT_LANG,
                        label='Patient language',
                        visible=len(LANGUAGES) > 1,
                    )
                    department_input = gr.Dropdown(
                        choices=DEPARTMENTS,
                        value='Not specified',
                        label='Department (optional)',
                    )
                    submit_btn = gr.Button(
                        GENERATE_LABEL,
                        elem_id='submit-btn',
                        variant='primary',
                    )
                    generate_status = gr.HTML('', elem_classes=['slot'])
                    gr.HTML("""
                    <div style="margin-top:4px; padding:10px 14px; background:#d8f3dc; border-radius:10px;">
                        <p style="margin:0; font-size:0.82rem; color:#2d6a4f; font-weight:500;">
                            🎤 <strong>Record with the microphone</strong> or upload a file (MP3, WAV, M4A, OGG)<br>
                            📍 <strong>Works with Yoruba speech.</strong> Most accurate for Ekiti dialect phrases.
                            Other phrases are marked "Low confidence" for you to check.
                        </p>
                    </div>
                    """)

                with gr.Column(scale=1):
                    gr.HTML('<div class="panel-label">🔬 Result</div>', elem_classes=['slot'])
                    transcript_out = gr.Textbox(label='What the patient said (Yoruba)', interactive=False)
                    with gr.Row():
                        english_out = gr.Textbox(label='English Symptom', interactive=False, scale=2)
                        confidence_out = gr.Textbox(label='Confidence', interactive=False, scale=1)
                    meaning_out = gr.Textbox(label='Literal Meaning (Ekiti dialect)', interactive=False)

            gr.HTML('<div class="panel-label" style="margin-top:6px;">📋 Clinical Nurse Handoff Note</div>',
                    elem_classes=['slot'], elem_id='note-anchor')
            danger_out = gr.HTML('', elem_classes=['slot'])
            clinical_note_out = gr.Textbox(
                label='Clinical note',
                show_label=False,
                interactive=False,
                lines=5,
                elem_id='clinical-note',
            )

            note_id_state = gr.State('')
            gr.HTML('<div class="panel-label" style="margin-top:6px;">✅ Was this note correct?</div>',
                    elem_classes=['slot'], elem_id='rating-anchor')
            feedback_comment = gr.Textbox(
                label='Comment (optional): write it first, then tap a rating below. '
                      'Please do not include patient names or details.',
                lines=2,
            )
            with gr.Row():
                correct_btn = gr.Button('👍 Note is correct')
                wrong_btn = gr.Button('👎 Note needs changes')
                unsure_btn = gr.Button('🤷 Not sure')
            feedback_status = gr.HTML('', elem_classes=['slot'])

            with gr.Accordion('🧠 Teach the app a phrase (if it was not recognised or was wrong)', open=False,
                              elem_classes=['soft-section']):
                gr.HTML("""
                <div class="section-intro">
                    <b>Help the app learn.</b> Tell us what the patient actually meant.
                    A Yoruba speaker will check it before it is added.
                    <div class="section-hint">Please do not include patient names or details.</div>
                </div>""")
                suggest_meaning = gr.Textbox(label='What the patient meant (English)', lines=2)
                suggest_term = gr.Textbox(label='Clinical term (optional), e.g. Headache')
                suggest_btn = gr.Button('📨 Send for review', elem_id='suggest-btn')
                suggest_status = gr.HTML('', elem_classes=['slot'])

            gr.HTML("""
            <p style="font-size:0.78rem; color:#6b9080; margin-top:6px;">
                🔒 Usage is counted anonymously to improve the app. No audio or patient details are stored.
                Only phrases you choose to send for review are saved, so they can be checked by a Yoruba speaker.
            </p>
            """)

        with gr.Tab('📊 Impact'):
            impact_out = gr.HTML(elem_classes=['slot'])
            impact_refresh = gr.Button('🔄 Refresh totals', elem_id='review-load-btn')
        with gr.Tab('🔑 Reviewer'):
            with gr.Accordion(REVIEWER_TITLE, open=True, elem_classes=['reviewer-section']) as reviewer_acc:
                gr.HTML("""
                <div class="section-intro reviewer-intro">
                    <b>For the Yoruba reviewer only.</b> Nurses send phrases the app did not understand.
                    Approve the correct ones and the app learns them for everyone.
                </div>""")
                gr.HTML(elem_classes=['slot'], value='<div class="step-title">Step 1 · Sign in</div>')
                with gr.Row(equal_height=True):
                    review_pw = gr.Textbox(label='Reviewer password', type='password', scale=3)
                    review_load_btn = gr.Button('🔓 Show waiting phrases', elem_id='review-load-btn', scale=2)
                review_status = gr.HTML('', elem_classes=['slot'])
                gr.HTML(elem_classes=['slot'], value='<div class="step-title">Step 2 · Check the phrase</div>')
                review_pick = gr.Dropdown(label='Phrase to review', choices=[])
                review_yoruba = gr.Textbox(label='Yoruba phrase (correct the spelling if needed)')
                with gr.Row():
                    review_meaning = gr.Textbox(label='English meaning')
                    review_term = gr.Textbox(label='Clinical term')
                with gr.Row():
                    review_dept = gr.Dropdown(choices=DEPARTMENTS, value='Not specified', label='Department')
                    review_name = gr.Textbox(label='Your name (reviewer)')
                gr.HTML(elem_classes=['slot'], value='<div class="step-title">Step 3 · Decide</div>')
                with gr.Row():
                    approve_btn = gr.Button('✅ Approve and add to glossary', elem_id='approve-btn')
                    reject_btn = gr.Button('❌ Reject', elem_id='reject-btn')

    gr.HTML("""
    <div class="footer-bar">
        <p>Developed by: <span class="highlight">Shalini Sivasamy</span> · Senior AI/ML Engineer · Virginia, USA</p>
        <p>Developed for: <span class="highlight">Christabel Ezem</span> · BeaGrace Foundation, Nigeria</p>
        <p>Programme: <span class="highlight">AI Ventures Accelerator · Technovation &amp; UNICEF Generation Unlimited 2026</span></p>
    </div>
    """)

    submit_btn.click(
        fn=generate_or_ask_for_rating,
        inputs=[audio_input, department_input, note_id_state, language_input],
        outputs=[transcript_out, english_out, meaning_out, confidence_out, clinical_note_out,
                 note_id_state, feedback_status, danger_out, generate_status, submit_btn],
    ).then(fn=None, js=SCROLL_TO_NOTE_JS)

    def send_suggestion(transcript, meaning, term, dept, note_id, language=DEFAULT_LANG):
        message = submit_suggestion(transcript, meaning, term, dept, note_id, language)
        if message.startswith('Thank you'):
            return status_html('success', message), '', '', gr.update(label=reviewer_title())
        return status_html('warn', message), meaning, term, gr.update()

    suggest_btn.click(
        fn=send_suggestion,
        inputs=[transcript_out, suggest_meaning, suggest_term, department_input, note_id_state, language_input],
        outputs=[suggest_status, suggest_meaning, suggest_term, reviewer_acc],
    )

    def _reviewer_ok(password):
        expected = os.environ.get('REVIEW_PASSWORD')
        return bool(expected) and password == expected

    def _pending_message(count):
        if count == 0:
            return status_html('info', '<b>All caught up.</b> No phrases are waiting for review.')
        return status_html('success', f'<b>{count} phrase{"s" if count != 1 else ""} waiting.</b> '
                                      'Choose one below in Step 2.')

    def load_pending(password):
        if not os.environ.get('REVIEW_PASSWORD'):
            return gr.update(choices=[], value=None), status_html(
                'error', 'Reviewer is not set up yet (add a REVIEW_PASSWORD secret).')
        if not _reviewer_ok(password):
            return gr.update(choices=[], value=None), status_html('error', '<b>Wrong password.</b> Please try again.')
        items = pending_suggestions()
        choices = [(f"{s['yoruba'][:50]}  →  {s['meaning'][:40]}", s['suggestion_id']) for s in items]
        return gr.update(choices=choices, value=None), _pending_message(len(items))

    def fill_review(suggestion_id):
        for s in pending_suggestions():
            if s['suggestion_id'] == suggestion_id:
                dept = s['department'] if s['department'] in DEPARTMENTS else 'Not specified'
                return s['yoruba'], s['meaning'], s['clinical_term'], dept
        return '', '', '', 'Not specified'

    def decide(password, suggestion_id, yoruba, meaning, term, dept, name, decision):
        if not _reviewer_ok(password):
            return status_html('error', '<b>Wrong password.</b>'), gr.update(), gr.update()
        if not suggestion_id:
            return status_html('warn', 'Choose a phrase in Step 2 first.'), gr.update(), gr.update()
        error = review(suggestion_id, decision, yoruba, meaning, term, dept, name)
        if error:
            return status_html('warn', error), gr.update(), gr.update()
        items = pending_suggestions()
        choices = [(f"{s['yoruba'][:50]}  →  {s['meaning'][:40]}", s['suggestion_id']) for s in items]
        done = ('<b>Approved.</b> The app now understands this phrase.' if decision == 'approved'
                else '<b>Rejected.</b> It will not be added.')
        left = f' {len(items)} left to review.' if items else ' All caught up.'
        return (status_html('success' if decision == 'approved' else 'info', done + left),
                gr.update(choices=choices, value=None), gr.update(label=reviewer_title()))

    review_load_btn.click(load_pending, inputs=[review_pw], outputs=[review_pick, review_status])
    review_pick.change(fill_review, inputs=[review_pick],
                       outputs=[review_yoruba, review_meaning, review_term, review_dept])
    review_fields = [review_pw, review_pick, review_yoruba, review_meaning, review_term, review_dept, review_name]
    approve_btn.click(lambda *a: decide(*a, 'approved'), inputs=review_fields,
                      outputs=[review_status, review_pick, reviewer_acc])
    reject_btn.click(lambda *a: decide(*a, 'rejected'), inputs=review_fields,
                     outputs=[review_status, review_pick, reviewer_acc])
    # Show the waiting count in the section title as soon as the page opens
    demo.load(lambda: gr.update(label=reviewer_title()), outputs=[reviewer_acc])

    def show_impact():
        try:
            return impact_html(compute_stats(sum(len(load_glossary(code)) for code in LANGUAGES)))
        except Exception as e:
            print(f'[impact] {e}')
            return status_html('info', 'Impact totals are not available right now. Please try again later.')

    demo.load(show_impact, outputs=[impact_out])
    impact_refresh.click(show_impact, outputs=[impact_out])

    def record_feedback(note_id, comment, rating):
        if not note_id:
            return (status_html('info', 'No note to rate yet, or this note is already rated.'),
                    comment, note_id, '', gr.update())
        log_feedback(note_id, rating, comment)
        # Clear the note id: this note is rated, so the next one can be generated
        return status_html('success', '<b>Thank you!</b> Your rating' + (' and comment were' if (comment or '').strip() else ' was') + ' saved.'), '', '', '', gr.update(value=GENERATE_LABEL)

    correct_btn.click(
        fn=lambda note_id, comment: record_feedback(note_id, comment, 'correct'),
        inputs=[note_id_state, feedback_comment],
        outputs=[feedback_status, feedback_comment, note_id_state, generate_status, submit_btn],
    )
    wrong_btn.click(
        fn=lambda note_id, comment: record_feedback(note_id, comment, 'needs_changes'),
        inputs=[note_id_state, feedback_comment],
        outputs=[feedback_status, feedback_comment, note_id_state, generate_status, submit_btn],
    )
    unsure_btn.click(
        fn=lambda note_id, comment: record_feedback(note_id, comment, 'not_sure'),
        inputs=[note_id_state, feedback_comment],
        outputs=[feedback_status, feedback_comment, note_id_state, generate_status, submit_btn],
    )

if __name__ == '__main__':
    demo.launch()