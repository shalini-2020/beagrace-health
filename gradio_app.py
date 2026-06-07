import gradio as gr
import json
import re
import os
import time
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fuzzywuzzy import fuzz
import assemblyai as aai
from deep_translator import GoogleTranslator
import librosa
import soundfile as sf

GLOSSARY_FILE = Path('glossary.json')
ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')


def load_glossary() -> list:
    if GLOSSARY_FILE.exists():
        with open(GLOSSARY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text)).lower().strip()


def fuzzy_match_glossary(text: str, threshold: int = 75) -> list:
    glossary = load_glossary()
    matches = []
    text_clean = clean_text(text)
    for entry in glossary:
        dialect = entry.get('dialect_expression', '')
        english = entry.get('english_medical_term', '')
        if not dialect:
            continue
        dialect_clean = clean_text(dialect)
        token_score = fuzz.token_set_ratio(text_clean, dialect_clean)
        partial_score = fuzz.partial_ratio(text_clean, dialect_clean)
        best_score = max(token_score, partial_score)
        if best_score >= threshold:
            matches.append({
                'ratio': best_score / 100.0,
                'dialect': dialect,
                'english': english,
                'meaning': entry.get('literal_meaning', ''),
            })
    return sorted(matches, key=lambda x: x['ratio'], reverse=True)


def preprocess_audio(audio_path: str) -> str:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp * 0.95
    cleaned_path = str(Path(audio_path).with_name(Path(audio_path).stem + '_cleaned.wav'))
    sf.write(cleaned_path, y, sr)
    return cleaned_path


def transcribe_audio(audio_path: str) -> str:
    cleaned_path = preprocess_audio(audio_path)
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    config = aai.TranscriptionConfig(language_code='yo')
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(cleaned_path)
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f'Transcription failed: {transcript.error}')
    return transcript.text.strip()


def translate_to_english(text: str):
    matches = fuzzy_match_glossary(text)
    if matches:
        match = matches[0]
        return match['english'], match['meaning'], match['ratio']
    translated = GoogleTranslator(source_language='yo', target_language='en').translate(text)
    return translated, '', 0.0


def generate_clinical_note(english_symptom: str, literal_meaning: str) -> str:
    context = f'Literal meaning from patient in Ekiti Yoruba dialect: "{literal_meaning}"' if literal_meaning else ''
    prompt = f"""You are a clinical documentation assistant supporting nurses in a rural Nigerian healthcare facility.

A patient has reported the following symptom (translated from Ekiti Yoruba):
"{english_symptom}"
{context}

Important: interpret the symptom based strictly on the literal meaning provided. Do not assume urinary symptoms unless explicitly stated.

Generate a brief structured nurse handoff note with these fields:
- Symptom: (clinical term)
- Patient report: (what the patient said, in plain English)
- Urgency: (Low / Medium / High)
- Recommended action: (what the nurse should do next)

Be concise. Use standard clinical language."""

    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': 'llama-3.3-70b-versatile',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 300,
    }
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
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f'  [Groq] attempt {attempt + 1} failed: {e}')
            time.sleep(3)
    raise RuntimeError('Clinical note generation failed after 3 retries.')


def process_audio(audio_file):
    if audio_file is None:
        return '', '', '', '', ''
    try:
        transcript = transcribe_audio(audio_file)
        english_symptom, literal_meaning, confidence = translate_to_english(transcript)
        clinical_note = generate_clinical_note(english_symptom, literal_meaning)
        confidence_str = f'{confidence:.0%} glossary match' if confidence > 0 else 'Google Translate fallback'
        return transcript, english_symptom, literal_meaning, confidence_str, clinical_note
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f'Error: {str(e)}', '', '', '', ''
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
    padding: 36px 40px;
    margin-bottom: 28px;
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
    font-size: 2.2rem !important;
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
    margin-bottom: 16px;
    padding-bottom: 10px;
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
    margin: 20px 0;
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
    <div class="pipeline-steps">
        <div class="step">🎙️ Yoruba Audio</div>
        <div class="step-arrow">→</div>
        <div class="step">🤖 AssemblyAI ASR</div>
        <div class="step-arrow">→</div>
        <div class="step">📖 Fuzzy Glossary</div>
        <div class="step-arrow">→</div>
        <div class="step">⚕️ Groq LLaMA-3.3</div>
        <div class="step-arrow">→</div>
        <div class="step">📋 Clinical Note</div>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML('<div class="panel-label">📂 Patient Audio Upload</div>')
            audio_input = gr.Audio(
                label='Upload audio file (Ekiti Yoruba)',
                type='filepath',
            )
            submit_btn = gr.Button(
                '⚕️ Generate Clinical Note',
                elem_id='submit-btn',
                variant='primary',
            )
            gr.HTML("""
            <div style="margin-top:16px; padding:14px; background:#d8f3dc; border-radius:10px;">
                <p style="margin:0; font-size:0.82rem; color:#2d6a4f; font-weight:500;">
                    💡 <strong>Supported formats:</strong> MP3, WAV, M4A, OGG<br>
                    📍 Optimised for <strong>Ekiti Yoruba</strong> dialect medical vocabulary
                </p>
            </div>
            """)

        with gr.Column(scale=1):
            gr.HTML('<div class="panel-label">🔬 Pipeline Output</div>')
            transcript_out = gr.Textbox(label='ASR Transcript (Yoruba)', interactive=False)
            with gr.Row():
                english_out = gr.Textbox(label='English Symptom', interactive=False, scale=2)
                confidence_out = gr.Textbox(label='Confidence', interactive=False, scale=1)
            meaning_out = gr.Textbox(label='Literal Meaning (Ekiti dialect)', interactive=False)

    gr.HTML('<div class="panel-label" style="margin-top:20px;">📋 Clinical Nurse Handoff Note</div>')
    clinical_note_out = gr.Textbox(
        label='',
        interactive=False,
        lines=7,
        elem_id='clinical-note',
    )

    gr.HTML("""
    <div class="footer-bar">
        <p>Mentor: <span class="highlight">Shalini Sivasamy</span> · New York Life · Virginia, USA</p>
        <p>Student: <span class="highlight">Christabel Ezem</span> · BeaGrace Foundation, Nigeria</p>
        <p>Programme: <span class="highlight">AIVA / Technovation 2026</span></p>
    </div>
    """)

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[transcript_out, english_out, meaning_out, confidence_out, clinical_note_out],
    )

if __name__ == '__main__':
    demo.launch()