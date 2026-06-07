from pathlib import Path
import json
import re
import os
import time
import requests
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from fuzzywuzzy import fuzz
import assemblyai as aai
from deep_translator import GoogleTranslator
import librosa
import soundfile as sf

AUDIO_FILE = Path('ekiti yoruba 3-WA0277.mp3')
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
                'standard_yoruba': entry.get('standard_yoruba', ''),
                'english': english,
                'meaning': entry.get('literal_meaning', ''),
            })

    print('---------------------matches')
    print(matches)
    return sorted(matches, key=lambda x: x['ratio'], reverse=True)


def preprocess_audio(audio_path: Path) -> Path:
    print(f'  [Preprocess] Loading {audio_path.name}...')
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp * 0.95
    cleaned_path = audio_path.with_name(audio_path.stem + '_cleaned.wav')
    sf.write(str(cleaned_path), y, sr)
    print(f'  [Preprocess] Saved cleaned audio -> {cleaned_path.name}')
    return cleaned_path


def transcribe(audio_path: Path) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f'Audio file not found: {audio_path}')
    cleaned_path = preprocess_audio(audio_path)
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    config = aai.TranscriptionConfig(language_code='yo')
    transcriber = aai.Transcriber(config=config)
    print(f'  [AssemblyAI] Uploading and transcribing {cleaned_path.name}...')
    transcript = transcriber.transcribe(str(cleaned_path))
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f'AssemblyAI transcription failed: {transcript.error}')
    return transcript.text.strip()


def translate_yoruba_to_english(text: str) -> tuple:
    """Returns (english_symptom, literal_meaning)"""""
    glossary_matches = fuzzy_match_glossary(text)
    if glossary_matches:
        match = glossary_matches[0]
        print(f'  [Glossary match: {match["dialect"]} -> {match["english"]} (confidence: {match["ratio"]:.2%})]')
        return match['english'], match.get('meaning', '')
    print('  [No glossary match, falling back to Google Translate]')
    return GoogleTranslator(source_language='yo', target_language='en').translate(text), ''


def generate_clinical_note(english_symptom: str, literal_meaning: str = '') -> str:
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

    raise RuntimeError('Groq unavailable after 3 retries.')


def main() -> None:
    transcript = transcribe(AUDIO_FILE)
    print('Transcript:')
    print(transcript)

    transcript_path = AUDIO_FILE.with_suffix('.txt')
    transcript_path.write_text(transcript, encoding='utf-8')
    print(f'Saved transcript to {transcript_path}')

    english_symptom, literal_meaning = translate_yoruba_to_english(transcript)
    print(f'\nEnglish symptom: {english_symptom}')
    print(f'Literal meaning: {literal_meaning}')

    clinical_note = generate_clinical_note(english_symptom, literal_meaning)
    print('\nClinical note:')
    print(clinical_note)

    note_path = AUDIO_FILE.with_name(AUDIO_FILE.stem + '.clinical.txt')
    note_path.write_text(clinical_note, encoding='utf-8')
    print(f'Saved clinical note to {note_path}')


if __name__ == '__main__':
    main()