from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import os
import numpy as np
import librosa
import soundfile as sf
import assemblyai as aai

AUDIO_FILE = Path('ekiti yoruba 1-WA0273.mp3')
ASSEMBLYAI_API_KEY = os.environ.get('ASSEMBLYAI_API_KEY')


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
    cleaned_path = preprocess_audio(audio_path)

    aai.settings.api_key = ASSEMBLYAI_API_KEY
    config = aai.TranscriptionConfig(language_code='yo')
    transcriber = aai.Transcriber(config=config)

    print(f'  [AssemblyAI] Uploading {cleaned_path.name}...')
    transcript = transcriber.transcribe(str(cleaned_path))

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f'Transcription failed: {transcript.error}')

    return transcript.text.strip()


if __name__ == '__main__':
    result = transcribe(AUDIO_FILE)
    print(f'\nTranscript: "{result}"')
