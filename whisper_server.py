"""
Runs as a separate process called by streamlit_app.py.
Transcribes audio and prints JSON to stdout.
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

def transcribe(audio_path: str) -> str:
    import librosa
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        device=-1,
        return_timestamps=True,
    )
    audio, _ = librosa.load(audio_path, sr=16000)
    result = pipe(audio, generate_kwargs={"language": "yo"})
    return result["text"].strip()

if __name__ == "__main__":
    try:
        audio_path = sys.argv[1]
        text = transcribe(audio_path)
        # Only print JSON — nothing else to stdout
        print(json.dumps({"transcript": text}))
    except Exception as e:
        # Print error as JSON so caller can handle it
        print(json.dumps({"error": str(e)}))
        sys.exit(1)