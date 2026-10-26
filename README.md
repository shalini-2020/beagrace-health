---
title: Beagrace
emoji: 🏃
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 6.17.3
python_version: '3.13'
app_file: app.py
pinned: false
---

# 🏥 BeaGrace Health

**Ekiti Yoruba Speech → Clinical Nurse Handoff Note**

An AI pipeline that transcribes patient speech in Ekiti Yoruba dialect and generates structured clinical nurse handoff notes in English — supporting rural healthcare workers in Nigeria.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-brightgreen)](https://ss-shalini-2021-beagrace.hf.space)
[![Technovation 2026](https://img.shields.io/badge/Technovation-2026-blue)](https://technovation.org)
[![UNICEF Generation Unlimited](https://img.shields.io/badge/UNICEF-Generation%20Unlimited-009edb)](https://www.generationunlimited.org)

---

## 🌍 The Problem

In rural Nigeria, patients speak local dialects like Ekiti Yoruba. Nurses are trained to document in clinical English. This communication gap leads to misdiagnosis, poor documentation, and compromised patient care.

**Google Translate tells you what a patient said. BeaGrace tells a nurse what to do about it.**

---

## 🚀 Live Demo

👉 **[https://ss-shalini-2021-beagrace.hf.space](https://ss-shalini-2021-beagrace.hf.space)**

Upload an audio file of a patient speaking in Ekiti Yoruba. The app returns a structured clinical note in seconds.

---

## 🔬 Pipeline

```
Patient speaks Ekiti Yoruba
        ↓
Audio Preprocessing (librosa)
16kHz resampling · silence trimming · amplitude normalization
        ↓
AssemblyAI Transcription (Yoruba ASR)
        ↓
Fuzzy Glossary Matching (fuzzywuzzy)
60-entry Ekiti Yoruba medical vocabulary · 75% confidence threshold
        ↓
Groq GPT-OSS-120B Clinical Note Generation
        ↓
Structured Nurse Handoff Note
Symptom · Patient Report · Urgency · Recommended Action
```

---

## 📋 Example Output

**Audio input:** Patient says *"Ara mi n jo"* (Ekiti Yoruba for "my body is burning")

**Clinical note generated:**
```
- Symptom: Pyrexia / Generalized burning sensation
- Patient report: Patient reports their body feels like it is burning
- Urgency: Medium
- Recommended action: Check temperature and vital signs, assess for 
  fever or dermal irritation, rule out infection
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Audio preprocessing | librosa + soundfile |
| Speech to text | AssemblyAI (Yoruba) |
| Dialect matching | fuzzywuzzy |
| Clinical structuring | Groq GPT-OSS-120B |
| Web interface | Gradio |
| Hosting | Hugging Face Spaces |

---

## 📁 Repository Structure

```
beagrace-health/
├── app.py              # Gradio web app (main entry point)
├── cli_transcribe.py   # Command-line version: one audio file → clinical note
├── safety.py           # Danger-sign check, independent of the AI
├── danger_signs.json   # Editable list of danger signs (nurse-reviewed)
├── learning.py         # Learning glossary: nurse suggestions + reviewer approval
├── usage_log.py        # Anonymous usage, feedback and review log
├── glossary.json       # 60-entry Ekiti Yoruba medical vocabulary
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/shalini-2020/beagrace-health.git
cd beagrace-health
```

**2. Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set environment variables**

Create a `.env` file in the project root:
```
ASSEMBLYAI_API_KEY=your_assemblyai_key
GROQ_API_KEY=your_groq_key
```

Get your keys:
- AssemblyAI: [assemblyai.com](https://assemblyai.com) — free tier includes $50 credits
- Groq: [console.groq.com](https://console.groq.com) — free tier, 14,400 requests/day

**5. Run the app**
```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## 🗂️ Glossary

The `glossary.json` file contains 60 entries: Ekiti Yoruba symptom expressions (with spelling variants), including 12 Accident & Emergency, 12 antenatal and 11 postnatal phrases. Each is mapped to its clinical English equivalent and tagged with the department it came from. Each entry includes:

```json
{
  "dialect_expression": "Ara mi n jo",
  "standard_yoruba": "Ara mi n jo",
  "english_medical_term": "Burning sensation",
  "literal_meaning": "my body is burning",
  "department": "General"
}
```

Symptoms covered include headache, burning sensation, nausea, fever, body aches, chest pain, dizziness, shortness of breath, joint pain, muscle weakness, tingling sensation, sore throat, tinnitus, and more.

---

## 📊 Evaluation

| Metric | Value |
|---|---|
| Glossary match recall | 100% on 21 known symptoms |
| Fuzzy match threshold | 75% confidence |
| Clinical note structure | 4/4 fields on every output |
| End-to-end latency | ~10-15 seconds |

For cases where confidence falls below 75%, the system translates with the LLM instead and marks the note LOW CONFIDENCE for nurse review.

---

## 🗺️ Roadmap

**Phase 1 — Live (June 2026)**
- Ekiti Yoruba speech → clinical nurse handoff note
- 60-entry symptom glossary
- Deployed on Hugging Face Spaces

**Phase 2 — October 2026**
- Expand glossary to 100+ symptoms
- Low-confidence flagging for manual nurse review
- Clinical validation with nurses at rural facilities in Ekiti State
- Nurse → patient response in Yoruba (English speech → Yoruba text)

---

## 👥 Team

**Founder:** Christabel Ezem · BeaGrace Foundation, Nigeria

**Technical Mentor:** Shalini Sivasamy · Senior AI/ML Engineer · IEEE Senior Member

**Programme:** AIVA / Technovation 2026 · UNICEF Generation Unlimited

---

## 📄 License

This project is open source. The glossary and pipeline architecture are available for adaptation to other African dialect communities under the MIT License.

---

## 🙏 Acknowledgements

- [AssemblyAI](https://assemblyai.com) for Yoruba speech recognition
- [Groq](https://groq.com) for GPT-OSS-120B inference
- [Hugging Face](https://huggingface.co) for free Spaces hosting
- [AIVA / Technovation](https://technovation.org) and [UNICEF Generation Unlimited](https://www.generationunlimited.org) for the mentorship programme

---

## ⚠️ Disclaimer

This is a working prototype built for the AIVA / Technovation 2026 programme. It is not a fully developed or clinically validated product. The clinical notes generated by this system are AI-assisted and should not be used as a substitute for professional medical judgment. Clinical validation with qualified healthcare workers is planned for Phase 2.
