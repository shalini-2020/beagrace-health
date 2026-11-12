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
Glossary Matching and Danger-Sign Checks (ClinGuard)
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
| Dialect matching and safety checks | [ClinGuard](https://pypi.org/project/clinguard/) (`pip install clinguard`) |
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
├── usage_log.py        # Anonymous usage, feedback and review log (content logging off by default)
├── compare_baselines.py # Blinded comparison: general translator vs LLM vs ClinGuard
├── eval_cases_template.csv # Template for reviewer-labelled evaluation cases
├── impact.py           # Live Impact tab (totals from the pilot)
├── languages.json      # Supported languages and their glossary files
├── glossary.json       # 60-entry Ekiti Yoruba medical vocabulary
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🔒 Privacy and logging

BeaGrace is privacy-first. By default it stores **no audio, no transcript and no
patient details**. Each interaction (note, no speech or error) gets one
`note_id`, and the anonymous usage record holds only:

| Field | Meaning |
|---|---|
| `note_id` | Links the usage record to the nurse's rating (and to content records, if enabled) |
| `language`, `department`, `outcome`, `error_stage` | What happened, and where it failed if it did |
| `nurse_code` | Optional pseudonymous code (must contain a digit, e.g. N07), never a name |
| `confidence`, `confidence_level` | Match score and category: `high`, `partial`, `low` (AI fallback), `none` |
| `match_id`, `symptom` | Which curated glossary entry matched (an id, not patient words) |
| `danger_sign`, `danger_rule_ids` | Whether the safety check fired, and which rules (positions in `danger_signs.json`) |
| `urgency` | Urgency level from the note |
| `speech_model`, `app_version`, `clinguard_version`, `glossary_version`, `rules_version` | Which versions produced the result |

Nurse feedback is linked by `note_id`: the quick rating, optional answers on whether the meaning was preserved, whether the note is useful and whether danger signs were handled well (each with an "unable to assess" or "not applicable" choice), and an optional correction comment.
Obvious identifiers in comments (phone numbers, emails, dates, long numbers)
are masked before saving. Names cannot be detected reliably, so the comment
box asks nurses not to include patient details.

**Clinical content logging is OFF by default.** Setting `LOG_CONTENT=true`
also stores the transcript, English meaning, AI fallback translation and
final note under the same `note_id`, in a separate folder that syncs only to
its own private dataset (`CONTENT_LOG_REPO`). A switch is not compliance:
enable it only after the pilot site's written authorization, a lawful basis
and patient notice or consent where required, restricted access, and agreed
retention and deletion rules.

For accuracy evaluation, use a separate dataset of consented, de-identified
recordings or standard test phrases rather than routine patient encounters.

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

## 🌐 Adding a new language

The method is not tied to one dialect. To add a language (for example Hausa or Igbo):

1. Native speakers create a glossary file in the same format as `glossary.json`, with each local phrase mapped to its English meaning and clinical term.
2. Add an entry to `languages.json` with the language name, its speech-recognition code, and the glossary file name.
3. Optionally, add local danger-sign words to `danger_signs.json`.

A **Patient language** selector appears automatically once more than one language is listed.

---

## 📊 Evaluation

**Status (11 November 2026): the clinical accuracy of BeaGrace has not yet been measured.**
The figures below describe how the system behaves, not how often its notes are right.

| Measure | Value | What it does and does not show |
|---|---|---|
| Glossary matching on written text | 177 of 178 misspelled variants matched; 2 of 60 phrases matched a wrong term when their own entry was removed | Run with ClinGuard 0.1.2's `scripts/evaluate_lexicon.py` on the 60-entry glossary. Tests text matching only: no speech recognition and no clinical judgement. |
| Fuzzy match threshold | 75% confidence | Below this, the system translates with the LLM instead and marks the note LOW CONFIDENCE for nurse review. |
| Clinical note structure | 4 of 4 fields on every output | Format only. |
| Time to a note | About 3 to 15 seconds on a strong connection | As timed by testers in the first pilot session. |

### Early pilot feedback (6 November 2026)

The first simulated-use session was held on 6 November 2026: nurses, a midwife and a
student nurse acted out patient scenarios (no real patients) and filled in five written
feedback forms. This is informal usability feedback, **not accuracy evidence**: testers
assessed their own attempts, and some forms did not record what the app showed.

What it showed:

- The app ran without crashing and produced notes within seconds.
- Testers found the notes clear and said the tool would help with the language barrier
  between patients and nurses.

Problems reported:

- **Natural speech:** when the speaker added extra words or spoke freely, the app often
  could not interpret the phrase. In the usage log for the session, 4 of the 8 notes
  were "not understood" and fell back to the LLM.
- **Yoruba spelling:** the transcription was not written in correct Ekiti Yoruba, so it
  could not be used for documentation as it stood.
- **A possible wrong match:** one form recorded a symptom that did not fit the intended
  meaning. The output was not saved, so this is unconfirmed.

Requests from testers, recorded for later and not yet built: fields for vital signs,
gestational age, expected delivery date and pain intensity; and low cost and easy access.

From 11 November 2026, each session uses a scenario sheet that records exactly what the
app showed for every attempt. Accuracy will be reported only from the independent
evaluation below, where reference answers are written by reviewers who never see the
app's output.

### Comparing against baselines (`compare_baselines.py`)

A blinded three-way comparison on the same input text: a general-purpose
translator, the same LLM without the glossary or ClinGuard, and BeaGrace with
ClinGuard. Qualified Yoruba-speaking and clinical reviewers label the cases
(`eval_cases_template.csv`) without seeing any system output, then score the
shuffled outputs. The script reports meaning accuracy, important-omission
rate, danger-sign recall, false-alert rate and the fallback rate, with 95%
intervals, and splits ClinGuard results into glossary-matched and fallback
cases. Use synthetic phrases or consented, de-identified transcripts only.
Limitation: output wording styles differ by system, so blinding is partial.

Nurse ratings in the Impact tab show acceptance by nurses, not verified accuracy.

### Independent nurse evaluation (🧪 Evaluation tab, `nurse_eval.py`)

Every evaluator rates the **same** notes, made once from the **same** approved
recordings, so results can be compared across nurses. Kept separate from pilot
use: none of it is counted in the usage, feedback or Impact totals.

**Before any nurse rates anything: lock the reference answers**

1. Approved, simulated Ekiti Yoruba recordings (consenting speakers; never a real
   patient encounter).
2. An independent Yoruba reviewer writes the transcript and meaning of each.
3. An independent clinical reviewer writes the clinical term, urgency and any
   danger sign.
4. Disagreements are resolved, and the answers are saved in `eval_references.csv`
   (template: `eval_references_template.csv`) and **locked**: record the date and
   a file hash (`certutil -hashfile eval_references.csv SHA256` on Windows) and do
   not change them after nurse rating starts. Reviewers never see system output.
   The file is git-ignored: keep it private.

**Running the evaluation**

1. **Scenarios.** The clinical lead writes standardised scenarios in
   `eval_scenarios.json` and sets `"approved": true` once approved. Evaluators see
   the `situation` and the Yoruba `phrase`, so the situation must give the **setting
   only** (who, where), never the symptom or its English meaning: that would answer
   question 1 for them. Reference answers live only in `eval_references.csv`.
2. **Recordings.** Name files `<scenario_id>__<speaker>.<ext>` (e.g.
   `ANC-01__S1.m4a`) in a private folder such as `eval_recordings/` (git-ignored).
   Speakers' consent must cover evaluators listening to the recordings.
3. **Generate the notes once**, on your computer (API keys in `.env`,
   `hf auth login` with write access):
   `python nurse_eval.py generate eval_recordings --with-baseline`
   This makes the BeaGrace + ClinGuard note and an LLM-only note from the same
   transcript (same LLM and note prompt, no glossary, no rule-based danger check).
   Notes and recordings go to the **private** dataset only. Each note gets a random
   label; its condition is stored for the administrator only.
4. **Check them (admin only):** `python nurse_eval.py show` prints each recording's
   notes side by side with their condition. Never share this.
5. **Evaluators** (codes in an `EVALUATOR_CODES` secret, e.g. `EV01`) see only the
   scenario, the original recording and the note. They answer the same five
   questions, in their own shuffled order, with the two versions of a scenario never
   one after the other. Question 1 counts only from evaluators who understand Ekiti
   Yoruba. After rating everything, a short **debrief** asks which kind of system made
   each note, to check whether the blinding held.
6. **Reviewer adjudication:** `python nurse_eval.py adjudication eval_references.csv`
   writes `eval_out/reviewer_adjudication_<batch>.csv`: every note, shuffled, next to
   its reference answer, without the condition. The reference reviewers mark, with
   one written criterion for both systems, whether the note identifies the reference
   danger sign, signals a danger that is not there, and matches the reference term.
   Together these separate the four danger-sign cases that nurse question 4 can blur:
   genuine danger sign identified, genuine danger sign missed, no danger sign and none
   signalled, and no danger sign but a warning wrongly introduced.
7. **Score:** `python nurse_eval.py score eval_references.csv` reports, per system,
   danger-sign detection, false danger signals and meaning accuracy (from the
   reviewers), urgency classification (automatic, reported separately), the
   rule-based alert (ClinGuard only), evaluator answers and agreement; then a
   **paired** ClinGuard vs LLM-only comparison on the same recordings (exact
   McNemar) and the blinding check. `eval_out/nurse_evaluation_results.csv` is
   admin-only: it shows each note's condition.

8. **Speech recognition, reported separately:**
   `python nurse_eval.py asr eval_references.csv` measures how accurately speech
   recognition transcribed each recording against the Yoruba reviewer's reference
   transcript (`reference_transcript`, written from the recording, never from the
   app's output): word error rate, corpus-level (total errors / total reference
   words), per recording, by scenario `category` and by department, plus
   clinical-term preservation for the key terms the reviewer lists
   (`reference_key_terms`, separated by `;`). Normalisation is fixed in advance:
   the **primary** WER ignores punctuation, case and all diacritics (tone marks and
   the dot below); the **tone-sensitive** WER keeps them. The scoring refuses a
   malformed reference file (e.g. an unquoted comma), so edit it in Excel.

Limitations: blinding is partial, because ClinGuard notes can carry the
safety-check banner and a "LOW CONFIDENCE" prefix, which LLM-only notes never do.
The interpretation comparison gives both systems the same transcript, so it does not
measure end-to-end accuracy; that is what the separate speech-recognition measure is
for. Audio is fetched only after a valid evaluator code, but once playing, its
temporary address could be passed on by someone who already has access: keep
evaluator codes private and make sure the consent forms cover playback.

The 👍 / 👎 / 🤷 buttons in the Nurse tool remain the everyday pilot feedback.

### Several complaints in one recording

A glossary phrase covers one complaint. After a glossary match, the app asks the AI
whether the patient clearly mentioned anything else; if so, it is added to the note as
"ALSO MENTIONED (AI, not glossary-verified) - confirm with the patient" and included in
the danger-keyword check. When nothing matches the glossary, the AI translates the whole
transcript as before.

### Required fields, facility codes and type of use

For real use, every note needs a department, a nurse code (e.g. `N07`) and a
facility code (e.g. `FAC001`); the app will not generate a note without them. The
department "Testing (not real use)" is exempt, so testing never needs codes. The
facility code is remembered on each phone; the nurse code is not, in case phones are
shared.

`PILOT_MODE` (Space variable, default `simulated`) is recorded with every interaction
and sets the Impact page and PDF wording: `simulated` = nurses acting out patient
scenarios, no real patients; `clinical` = real patients. Record any change in
`CHANGELOG.md`.

Nurses enter a facility code (e.g. `FAC001`). Set the issued codes
in a `FACILITY_CODES` secret; the pilot lead keeps the code-to-facility list
privately. Facility names are never stored.

### Pilot health check

`python pilot_check.py` (after `hf auth login`) shows attempts, errors by stage,
ratings, missing codes, notes per nurse, facility and department, and unrated notes,
since the pilot started. It reads the private logs only and shows no patient data.

### Packages

`requirements.txt` pins exact versions. Upgrade on purpose: change a version, test,
then deploy, and note it in `CHANGELOG.md`.

### Dated reports and versions

The Impact tab's **Download dated PDF report** button creates a snapshot with
the period, counts, denominators, interpretation paths, software versions and
definitions. Its data fingerprint can be re-checked later with
`python impact_report.py --verify <records counted up to>`. Every interaction
records the app version, a code fingerprint and the ClinGuard, glossary and
danger-rule versions at that moment, so results stay tied to the release that
produced them. Bump `APP_VERSION` in `app.py` with each release.

---

## 🗺️ Roadmap

**Phase 1 — Patient speech → nurse note (in progress)**

Done:
- Ekiti Yoruba speech → clinical nurse handoff note, deployed on Hugging Face Spaces (June 2026)
- 60-entry symptom glossary, low-confidence flagging and rule-based danger-sign check
- Simulated-use pilot with nurses at one facility (from 6 November 2026)

Next, in this order:
- Written authorisation from the facility, and its decision on ethics review
- Independent, blinded nurse evaluation (see Evaluation)
- Use with real patients only after that evaluation and the facility's approvals

Features are frozen until the evaluation is complete.

**Phase 2 — After Phase 1 is live and evaluated**
- Nurse → patient messages in Yoruba, from a phrase bank translated and checked by
  native speakers and clinicians (not free AI translation), under its own protocol
- Expand the glossary, including Medical/Surgical phrases

---

## 👥 Team

**Founder:** Christabel Ezem · BeaGrace Foundation, Nigeria

**Principal AI Architect and Engineering Lead:** Shalini Sivasamy · designed and built the system · AIVA Expert Mentor to the BeaGrace team · IEEE Senior Member

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

This is a working prototype built for the AIVA / Technovation 2026 programme. It is not a fully developed or clinically validated product. The clinical notes generated by this system are AI-assisted and should not be used as a substitute for professional medical judgment. Its clinical accuracy has not yet been measured; an independent evaluation with qualified healthcare workers is planned before any use with real patients.
