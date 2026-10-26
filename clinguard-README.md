# ClinGuard

**Safety-first building blocks for clinical voice AI in low-resource languages and dialects.**

Mainstream speech and translation tools work poorly for many of the world's languages and
dialects. In healthcare, a misunderstood symptom can become a wrong note, and a wrong note can
harm a patient. ClinGuard gives developers the safety layer around their AI: a native-speaker
lexicon that grounds meaning, rule-based danger-sign checks that never depend on the AI, clear
confidence levels, and a human-reviewed learning loop.

ClinGuard is **model-agnostic**. It does not call any AI provider. Plug in any speech recogniser,
translator or note generator as plain Python functions.

```bash
pip install clinguard
```

## Why

| Problem | What ClinGuard does |
|---|---|
| Speech recognition writes dialect speech with varying tone marks and spellings | Normalises tone marks and accents; tolerant, length-aware fuzzy matching |
| General translators misread dialect phrases | **Lexicon first**: meaning comes from a glossary curated by native speakers |
| AI output can look confident when it is wrong | Every result carries a confidence level: `high` (lexicon), `low` (fallback), `none` |
| Emergencies must never be missed because of an AI error | **Rule-based danger-sign check**, independent of any model |
| Lexicons are never complete | **Human-in-the-loop learning**: users suggest, a trusted reviewer approves, with a full audit trail |

## Quick start

```python
from clinguard import ClinGuard, DangerRules, Lexicon

lexicon = Lexicon.from_file("my_lexicon.json", language="yo")

guard = ClinGuard(
    lexicon,
    DangerRules(local_keywords={"yo": ["warapa"]}),   # rules reviewed by your clinicians
    language="yo",
    translate=my_translate,      # optional: your model, used only when the lexicon has no match
    write_note=my_note_writer,   # optional: your note generator
)

a = guard.assess("Orí mi ń fọ́ mi")      # text from any speech recogniser
a.confidence      # "high"
a.clinical_term   # "Headache"
a.danger          # False
a.needs_review    # False
a.banners()       # [] or e.g. ["URGENT - possible danger sign ...", "LOW CONFIDENCE ..."]
```

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable demo.

## Lexicon format

JSON (a list of objects) or CSV with a header row:

| field | required | meaning |
|---|---|---|
| `phrase` | yes | what the patient says, in the local language or dialect |
| `meaning` | yes | plain-English meaning |
| `clinical_term` | yes | the term a clinician would record |
| `language` | no | language code, e.g. `yo`, `ha`, `sw` |
| `department` | no | context, e.g. `Antenatal` |
| `danger` | no | `true` if this phrase is a danger sign |

Check a lexicon for clashing entries before deploying:

```bash
clinguard validate my_lexicon.json
clinguard check my_lexicon.json "Mi ò le mí" --language yo
```

## Human-in-the-loop learning

```python
from clinguard import SuggestionStore

store = SuggestionStore("lexicon_reviews/")
sid = store.submit("Ese mi n wu", "My leg is swelling", "Leg swelling (edema)", language="yo")
store.review(sid, approve=True, reviewer="Native-speaker reviewer")   # reviewer name is required
lexicon = lexicon.extend(store.approved_entries("yo"))
```

Suggestions and decisions are stored in append-only JSON Lines files, so every change to the
lexicon can be traced to a person and a time.

## Used in

- **BeaGrace Health**: ClinGuard grew out of, and now powers the safety and lexicon layer of, BeaGrace Health, a voice app
that turns Ekiti Yoruba patient speech into structured nurse handoff notes, being piloted with
nurses in Nigeria (AI Ventures Accelerator · Technovation & UNICEF Generation Unlimited).
ClinGuard packages its safety and lexicon methods so any team can use them for any language.

Using ClinGuard in your project? Please open an issue to tell us. It helps guide the roadmap.

## Safety notice

ClinGuard is a toolkit, not a medical device. The built-in English danger keywords are a starter
list for illustration only. Every deployment must have its lexicon and danger rules reviewed and
approved by qualified clinicians, and a clinician must review AI-assisted output before it is used
in care. The example lexicon is illustrative and not clinically validated.

## Contributing

Issues and pull requests are welcome, especially lexicons and danger-sign rules for new
languages, reviewed by native speakers and clinicians.

## License

Apache License 2.0. Created by Shalini Sivasamy.
