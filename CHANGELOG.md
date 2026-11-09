# Changelog

Deployment notes for the BeaGrace Space. Each interaction in the private logs also
records the app version and a code fingerprint (from release 2026.11.06), so results
can be tied to the code that produced them.

## Pilot reporting period

- The pilot reporting period starts at **2026-11-06 19:00 UTC** (8:00 PM in Nigeria),
  just before nurses began pilot use. Every record before that was developer or tester
  activity (below). Those records are kept unchanged in the private logs and are left
  out of the Impact page and PDF reports by the reporting period, not by deletion.
  Set as the default `PILOT_START` in `impact.py`.

## Type of use

- From the start of the pilot (2026-11-06 19:00 UTC): **simulated use**. Nurses act out
  patient scenarios at one facility (FAC001); no real patients. Interactions before
  release 2026.11.06.3 do not carry the `use_mode` field; this note covers them.
- From release 2026.11.06.3, every interaction records `use_mode` (`simulated` or
  `clinical`, from the `PILOT_MODE` setting). Change `PILOT_MODE` to `clinical` only when
  real patient use begins, and record the date here.

## 2026-11-06

- **14:10-14:19 UTC** - 14 attempts failed at the transcription stage (a pilot tester
  uploading WhatsApp voice notes). Cause: `librosa` was unpinned; the 2 Nov rebuild
  installed librosa 1.0, which removed the fallback that decoded AAC / MP4 audio.
  No nurse pilot use occurred between 29 Oct and 5 Nov, so no real use was affected.
- **~14:45 UTC** - fix deployed: ffmpeg fallback for audio decoding, `librosa` pinned
  below 1.0, `ffmpeg` added to `packages.txt`. All 6 later attempts (15:09-15:51 UTC)
  produced notes.
- The 20 attempts on 6 Nov were troubleshooting and testing, logged under department
  "Not specified" because the testing department was not selected.
- Same day, also deployed under the `2026.11.02` label: matching when a patient repeats
  a phrase; a WhatsApp voice-note upload box; the correct Ekiti Yoruba spelling from the
  glossary shown next to the transcript; Impact page redesign.
- **17:31 UTC** - release **2026.11.06**: dated PDF report, facility codes, independent
  nurse evaluation, version and code fingerprint recorded with every interaction.
- **After the evening session** - release **2026.11.06.3**: department, nurse code and
  facility code required for every real-use note (testing department exempt); facility
  code remembered on each phone; copy button on the note; `use_mode` recorded with every
  interaction; Impact page and PDF labelled as a simulated-use pilot; readable tips text.
- 19:12 and 19:31 UTC - first two pilot notes. The second was generated without
  department, nurse or facility code (fields were optional then).
- **~18:45 UTC** - release **2026.11.06.2**: blinded nurse evaluation (internals hidden
  from evaluators, original recording, optional LLM-only comparison condition); clearer
  AI-fallback labels; pilot reporting period starts 19:00 UTC; this changelog.

## 2026-11-02

- Release `2026.11.02`: ClinGuard 0.1.2, privacy-safe logging, structured nurse
  feedback, nurse codes. Interpretation path and versions recorded from this release.

## 2026-10-28

- 1 note generated: a test, not pilot use.
