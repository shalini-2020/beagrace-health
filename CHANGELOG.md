# Changelog

Deployment notes for the BeaGrace Space. Each interaction in the private logs also
records the app version and a code fingerprint (from release 2026.11.06), so results
can be tied to the code that produced them.

## Pilot reporting period

- The pilot reporting period starts at **2026-11-06 19:00 UTC** (8:00 PM in Nigeria),
  just before nurses began real use. Every record before that was developer or tester
  activity (below). Those records are kept unchanged in the private logs and are left
  out of the Impact page and PDF reports by the reporting period, not by deletion.
  Set as the default `PILOT_START` in `impact.py`.

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
- **~18:45 UTC** - release **2026.11.06.2**: blinded nurse evaluation (internals hidden
  from evaluators, original recording, optional LLM-only comparison condition); clearer
  AI-fallback labels; pilot reporting period starts 19:00 UTC; this changelog.

## 2026-11-02

- Release `2026.11.02`: ClinGuard 0.1.2, privacy-safe logging, structured nurse
  feedback, nurse codes. Interpretation path and versions recorded from this release.

## 2026-10-28

- 1 note generated: a test, not pilot use.
