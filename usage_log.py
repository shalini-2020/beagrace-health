"""
Anonymous usage log and nurse feedback for BeaGrace Health.

What is recorded (no audio, no transcript, no patient details):
  usage    - one record per interaction, each with its own note_id (also for
             no-speech and error outcomes): time, department, language,
             outcome, confidence score and category, glossary match id and
             term (only when matched), danger-sign rule ids, urgency, and the
             app / ClinGuard / glossary / danger-rule versions
  feedback - which note it is about, the quick rating (correct / needs changes /
             not sure), optional answers on meaning, usefulness and danger-sign
             handling, an optional correction comment and optional nurse code
             (obvious identifiers such as phone numbers, emails and dates are
             masked before saving)
  content  - OFF by default. Only when LOG_CONTENT=true: the transcript, the
             English meaning, the AI fallback translation and the final note,
             under the same note_id, in a SEPARATE folder (content_logs/) that
             syncs only to its own private dataset (CONTENT_LOG_REPO). Turn this
             on only after the pilot site has approved content collection in
             writing and access, retention and deletion rules are in place.
  suggestions / reviews - glossary phrases a nurse chose to submit, and the
             reviewer's approve / reject decision (see learning.py)
  evaluations - independent nurse evaluation of standardised, simulated scenarios
             (see nurse_eval.py). Kept apart from usage and feedback; because the
             scenarios are scripted, their transcripts and notes are kept for scoring.

On Hugging Face, records are synced every 10 minutes to a PRIVATE dataset
(default: ss-shalini-2021/beagrace-usage) so they survive app restarts.
This needs an HF_TOKEN secret (write access) in the Space settings.
Running locally, records are only written to the usage_logs/ folder.
"""
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_DIR = Path('usage_logs')
LOG_DIR.mkdir(exist_ok=True)

# A new file per app start, so a restart never overwrites earlier records
_RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
USAGE_FILE = LOG_DIR / f'usage_{_RUN_ID}.jsonl'
FEEDBACK_FILE = LOG_DIR / f'feedback_{_RUN_ID}.jsonl'
SUGGESTIONS_FILE = LOG_DIR / f'suggestions_{_RUN_ID}.jsonl'
REVIEWS_FILE = LOG_DIR / f'reviews_{_RUN_ID}.jsonl'
EVALUATIONS_FILE = LOG_DIR / f'evaluations_{_RUN_ID}.jsonl'

LOG_REPO = os.environ.get('USAGE_LOG_REPO', 'ss-shalini-2021/beagrace-usage')

# Clinical content logging: OFF unless explicitly enabled (see docstring)
LOG_CONTENT = os.environ.get('LOG_CONTENT', 'false').strip().lower() in ('1', 'true', 'yes')
CONTENT_LOG_REPO = os.environ.get('CONTENT_LOG_REPO', '')
CONTENT_DIR = Path('content_logs')
CONTENT_FILE = CONTENT_DIR / f'content_{_RUN_ID}.jsonl'

_lock = threading.Lock()
_scheduler = None

if os.environ.get('SPACE_ID') and os.environ.get('HF_TOKEN'):
    try:
        from huggingface_hub import CommitScheduler
        _scheduler = CommitScheduler(
            repo_id=LOG_REPO,
            repo_type='dataset',
            folder_path=LOG_DIR,
            path_in_repo='logs',
            every=10,          # minutes
            private=True,
            token=os.environ['HF_TOKEN'],
        )
        print(f'[usage log] syncing to private dataset {LOG_REPO}')
    except Exception as e:  # logging must never break the app
        print(f'[usage log] sync disabled: {e}')
else:
    print('[usage log] local only (no HF_TOKEN or not running on a Space)')

_content_scheduler = None
if LOG_CONTENT:
    CONTENT_DIR.mkdir(exist_ok=True)
    if os.environ.get('SPACE_ID') and os.environ.get('HF_TOKEN') and CONTENT_LOG_REPO:
        try:
            from huggingface_hub import CommitScheduler
            _content_scheduler = CommitScheduler(
                repo_id=CONTENT_LOG_REPO, repo_type='dataset', folder_path=CONTENT_DIR,
                path_in_repo='content', every=10, private=True, token=os.environ['HF_TOKEN'])
            print(f'[content log] ON, syncing to private dataset {CONTENT_LOG_REPO}')
        except Exception as e:
            print(f'[content log] ON, local only, sync failed: {e}')
    else:
        print('[content log] ON, local only (set CONTENT_LOG_REPO to keep records)')
else:
    print('[content log] OFF (LOG_CONTENT is not enabled)')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def new_interaction_id() -> str:
    return uuid.uuid4().hex[:12]


def _append(path: Path, record: dict, scheduler=None) -> None:
    scheduler = scheduler if scheduler is not None else _scheduler
    try:
        lock = scheduler.lock if scheduler else _lock
        with lock:
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f'[usage log] could not write record: {e}')


def log_usage(department: str, outcome: str, glossary_match: bool = False,
              confidence: float = 0.0, symptom: Optional[str] = None,
              urgency: Optional[str] = None, danger_sign: bool = False,
              note_id: Optional[str] = None, language: Optional[str] = None,
              confidence_level: Optional[str] = None, match_id: Optional[str] = None,
              danger_rule_ids: Optional[list] = None, error_stage: Optional[str] = None,
              versions: Optional[dict] = None, nurse_code: Optional[str] = None,
              speech_model: Optional[str] = None, facility_code: Optional[str] = None) -> str:
    """Record one interaction (note / no_speech / error). Returns its note_id,
    which links feedback and (if enabled) content records to it."""
    note_id = note_id or new_interaction_id()
    _append(USAGE_FILE, {
        'note_id': note_id,
        'time_utc': _now(),
        'department': department or 'Not specified',
        'language': language,
        'nurse_code': nurse_code,                # optional pseudonymous code, never a name
        'facility_code': facility_code,          # optional code like FAC001, never a facility name
        'outcome': outcome,                      # note / no_speech / error
        'error_stage': error_stage,              # transcription / translation / note / None
        'glossary_match': glossary_match,
        'confidence': round(confidence * 100),
        'confidence_level': confidence_level,    # high / partial / low (AI fallback) / none
        'match_id': match_id if glossary_match else None,
        'symptom': symptom if glossary_match else None,
        'urgency': urgency,
        'danger_sign': danger_sign,
        'danger_rule_ids': danger_rule_ids or [],
        'speech_model': speech_model,
        **(versions or {}),                      # app_version, clinguard_version, glossary_version, rules_version
    })
    return note_id


def log_content(note_id: str, **fields) -> None:
    """Clinical content for approved evaluation only. Does nothing unless LOG_CONTENT is on."""
    if not LOG_CONTENT or not note_id:
        return
    _append(CONTENT_FILE, {'note_id': note_id, 'time_utc': _now(), **fields},
            scheduler=_content_scheduler or False)


_REDACTIONS = [
    (re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'), '[email]'),
    (re.compile(r'\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\b'), '[date]'),
    (re.compile(r'\+?\d[\d\s-]{5,}\d'), '[number]'),
    (re.compile(r'\b\d{4,}\b'), '[number]'),
]


def redact_comment(text: str) -> str:
    """Mask obvious identifiers (emails, dates, phone / hospital numbers).
    Names cannot be detected reliably, so the comment box also asks nurses
    not to include patient details."""
    text = (text or '').strip()
    for pattern, mask in _REDACTIONS:
        text = pattern.sub(mask, text)
    return text[:300]


FEEDBACK_ANSWERS = {
    'meaning_preserved': ('yes', 'mostly', 'no', 'unable_to_assess'),
    'handoff_useful': ('yes', 'partly', 'no', 'unable_to_assess'),
    'danger_handling': ('yes', 'missed_warning', 'false_warning', 'not_applicable', 'unsure'),
}


def log_feedback(note_id: str, rating: str, comment: str = '', details: Optional[dict] = None,
                 nurse_code: Optional[str] = None) -> None:
    """rating: the quick correct / needs_changes / not_sure tap.
    details: optional structured answers (see FEEDBACK_ANSWERS); unanswered = None."""
    details = details or {}
    record = {
        'note_id': note_id,
        'time_utc': _now(),
        'nurse_code': nurse_code,
        'rating': rating,                        # correct / needs_changes / not_sure
        'comment': redact_comment(comment),      # what expression or wording needs correction
    }
    for key, allowed in FEEDBACK_ANSWERS.items():
        value = details.get(key)
        record[key] = value if value in allowed else None
    _append(FEEDBACK_FILE, record)


def append_record(kind: str, record: dict) -> None:
    """kind: 'suggestions', 'reviews' or 'evaluations'."""
    path = {'suggestions': SUGGESTIONS_FILE, 'evaluations': EVALUATIONS_FILE}.get(kind, REVIEWS_FILE)
    _append(path, {'time_utc': _now(), **record})


def load_records(kind: str) -> list:
    """All records of one kind: this run's local file plus earlier runs synced
    to the private dataset (so approvals survive restarts)."""
    records, seen_files = [], set()
    for path in sorted(LOG_DIR.glob(f'{kind}_*.jsonl')):
        seen_files.add(path.name)
        records += _read_jsonl(path)
    if os.environ.get('SPACE_ID') and os.environ.get('HF_TOKEN'):
        try:
            from huggingface_hub import HfApi, hf_hub_download
            token = os.environ['HF_TOKEN']
            for name in HfApi(token=token).list_repo_files(LOG_REPO, repo_type='dataset'):
                base = name.split('/')[-1]
                if name.startswith(f'logs/{kind}_') and base not in seen_files:
                    local = hf_hub_download(LOG_REPO, name, repo_type='dataset', token=token)
                    records += _read_jsonl(Path(local))
        except Exception as e:
            print(f'[usage log] could not load earlier {kind}: {e}')
    return records


def _read_jsonl(path: Path) -> list:
    out = []
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                out.append(json.loads(line))
    except Exception as e:
        print(f'[usage log] could not read {path}: {e}')
    return out
