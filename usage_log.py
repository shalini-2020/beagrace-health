"""
Anonymous usage log and nurse feedback for BeaGrace Health.

What is recorded (no audio, no transcript, no patient details):
  usage    - time, department, outcome, whether the glossary matched, the
             confidence score, the glossary term (only when matched) and urgency
  feedback - which note it is about, correct / needs changes, optional comment
  suggestions / reviews - glossary phrases a nurse chose to submit, and the
             reviewer's approve / reject decision (see learning.py)

On Hugging Face, records are synced every 10 minutes to a PRIVATE dataset
(default: ss-shalini-2021/beagrace-usage) so they survive app restarts.
This needs an HF_TOKEN secret (write access) in the Space settings.
Running locally, records are only written to the usage_logs/ folder.
"""
import json
import os
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

LOG_REPO = os.environ.get('USAGE_LOG_REPO', 'ss-shalini-2021/beagrace-usage')

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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _append(path: Path, record: dict) -> None:
    try:
        lock = _scheduler.lock if _scheduler else _lock
        with lock:
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f'[usage log] could not write record: {e}')


def log_usage(department: str, outcome: str, glossary_match: bool = False,
              confidence: float = 0.0, symptom: Optional[str] = None,
              urgency: Optional[str] = None, danger_sign: bool = False) -> str:
    """Record one use of the app. Returns the note id used to link feedback."""
    note_id = uuid.uuid4().hex[:12]
    _append(USAGE_FILE, {
        'note_id': note_id,
        'time_utc': _now(),
        'department': department or 'Not specified',
        'outcome': outcome,                      # note / no_speech / error
        'glossary_match': glossary_match,
        'confidence': round(confidence * 100),
        'symptom': symptom if glossary_match else None,
        'urgency': urgency,
        'danger_sign': danger_sign,
    })
    return note_id


def log_feedback(note_id: str, rating: str, comment: str = '') -> None:
    _append(FEEDBACK_FILE, {
        'note_id': note_id,
        'time_utc': _now(),
        'rating': rating,                        # correct / needs_changes / not_sure
        'comment': (comment or '').strip()[:300],
    })


def append_record(kind: str, record: dict) -> None:
    """kind: 'suggestions' or 'reviews'."""
    path = SUGGESTIONS_FILE if kind == 'suggestions' else REVIEWS_FILE
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
