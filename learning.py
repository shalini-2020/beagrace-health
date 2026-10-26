"""
Learning glossary for BeaGrace Health.

1. A nurse submits a suggestion: what the patient said (the transcript) and what
   they actually meant, when the app didn't recognise it or got it wrong.
2. A native-speaker reviewer opens the password-protected Reviewer tab and
   approves (optionally correcting the wording) or rejects it.
3. Approved phrases are added to the glossary straight away and saved to the
   private dataset, so they are still there after the app restarts.
Nothing is added to the glossary without a reviewer's approval.
"""
import uuid
from typing import Optional

from usage_log import append_record, load_records

_cache = {'suggestions': None, 'reviews': None}


def _records(kind: str) -> list:
    if _cache[kind] is None:
        _cache[kind] = load_records(kind)
    return _cache[kind]


def submit_suggestion(transcript: str, meaning: str, clinical_term: str,
                      department: str, note_id: str = '', language: str = 'yo') -> str:
    transcript, meaning = (transcript or '').strip(), (meaning or '').strip()
    if not transcript or transcript.lower().startswith(('error', 'no speech')):
        return 'Generate a note first, so there is something to suggest.'
    if not meaning:
        return 'Please write what the patient meant (in English).'
    record = {
        'suggestion_id': uuid.uuid4().hex[:10],
        'note_id': note_id,
        'yoruba': transcript[:300],
        'meaning': meaning[:300],
        'clinical_term': (clinical_term or '').strip()[:120],
        'department': department or 'Not specified',
        'language': language or 'yo',
    }
    existing = _records('suggestions')   # load earlier ones first, so this one isn't counted twice
    append_record('suggestions', record)
    existing.append(record)
    return 'Thank you. Your suggestion was sent for review by a Yoruba speaker.'


def pending_suggestions() -> list:
    decided = {r['suggestion_id'] for r in _records('reviews')}
    return [s for s in _records('suggestions') if s['suggestion_id'] not in decided]


def review(suggestion_id: str, decision: str, yoruba: str = '', meaning: str = '',
           clinical_term: str = '', department: str = '', reviewer: str = '') -> Optional[str]:
    """decision: 'approved' or 'rejected'. Returns an error message or None."""
    match = [s for s in pending_suggestions() if s['suggestion_id'] == suggestion_id]
    if not match:
        return 'That suggestion was already reviewed.'
    s = match[0]
    record = {
        'suggestion_id': suggestion_id,
        'decision': decision,
        'reviewer': (reviewer or '').strip()[:60],
        'dialect_expression': (yoruba or s['yoruba']).strip(),
        'literal_meaning': (meaning or s['meaning']).strip(),
        'english_medical_term': (clinical_term or s['clinical_term'] or meaning or s['meaning']).strip(),
        'department': department or s['department'],
        'language': s.get('language', 'yo'),
    }
    if decision == 'approved' and not record['dialect_expression']:
        return 'The Yoruba phrase cannot be empty.'
    existing = _records('reviews')
    append_record('reviews', record)
    existing.append(record)
    return None


def learned_entries(language: str = 'yo') -> list:
    """Approved phrases for one language, in the same shape as glossary.json entries."""
    return [{
        'dialect_expression': r['dialect_expression'],
        'standard_yoruba': '',
        'english_medical_term': r['english_medical_term'],
        'literal_meaning': r['literal_meaning'],
        'department': r.get('department', 'Not specified'),
        'source': 'learned',
    } for r in _records('reviews')
        if r.get('decision') == 'approved' and r.get('language', 'yo') == language]
