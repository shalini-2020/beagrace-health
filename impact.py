"""
Public Impact page: live totals from the anonymous usage log.

Only real pilot use is counted:
  - records on or after PILOT_START (default 2026-10-27, can be changed with
    a PILOT_START variable in the Space settings, format YYYY-MM-DD)
  - the 'Testing (not real use)' department is always left out
Only totals are shown. No individual notes, phrases or comments.
"""
import os
import time
from collections import Counter

from usage_log import load_records

PILOT_START = os.environ.get('PILOT_START', '2026-10-27')
TEST_DEPARTMENT = 'Testing (not real use)'
_CACHE = {'at': 0.0, 'stats': None}
CACHE_SECONDS = 600


def _is_real(record: dict) -> bool:
    return (record.get('time_utc', '') >= PILOT_START
            and record.get('department') != TEST_DEPARTMENT)


def compute_stats(glossary_size: int) -> dict:
    if _CACHE['stats'] and time.time() - _CACHE['at'] < CACHE_SECONDS:
        stats = dict(_CACHE['stats'])
        stats['glossary_size'] = glossary_size
        return stats
    usage = [r for r in load_records('usage') if _is_real(r)]
    notes = [r for r in usage if r.get('outcome') == 'note']
    note_ids = {r['note_id'] for r in notes}
    feedback = [f for f in load_records('feedback') if f.get('note_id') in note_ids]
    ratings = Counter(f.get('rating') for f in feedback)
    rated = ratings['correct'] + ratings['needs_changes']   # 'not sure' is not counted either way
    reviews = [r for r in load_records('reviews')
               if r.get('decision') == 'approved' and r.get('time_utc', '') >= PILOT_START]
    departments = Counter(r.get('department', 'Not specified') for r in notes
                          if r.get('department') not in (None, 'Not specified'))
    days = sorted({r['time_utc'][:10] for r in notes})
    stats = {
        'notes': len(notes),
        'days_active': len(days),
        'first_day': days[0] if days else None,
        'departments': departments.most_common(),
        'glossary_match_pct': round(100 * sum(1 for r in notes if r.get('glossary_match')) / len(notes)) if notes else None,
        'rated': rated,
        'correct_pct': round(100 * ratings['correct'] / rated) if rated else None,
        'danger_alerts': sum(1 for r in notes if r.get('danger_sign')),
        'phrases_learned': len(reviews),
    }
    _CACHE.update(at=time.time(), stats=stats)
    stats = dict(stats)
    stats['glossary_size'] = glossary_size
    return stats


def _card(number, label, note=''):
    note_html = f'<div class="impact-note">{note}</div>' if note else ''
    return (f'<div class="impact-card"><div class="impact-number">{number}</div>'
            f'<div class="impact-label">{label}</div>{note_html}</div>')


def impact_html(stats: dict) -> str:
    dash = '–'
    correct = f"{stats['correct_pct']}%" if stats['correct_pct'] is not None else dash
    match = f"{stats['glossary_match_pct']}%" if stats['glossary_match_pct'] is not None else dash
    cards = ''.join([
        _card(stats['notes'], 'Clinical notes generated', f"over {stats['days_active']} day(s)" if stats['notes'] else 'pilot starting'),
        _card(correct, 'Rated correct by nurses', f"{stats['rated']} rated" if stats['rated'] else 'no ratings yet'),
        _card(match, 'Recognised from the dialect glossary'),
        _card(stats['danger_alerts'], 'Danger signs flagged for escalation'),
        _card(stats['glossary_size'], 'Phrases in the glossary', f"{stats['phrases_learned']} learned from nurses"),
        _card(len(stats['departments']), 'Departments using it'),
    ])
    if stats['departments']:
        dept_rows = ''.join(f'<li><b>{name}</b> · {count} note{"s" if count != 1 else ""}</li>'
                            for name, count in stats['departments'])
        dept_html = f'<div class="impact-depts"><b>By department</b><ul>{dept_rows}</ul></div>'
    else:
        dept_html = ''
    return f"""
    <div class="impact-wrap">
        <div class="impact-head">
            <div class="impact-title">📊 Impact so far</div>
            <div class="impact-sub">Live totals from the nurse pilot (since {PILOT_START}).
            Updated about every 10 minutes. Test use is excluded, and no patient information is shown.</div>
        </div>
        <div class="impact-grid">{cards}</div>
        {dept_html}
    </div>"""
