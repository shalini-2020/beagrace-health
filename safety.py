"""
Danger-sign check for BeaGrace Health.

Runs WITHOUT the AI: it looks at (1) the glossary term that matched, (2) the
patient's Yoruba words, and (3) the English meaning, against the editable list
in danger_signs.json. If anything matches, the app shows a red URGENT banner,
even if the AI-written note says otherwise.
"""
import json
import re
import unicodedata
from pathlib import Path

DANGER_FILE = Path('danger_signs.json')


def _plain(text: str) -> str:
    """Lower-case, no tone marks, no punctuation (same idea as the glossary matcher)."""
    text = unicodedata.normalize('NFKD', text or '')
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    return ' ' + re.sub(r'\s+', ' ', re.sub(r"[^\w\s']", ' ', text)).lower().strip() + ' '


def _load() -> dict:
    try:
        return json.loads(DANGER_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'[safety] could not read {DANGER_FILE}: {e}')
        return {}


def check_danger(glossary_term: str = '', transcript: str = '', english_text: str = '',
                 keywords_key: str = 'yoruba_keywords') -> list:
    """Returns a list of reasons (empty list = no danger sign found)."""
    rules = _load()
    reasons = []
    terms = {t.lower() for t in rules.get('glossary_terms', [])}
    if glossary_term and glossary_term.lower() in terms:
        reasons.append(glossary_term)
    yoruba = _plain(transcript)
    for word in rules.get(keywords_key or '', []):
        if f' {_plain(word).strip()} ' in yoruba:
            reasons.append(f'"{word}"')
    english = _plain(english_text)
    for word in rules.get('english_keywords', []):
        if f' {_plain(word).strip()} ' in english:
            reasons.append(f'"{word}"')
    # keep order, drop duplicates
    return list(dict.fromkeys(reasons))


def danger_banner(reasons: list) -> str:
    if not reasons:
        return ''
    shown = ', '.join(reasons[:3])
    return f"""
    <div class="danger-banner" style="background:#c1440e; color:#fff; border-radius:12px; padding:16px 20px; margin:12px 0;">
        <div style="font-size:1.15rem; font-weight:700;">🚨 URGENT – possible danger sign. Escalate now.</div>
        <div style="font-size:0.9rem; margin-top:6px;">Detected: {shown}</div>
        <div style="font-size:0.8rem; margin-top:6px; opacity:0.9;">
            This safety check works independently of the AI note. Inform the senior nurse or doctor immediately.
        </div>
    </div>"""
