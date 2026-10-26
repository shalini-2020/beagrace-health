"""
Danger-sign check for BeaGrace Health.

Runs WITHOUT the AI (uses the ClinGuard package): it looks at (1) the glossary term that matched, (2) the
patient's Yoruba words, and (3) the English meaning, against the editable list
in danger_signs.json. If anything matches, the app shows a red URGENT banner,
even if the AI-written note says otherwise.
"""
import json
from pathlib import Path

from clinguard import DangerRules, Entry

DANGER_FILE = Path('danger_signs.json')


def _load() -> dict:
    try:
        return json.loads(DANGER_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'[safety] could not read {DANGER_FILE}: {e}')
        return {}


def check_danger(glossary_term: str = '', transcript: str = '', english_text: str = '',
                 keywords_key: str = 'yoruba_keywords') -> list:
    """Returns a list of reasons (empty list = no danger sign found).
    The rule check itself is ClinGuard's DangerRules (pip install clinguard)."""
    rules = _load()
    checker = DangerRules(
        clinical_terms=rules.get('glossary_terms', []),
        english_keywords=rules.get('english_keywords', []),
        local_keywords={'local': rules.get(keywords_key or '', [])},
    )
    entry = Entry(phrase='', meaning='', clinical_term=glossary_term) if glossary_term else None
    return checker.check(said=transcript, meaning=english_text, entry=entry, language='local')


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
