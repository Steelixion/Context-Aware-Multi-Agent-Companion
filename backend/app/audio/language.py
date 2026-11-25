"""Text-based language identification for ASR transcripts.

The Omnilingual ASR CTC model transcribes 1600+ languages but, unlike Whisper, does not say
which language it heard. We infer it from the transcript instead:
  * the Unicode script settles most non-Latin languages (Urdu vs Arabic vs Persian by letters
    that only one of them uses), and
  * a small function-word vote separates the common Latin-script languages.
detect_language() returns None when unsure, so the caller keeps the session's current language.
"""
from __future__ import annotations

import re
from collections import Counter

# (first code point, last code point, language) for scripts that map (mostly) to one language
_SCRIPTS = [
    (0x0370, 0x03FF, "el"), (0x0400, 0x04FF, "ru"), (0x0530, 0x058F, "hy"),
    (0x0590, 0x05FF, "he"), (0x0600, 0x06FF, "ar"), (0x0750, 0x077F, "ar"),
    (0xFB50, 0xFDFF, "ar"), (0xFE70, 0xFEFF, "ar"), (0x0900, 0x097F, "hi"),
    (0x0980, 0x09FF, "bn"), (0x0A00, 0x0A7F, "pa"), (0x0A80, 0x0AFF, "gu"),
    (0x0B80, 0x0BFF, "ta"), (0x0C00, 0x0C7F, "te"), (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"), (0x0E00, 0x0E7F, "th"), (0x10A0, 0x10FF, "ka"),
    (0x1200, 0x139F, "am"), (0x3040, 0x30FF, "ja"), (0x3400, 0x4DBF, "zh"),
    (0x4E00, 0x9FFF, "zh"), (0x1100, 0x11FF, "ko"), (0xAC00, 0xD7AF, "ko"),
]

# Arabic-script letters that only one of Urdu / Arabic / Persian uses
_URDU_ONLY = set("ٹڈڑںےۓۂۃ")            # tteh, ddal, rreh, noon ghunna, yeh barree, ...
_ARABIC_ONLY = set("ةكيىإأؤ")            # teh marbuta, Arabic kaf/yeh, alef maksura, hamza seats
_PERSIAN_OR_URDU = set("پچژگکیہھ")       # shared by Persian and Urdu, absent from Arabic

# Function words / greetings that dominate short conversational utterances.
_WORDS = {
    "en": """the is are was were i you it to and of what how my me this that do don't have can
        with for not we your hello hi hey yes please thanks thank it's i'm im would could will
        there about just like so but really need want going where when why who an on at am be
        tired play music""",
    "es": """el la los las que de y es en un una por para con no yo tú tu mi me estoy está estás
        cómo qué hola gracias sí muy pero quiero puedes tengo del al se lo su hay bien buenos
        días noches también ahora dónde cuándo música ella él usted cansado""",
    "fr": """le la les de des du et est je tu il elle nous vous que qui pas ne un une pour avec
        dans sur mon ma mes suis c'est j'ai bonjour salut merci oui très mais au aux ce cette
        comment quoi où ça fatigué musique peux veux""",
    "de": """der die das und ist ich du nicht ein eine einen mit zu von mein meine wie was bin
        hallo danke ja nein bitte auf für es sie wir ihr den dem auch aber sehr müde musik
        kannst möchte guten morgen tag haben habe geht wo warum""",
    "pt": """o os as um uma de do da dos das que e é não eu você meu minha em no na com para
        por muito estou está olá obrigado obrigada sim mas isso isto como onde quando também
        agora bom dia música cansado quero posso tenho vamos""",
    "it": """il lo la gli le di che e è non io tu mio mia un una per con sono sei ciao grazie
        sì ma come cosa dove quando anche ora molto bene buongiorno musica stanco voglio posso
        ho hai della del nel alla questo questa""",
    "tr": """bir ve bu ne ben sen o mi mı çok var yok değil ile için ama nasıl merhaba teşekkür
        ederim evet hayır lütfen iyi benim senin şey gibi daha sonra şimdi nerede neden yorgun
        müzik istiyorum misin musun""",
}
_LEXICON = {lang: frozenset(words.split()) for lang, words in _WORDS.items()}
# characters that (among the languages above) only one language uses
_MARKERS = {"ñ": "es", "¿": "es", "¡": "es", "ß": "de", "ã": "pt", "õ": "pt",
            "ğ": "tr", "ş": "tr", "ı": "tr", "œ": "fr"}
_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def _is_latin(cp: int) -> bool:
    return cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF


def detect_language(text: str, current: str | None = None) -> str | None:
    """ISO 639-1 code of the transcript's language, or None if it can't tell.
    `current` (the session language) breaks ties the text alone can't settle."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    scripts: Counter = Counter()
    latin = 0
    for c in letters:
        cp = ord(c)
        if _is_latin(cp):
            latin += 1
            continue
        for lo, hi, lang in _SCRIPTS:
            if lo <= cp <= hi:
                scripts[lang] += 1
                break
    if scripts and sum(scripts.values()) >= latin:
        lang = scripts.most_common(1)[0][0]
        if lang == "zh" and scripts["ja"]:
            return "ja"                       # kanji + kana -> Japanese
        if lang == "ar":
            return _arabic_script(set(text), current)
        if lang == "ru" and set(text) & set("іїєґІЇЄҐ"):
            return "uk"
        if lang == "hi" and current in ("mr", "ne"):
            return current                    # Devanagari is shared with Marathi / Nepali
        return lang
    return _latin_script(text)


def _arabic_script(chars: set, current: str | None) -> str:
    if chars & _URDU_ONLY:
        return "ur"
    if chars & _ARABIC_ONLY:
        return "ar"
    if current in ("ur", "fa"):
        return current
    return "fa" if chars & _PERSIAN_OR_URDU else "ar"


def _latin_script(text: str) -> str | None:
    low = text.lower()
    tokens = _TOKEN.findall(low)
    if not tokens:
        return None
    score: Counter = Counter()
    for tok in tokens:
        for lang, lex in _LEXICON.items():
            if tok in lex:
                score[lang] += 1
    for ch, lang in _MARKERS.items():
        if ch in low:
            score[lang] += 2
    if not score:
        return None
    (best, s1), *rest = score.most_common()
    s2 = rest[0][1] if rest else 0
    if s1 - s2 < 1:
        return None                           # tie, e.g. "no" (es/pt)
    if len(tokens) <= 2:
        return best                           # "hello", "hola", "merci"
    if s1 >= 2 and s1 / len(tokens) >= 0.2:
        return best
    return None
