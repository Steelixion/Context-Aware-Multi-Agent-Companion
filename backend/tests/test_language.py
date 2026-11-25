"""Language ID from transcripts (replaces Whisper's built-in language detection)."""
import pytest

from app.audio.language import detect_language


@pytest.mark.parametrize("text, lang", [
    # Latin script: function-word vote (lowercase, no punctuation, like the CTC model's output)
    ("what is the weather like today", "en"),
    ("i am really tired can you play some music", "en"),
    ("hello", "en"),
    ("hola cómo estás", "es"),
    ("quiero escuchar música por favor", "es"),
    ("mañana", "es"),
    ("bonjour je suis fatigué", "fr"),
    ("est-ce que tu peux mettre de la musique", "fr"),
    ("ich bin sehr müde", "de"),
    ("kannst du bitte musik spielen", "de"),
    ("eu estou muito cansado", "pt"),
    ("ciao come stai oggi", "it"),
    ("sono molto stanco", "it"),
    ("merhaba nasılsın", "tr"),
    ("çok yorgunum ama iyi", "tr"),
    # non-Latin scripts
    ("آپ کیسے ہیں", "ur"),
    ("میں بہت تھکا ہوا ہوں", "ur"),
    ("كيف حالك اليوم", "ar"),
    ("حال شما چطور است", "fa"),
    ("आप कैसे हैं", "hi"),
    ("你好吗", "zh"),
    ("こんにちは元気ですか", "ja"),
    ("안녕하세요", "ko"),
    ("как дела", "ru"),
    ("як справи і що нового", "uk"),
    ("γεια σου τι κάνεις", "el"),
    ("שלום מה שלומך", "he"),
])
def test_detects(text, lang):
    assert detect_language(text) == lang


@pytest.mark.parametrize("text", [
    "", "   ", "123 456", "ok", "no",                      # nothing to go on / es-pt tie
    "habari yako rafiki yangu leo tunaenda sokoni",          # Swahili: not in the Latin lexicon
])
def test_unsure_returns_none(text):
    assert detect_language(text) is None


def test_session_language_breaks_arabic_script_ties():
    # no Urdu-only or Arabic-only letters: the session language decides between ur and fa
    assert detect_language("کیا", current="ur") == "ur"
    assert detect_language("کیا", current="fa") == "fa"
    # Urdu-only letters win regardless of the session language
    assert detect_language("ٹھیک ہے", current="ar") == "ur"


def test_devanagari_keeps_marathi_session():
    assert detect_language("तुम्ही कसे आहात", current="mr") == "mr"
    assert detect_language("तुम्ही कसे आहात", current="en") == "hi"


def test_mixed_script_follows_majority():
    assert detect_language("ok آپ کیسے ہیں") == "ur"
    assert detect_language("what is the time 你") == "en"
