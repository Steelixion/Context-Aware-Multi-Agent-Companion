"""Maps (visual fatigue level, acoustic emotion) -> system-prompt tone instructions."""

LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "ur": "Urdu",
              "ar": "Arabic", "hi": "Hindi", "zh": "Chinese", "pt": "Portuguese", "it": "Italian",
              "tr": "Turkish", "ru": "Russian", "fa": "Persian", "uk": "Ukrainian", "el": "Greek",
              "he": "Hebrew", "hy": "Armenian", "bn": "Bengali", "pa": "Punjabi", "gu": "Gujarati",
              "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "th": "Thai",
              "ka": "Georgian", "am": "Amharic", "ja": "Japanese", "ko": "Korean",
              "mr": "Marathi", "ne": "Nepali"}


def tone_instructions(level: str, emotion: str, fatigue: float) -> str:
    parts = []
    if level == "HIGH":
        parts.append("The driver is very drowsy. Say only one short sentence urging them to stop "
                     "somewhere safe and rest.")
    elif level == "MILD":   # Table 6.2: shorter, simpler, more engaging questions
        parts.append("The driver shows mild signs of fatigue. Use short, simple, energetic sentences, "
                     "ask an easy, interesting question to keep them mentally active, and gently "
                     "suggest a rest stop.")
    else:
        parts.append("The driver is alert. Be relaxed, friendly and conversational.")
    if emotion in ("angry", "stressed"):
        parts.append("They sound stressed or irritated: stay calm, slow, soothing; never argue; "
                     "do not add cognitive load.")
    elif emotion == "sad":
        parts.append("They sound low: be empathetic and supportive, not overly cheerful.")
    elif emotion == "happy":
        parts.append("They sound happy: match their positive energy.")
    return " ".join(parts)


def build_system_prompt(name: str, lang: str, level: str, emotion: str, fatigue: float,
                        memories: list[dict]) -> str:
    lang_name = LANG_NAMES.get(lang, lang)
    mem = ""
    if memories:
        mem = "\nWhat you remember about the driver (use naturally, never recite):\n" + \
              "\n".join(f"- {m['text']}" for m in memories)
    return (
        f"You are a voice companion in a car, speaking with {name}. Reply in {lang_name}. "
        "Responses are spoken aloud: at most 2 short sentences, no lists, no markdown, no emojis. "
        "Driving safety comes first; never ask the driver to look at a screen.\n"
        f"Driver state now: fatigue={fatigue:.2f} ({level}), voice emotion={emotion}.\n"
        f"{tone_instructions(level, emotion, fatigue)}{mem}"
    )
