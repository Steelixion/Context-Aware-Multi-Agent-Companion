"""Maps (visual fatigue level, acoustic emotion) -> system-prompt tone instructions."""

LANG_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "ur": "Urdu",
              "ar": "Arabic", "hi": "Hindi", "zh": "Chinese", "pt": "Portuguese", "it": "Italian",
              "tr": "Turkish", "ru": "Russian"}


def tone_instructions(level: str, emotion: str, fatigue: float) -> str:
    parts = []
    if level in ("HIGH", "MODERATE"):
        parts.append("The driver shows signs of fatigue. Use short, energetic, engaging sentences, "
                     "ask simple questions to keep them mentally active, and gently suggest a rest stop.")
    elif level == "MILD":
        parts.append("The driver is slightly tired. Be lively and warm; keep it light.")
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
