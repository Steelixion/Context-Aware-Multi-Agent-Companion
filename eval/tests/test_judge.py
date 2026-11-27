"""Judge plumbing without any network: prompt shape, verdict parsing, configuration rules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import judge as J  # noqa: E402

CFG = {"provider": "groq", "models": {"groq": "openai/gpt-oss-120b", "gemini": "gemini-2.5-flash"},
       "api_key_env": {"groq": "TEST_GROQ_KEY", "gemini": "TEST_GEMINI_KEY"},
       "family": {"groq": "openai", "gemini": "google"}, "runs": 3}


def test_parse_takes_last_json_verdict():
    txt = 'Reasoning: the reply invents 32 psi. {"x": 1}\n{"question_id": "grounded", "verdict": "No", "evidence": "32 psi"}'
    v = J.parse_verdict(txt, "grounded")
    assert v == {"question_id": "grounded", "verdict": "no", "evidence": "32 psi"}
    assert J.parse_verdict("no json here", "grounded") is None
    assert J.parse_verdict('{"verdict": "maybe"}', "grounded") is None


def test_prompt_is_one_case_one_question():
    p = J.build_prompt("claims_alert", {"assistant_reply": "You're fine!"})
    assert "ONE reply" in p and "claims_alert" in p and p.count("QUESTION") == 1
    assert '"verdict": "yes" or "no"' in p


def test_no_key_means_not_run(monkeypatch):
    monkeypatch.delenv("TEST_GROQ_KEY", raising=False)
    cfg, why = J.JudgeConfig.from_thresholds(CFG, "qwen")
    assert cfg is None and why.startswith(J.NOT_CONFIGURED)


def test_same_family_judge_is_refused(monkeypatch):
    monkeypatch.setenv("TEST_GEMINI_KEY", "x")
    cfg, why = J.JudgeConfig.from_thresholds(CFG | {"provider": "gemini"}, "google")
    assert cfg is None and "same as the app" in why
    cfg, why = J.JudgeConfig.from_thresholds(CFG | {"provider": "gemini"}, "qwen")
    assert cfg is not None and cfg.model == "gemini-2.5-flash" and cfg.runs == 3


def test_provider_none():
    cfg, why = J.JudgeConfig.from_thresholds(CFG | {"provider": "none"}, "qwen")
    assert cfg is None and why == J.NOT_CONFIGURED


def test_ask_majority_with_fake_transport(monkeypatch):
    monkeypatch.setenv("TEST_GROQ_KEY", "x")
    cfg, _ = J.JudgeConfig.from_thresholds(CFG, "qwen")
    j = J.Judge(cfg)
    answers = iter(['ok {"question_id":"safe","verdict":"yes","evidence":"a"}',
                    'ok {"question_id":"safe","verdict":"no","evidence":"b"}',
                    'ok {"question_id":"safe","verdict":"yes","evidence":"c"}'])
    monkeypatch.setattr(j, "_call", lambda prompt: next(answers))
    r = j.ask("safe", {"assistant_reply": "Take a break soon."})
    assert r["majority"] == "yes" and round(r["agreement"], 3) == 0.667 and r["passed"] is True
    assert [x["verdict"] for x in r["runs"]] == ["yes", "no", "yes"]
