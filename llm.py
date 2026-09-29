"""OpenAI 호환 엔드포인트 호출 래퍼 + 배관 점검용 가짜 모델"""
import json, os, time, random, urllib.request, urllib.error

BASE = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
KEY = os.environ.get("LLM_API_KEY", "")
MODELS = {
    "cheap":     os.environ.get("MODEL_CHEAP", "gpt-4.1-mini"),
    "expensive": os.environ.get("MODEL_EXPENSIVE", "gpt-4.1"),
    "rewriter":  os.environ.get("MODEL_REWRITER", os.environ.get("MODEL_CHEAP", "gpt-4.1-mini")),
    "judge":     os.environ.get("MODEL_JUDGE", os.environ.get("MODEL_EXPENSIVE", "gpt-4.1")),
}
FAKE = False                      # run_pilot.py 가 --fake 일 때 True 로 바꾼다
_rng = random.Random(20260922)

USAGE = {"calls": 0, "tokens_in": 0, "tokens_out": 0}


def chat(model_key, messages, temperature=0.0, item=None, tag=""):
    """returns (text, tokens_in, tokens_out)"""
    model = MODELS[model_key]
    if FAKE:
        t, a, b = _fake(model_key, messages, item, tag)
    else:
        t, a, b = _real(model, messages, temperature)
    USAGE["calls"] += 1
    USAGE["tokens_in"] += a
    USAGE["tokens_out"] += b
    return t, a, b


def _real(model, messages, temperature, retries=5):
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                d = json.loads(r.read())
            u = d.get("usage", {})
            return (d["choices"][0]["message"]["content"] or "",
                    u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        except urllib.error.HTTPError as e:
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and a < retries - 1:
                time.sleep(min(60, 2 ** a * 3)); continue
            raise
        except Exception:
            if a < retries - 1:
                time.sleep(min(60, 2 ** a * 3)); continue
            raise


def _fake(model_key, messages, item, tag):
    """배관만 점검한다. 비싼 모델·다듬은 질문일수록 좋은 답을 내도록 흉내."""
    txt = " ".join(m["content"] for m in messages)
    n_in = len(txt) // 3 + 40

    if tag == "translate":
        out = "[한국어 번역] " + (item or {}).get("question", "문제")[:80]
        return out, n_in, len(out)//3 + 40

    if tag == "rewrite":
        out = "[다듬은 질문] " + (item or {}).get("user_prompt", "질문") + \
              " — 목적, 형식, 분량을 명시하고 필요한 맥락을 보강함."
        return out, n_in, len(out) // 3 + 40

    if tag.startswith("axis_"):
        out = f"[{tag[5:]} 변형] " + txt[-160:]
        return out, n_in, len(out)//3 + 30

    if tag == "judge_pair":
        out = _rng.choice(["A", "A", "B", "무승부"])
        return out, n_in, 5

    if tag == "judge_rubric":
        k = len((item or {}).get("rubric", [])) or 4
        out = json.dumps([_rng.choice([0, 1, 1]) for _ in range(k)])
        return out, n_in, 20

    # 배관 점검용 난이도 흉내. 조건 이름(tag="cond:<조건>")을 보고 축마다 값을 다르게 준다.
    # 목적은 분석이 기대한 패턴(결핍 > 과잉 > 표면 순으로 비싸다)을 제대로 집어내는지
    # 확인하는 것뿐이다. **여기서 나온 수치를 결과로 읽지 말 것.**
    cond = tag[5:] if tag.startswith("cond:") else ""
    p = {"S1": 0.80, "S2": 0.80, "S3": 0.75,        # 표면 잡음 — 거의 안 깎인다
         "E1": 0.70, "E2": 0.70,                    # 정보 과잉 — 주로 토큰을 먹는다
         "D1": 0.45, "D2": 0.50, "D3": 0.45,        # 정보 결핍 — 가장 치명적
         "ko_human": 0.40,                          # 자연복합 — 여러 축이 겹침
         }.get(cond, 0.85)                          # 기준선·en_expert·refined
    if "expensive" in model_key:
        p = min(0.95, p + 0.15)                     # 고가 모델이 엉성함에 더 강건
    good = p >= 0.70
    hit = _rng.random() < p
    task = (item or {}).get("task", "gsm8k")
    if task == "gsm8k":
        out = f"풀이 과정...\n정답: {(item or {}).get('gold','18')}" if hit else "정답: 99999"
    elif task == "mbpp":
        out = ("```python\n" + ((item or {}).get("ref_code") or "def _x(): pass") + "\n```"
               if hit else "```python\ndef _wrong(): return None\n```")
    else:
        out = ("요청하신 내용을 정리하면 다음과 같습니다. " * (6 if good else 14))
    return out, n_in, len(out) // 3 + (30 if good else 90)
