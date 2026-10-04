"""OpenAI 호환 엔드포인트 호출 래퍼 + 배관 점검용 가짜 모델  (v5)

v5 변경
  - 중가 등급 MODEL_MID 추가 (cheap / mid / expensive)
  - 응답 usage 원본을 그대로 돌려줘 추론 토큰을 기록할 수 있게 함
  - temperature 를 거부하는 모델이면 temperature 없이 한 번 더 시도
  - MIN_INTERVAL(초) 로 호출 간격을 강제 — 무료 등급 분당 한도 대응
  - 여러 스레드에서 동시에 불러도 된다: 호출 시작 간격과 사용량 집계는 잠금으로 지킨다 (WORKERS)
"""
import envload  # noqa: F401  env.txt 읽기
import json, os, time, random, re, ssl, threading, urllib.request, urllib.error

# API_STYLE: openai (기본, /chat/completions) | anthropic (/v1/messages — 학교 라우터·Claude)
STYLE = (os.environ.get("API_STYLE") or "openai").lower()
BASE = (os.environ.get("LLM_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
        or ("https://api.anthropic.com" if STYLE == "anthropic" else "https://api.openai.com/v1")).rstrip("/")
KEY = (os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
       or os.environ.get("ANTHROPIC_API_KEY") or "")
# 1번 키 한도를 다 쓰면 2번 키로 넘어간다 (env.txt 의 LLM_API_KEY_2)
KEYS = [k for k in (KEY, os.environ.get("LLM_API_KEY_2") or "") if k]
KEY_IDX = 0


class CreditExhausted(Exception):
    """게이트웨이가 크레딧 부족으로 거절함"""


def use_key(i):
    global KEY, KEY_IDX
    KEY_IDX, KEY = i, KEYS[i]
try:
    EXTRA = json.loads(os.environ.get("EXTRA_BODY") or "{}")   # 예: {"enable_thinking": false}
except ValueError:
    raise SystemExit("[중단] EXTRA_BODY 가 올바른 JSON 이 아닙니다")
_CHEAP = os.environ.get("MODEL_CHEAP", "")
MODELS = {
    "cheap":     _CHEAP,
    "mid":       os.environ.get("MODEL_MID", ""),
    "expensive": os.environ.get("MODEL_EXPENSIVE", ""),
    "rewriter":  os.environ.get("MODEL_REWRITER") or _CHEAP,
    "judge":     os.environ.get("MODEL_JUDGE") or os.environ.get("MODEL_EXPENSIVE", ""),
}
TIERS = ("cheap", "mid", "expensive")
MIN_INTERVAL = float(os.environ.get("MIN_INTERVAL") or 0)
# 동시에 보내는 호출 수. 호출 하나가 5~10초라 순서대로 보내면 1문항(약 52회)에 5분 넘게 걸린다
WORKERS = max(1, int(os.environ.get("WORKERS") or 8))
_LOCK = threading.Lock()
# 답변 길이 상한 — 정상 풀이는 1천 토큰 안팎인데, 같은 말을 반복하며 수십 분 끝나지 않는 폭주가 있었다(m027, D-36).
# 상한에 걸린 답은 usage["finish_reason"] == "length" 로 기록돼 분석에서 구분할 수 있다
MAX_TOKENS = int(os.environ.get("MAX_TOKENS") or 8192)          # 호출 시작 간격(_last)과 사용량 집계를 스레드 사이에서 지킨다

FAKE = False                      # run_main.py 가 --fake 일 때 True 로 바꾼다
_rng = random.Random(20260922)
_no_temp = set()                  # temperature 를 거부한 모델
_last = [0.0]

# macOS 의 python.org 파이썬은 인증서 묶음이 없어 HTTPS 가 실패할 수 있다 → certifi 가 있으면 그걸 쓴다
try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()

USAGE = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
USAGE_BY = {}                     # 역할별 [입력, 출력] — 예산 계산용


def check_models():
    missing = [k for k in ("cheap", "mid", "expensive", "rewriter", "judge") if not MODELS[k]]
    if missing:
        raise SystemExit(f"[중단] env.txt 의 모델 칸이 비어 있습니다: {missing}")
    if not KEY:
        raise SystemExit("[중단] API 키가 없습니다. env.txt 의 LLM_API_KEY 를 채우세요")
    print(f"  연결: {STYLE} · {BASE} · 모델 {MODELS['cheap']} / {MODELS['mid']} / {MODELS['expensive']}")


def chat(model_key, messages, temperature=0.0, item=None, tag=""):
    """returns (text, tokens_in, tokens_out, usage_raw)"""
    if FAKE:
        t, a, b = _fake(model_key, messages, item, tag)
        u = {"prompt_tokens": a, "completion_tokens": b}
    else:
        t, a, b, u = _real(MODELS[model_key], messages, temperature)
    with _LOCK:
        USAGE["calls"] += 1
        USAGE["tokens_in"] += a
        USAGE["tokens_out"] += b
        ub = USAGE_BY.setdefault(model_key, [0, 0])
        ub[0] += a; ub[1] += b
    return t, a, b, u


def _post(model, messages, temperature, extra=None):
    """returns (text, tokens_in, tokens_out, usage_raw). extra 를 주면 EXTRA_BODY 대신 그걸 쓴다"""
    if STYLE == "anthropic":
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body = {"model": model, "max_tokens": 4096,
                "messages": [m for m in messages if m["role"] != "system"]}
        if system:
            body["system"] = system
        url = BASE + ("/messages" if BASE.endswith("/v1") else "/v1/messages")
        headers = {"x-api-key": KEY, "Authorization": f"Bearer {KEY}",
                   "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    else:
        # 스트리밍으로 받는다 — 게이트웨이(Cloudflare)는 응답이 120초 안에 안 끝나면 524 로 끊는데,
        # 중가(초당 약 47토큰)의 5천 토큰 넘는 답은 그 안에 못 끝난다. 조각이 계속 오면 끊기지 않는다 (D-32)
        body = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True},
                "max_tokens": MAX_TOKENS}
        url = f"{BASE}/chat/completions"
        headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    if model not in _no_temp:
        body["temperature"] = temperature
    body.update(EXTRA if extra is None else extra)
    # 파이썬 기본 User-Agent(Python-urllib)는 Cloudflare 가 봇으로 막는다(error 1010) → 일반 클라이언트처럼 보낸다
    headers["User-Agent"] = "Mozilla/5.0 (Macintosh) llm-prompt-cost/1.0"
    headers["Accept"] = "application/json" if STYLE == "anthropic" else "text/event-stream"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    if STYLE == "anthropic":
        with urllib.request.urlopen(req, timeout=240, context=_SSL) as r:
            d = json.loads(r.read())
        u = d.get("usage", {}) or {}
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        return text, u.get("input_tokens", 0), u.get("output_tokens", 0), u
    parts, u, finish = [], None, None
    with urllib.request.urlopen(req, timeout=240, context=_SSL) as r:   # timeout 은 조각 사이 대기 한도
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            for ch in d.get("choices") or []:
                parts.append((ch.get("delta") or {}).get("content") or "")
                finish = ch.get("finish_reason") or finish
            if d.get("usage"):
                u = d["usage"]
    if u is None:   # 토큰 수는 이 연구의 핵심 지표 — 없으면 기록하지 않고 다시 부른다
        raise RuntimeError("스트림 응답에 usage 가 없습니다")
    u = dict(u, finish_reason=finish)
    return "".join(parts), u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u


def _real(model, messages, temperature, retries=6):
    for a in range(retries):
        with _LOCK:               # 잠금을 쥔 채 기다려야 동시 호출에서도 시작 간격이 MIN_INTERVAL 이상 벌어진다
            wait = MIN_INTERVAL - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            return _post(model, messages, temperature)
        except urllib.error.HTTPError as e:
            msg = ""
            try:
                msg = e.read().decode("utf-8", "ignore")
            except Exception:
                pass
            if e.code in (402, 403, 429) and re.search(r"credit|크레딧|quota|insufficient|balance|한도", msg, re.I):
                raise CreditExhausted(f"HTTP {e.code}: {msg[:200]}")
            if e.code == 400 and re.search(r"temperature", msg, re.I) and model not in _no_temp:
                print(f"  [알림] {model} 이 temperature 를 거부 → 기본값으로 재시도 (재현성 주의)")
                _no_temp.add(model)
                continue
            # 520~524 는 Cloudflare 앞단의 일시 오류(524 = 120초 시간 초과)
            if e.code in (408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529) and a < retries - 1:
                time.sleep(min(90, 2 ** a * 4)); continue
            raise RuntimeError(f"HTTP {e.code}: {msg[:300]}")
        except Exception as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise SystemExit("[중단] 인증서 문제입니다. 터미널에서  pip3 install certifi  를 한 번 실행한 뒤 다시 해 주세요.")
            if a < retries - 1:
                time.sleep(min(90, 2 ** a * 4)); continue
            raise


# ───────────────────────────────────────────── 가짜 모델 (배관 점검 전용)
_P_TIER = {"cheap": 0.55, "mid": 0.70, "expensive": 0.85}
_PEN = {"base": 0.0, "S1_lo": 0.02, "S1_hi": 0.08, "E2_lo": 0.03, "E2_hi": 0.07,
        "D2": 0.05, "D1": 0.20}


def _fake(model_key, messages, item, tag):
    """배관만 점검한다. 숫자는 의미 없음."""
    txt = " ".join(m["content"] for m in messages)
    n_in = len(txt) // 3 + 40
    it = item or {}
    if tag == "translate":
        out = "[번역] " + it.get("question", "문제")
        return out, n_in, len(out) // 3 + 30
    if tag.startswith("rewrite"):
        out = "[다듬은 질문] " + messages[-1]["content"][:200] + " 단계별로 풀어 주세요."
        return out, n_in, len(out) // 3 + 30
    if tag == "axis_N1":
        # 숫자가 든 무관한 문장을 기준선 앞에 끼운다 — 바뀌었고, 원래 숫자는 그대로, 새 숫자가 있다
        out = "그중 7개는 다른 것보다 조금 작았다. " + messages[-1]["content"].split("---\n")[-1]
        return out, n_in, len(out) // 3 + 30
    if tag == "axis_D1":
        # 숫자가 없는 첫 낱말 하나를 뺀다 — 진짜 D1 처럼 '바뀌었고 숫자는 그대로'인 결과를 내야 재시도 배관을 점검할 수 있다
        out = messages[-1]["content"].split("---\n")[-1]
        w = next((w for w in out.split() if not re.search(r"\d", w)), "")
        out = out.replace(w + " ", "", 1) if w else out
        return out, n_in, len(out) // 3 + 30
    if tag.startswith("check_"):
        return _rng.choice(["같다", "같다", "같다", "다르다"]), n_in, 3
    if tag == "judge_rubric":
        k = len(it.get("rubric", [])) or 4
        return json.dumps([_rng.choice([0, 1, 1]) for _ in range(k)]), n_in, 20
    # answer:<variant>:<tier>:<raw|rw>
    _, var, tier, act = (tag.split(":") + ["", "", "", ""])[:4]
    p = _P_TIER.get(tier, 0.6) - _PEN.get(var.split("_")[0] if var.startswith("HUMAN") else var, 0.10)
    if act == "rw":
        p += 0.06 if var in ("D1", "D2") or var.startswith("HUMAN") else -0.01
    hit = _rng.random() < max(0.02, min(0.98, p))
    out = (f"풀이...\n정답: {it.get('gold', '0')}" if hit else "풀이...\n정답: 99999")
    pad = {"cheap": 120, "mid": 160, "expensive": 220}.get(tier, 150)
    pad += {"D1": 120, "D2": -60, "E2_hi": 30}.get(var, 0)
    return out, n_in, max(20, pad + _rng.randint(-30, 30))
