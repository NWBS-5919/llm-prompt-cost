"""analyze.py 와 partc.py 가 같이 쓰는 것 — 실행 기록 로드, 한도 비용 환산.

한도 비용 = 입력 토큰 × 입력단가 + 출력 토큰 × 출력단가  (단가: 100만 토큰당, 단위는 env.txt 를 따름 — ChatKHU 크레딧)
"비싼 모델은 구독 한도를 더 빨리 쓴다"는 가정 아래 API 정가를 대리 지표로 쓴다.
단가는 환경변수로 받는다:  PRICE_CHEAP="0.10,0.40"  PRICE_MID="..."  PRICE_EXPENSIVE="..."
단가를 바꿔도 API 를 다시 부를 필요가 없다 — 토큰은 runs.jsonl 에 있다.
"""
import envload  # noqa: F401  env.txt 읽기
import json, os, pathlib

TIERS = ("cheap", "mid", "expensive")
ACTS = ("raw", "rw")
_DEFAULT = {"cheap": "0.10,0.40", "mid": "0.30,2.50", "expensive": "1.25,10.00"}   # 자리표시자


def prices():
    out = {}
    for t in TIERS:
        a, b = (os.environ.get(f"PRICE_{t.upper()}") or _DEFAULT[t]).split(",")
        out[t] = (float(a), float(b))
    # 재작성기는 저가 등급과 다른 모델일 수 있다 (D-33: 저가 gemma, 재작성 qwen3.8-flash)
    rw = os.environ.get("PRICE_REWRITER")
    out["rewriter"] = tuple(float(x) for x in rw.split(",")) if rw else out["cheap"]
    j = os.environ.get("PRICE_JUDGE")
    out["judge"] = tuple(float(x) for x in j.split(",")) if j else out["expensive"]
    return out


PRICES = prices()
PRICES_ARE_DEFAULT = not any(os.environ.get(f"PRICE_{t.upper()}") for t in TIERS)


def usd(tier, tin, tout):
    pi, po = PRICES[tier]
    return (tin * pi + tout * po) / 1e6


def load_runs(path):
    p = pathlib.Path(path)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def excluded():
    """{문항 id: 이유} — 문제 자체가 성립하지 않아 분석에서 빼는 문항 (data/excluded_items.json, D-31)"""
    p = pathlib.Path(__file__).parent / "data" / "excluded_items.json"
    if not p.exists():
        return {}
    return {k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def load_for_analysis(path):
    """분석용 로드 — 제외 문항을 빼고, 뺀 것을 알린다"""
    ex = excluded()
    rs = load_runs(path)
    out = [r for r in rs if r["id"] not in ex]
    if len(out) < len(rs):
        print(f"  ※ 문제 결함으로 제외 {len(rs) - len(out)}문항: {sorted(r['id'] for r in rs if r['id'] in ex)}"
              f" (data/excluded_items.json)")
    return out


def cell(rec, v, act, tier):
    """(정답 0/1, 출력 토큰, 한도 비용 USD) — 재작성 행동은 재작성 비용을 더한다"""
    c = rec["cond"][f"{v}|{act}|{tier}"]
    cost = usd(tier, c["tokens_in"], c["tokens_out"])
    if act == "rw":
        r = rec["rewrites"][v]
        cost += usd("rewriter", r["tokens_in"], r["tokens_out"])
    return int(c["correct"]), c["tokens_out"], cost


def record_cost(rec):
    """문항 기록 하나에 쓴 전체 비용 (답변 + 재작성 + 번역·변형·판정).
    D1 을 다시 만들며 대체된 예전 기록(superseded)도 실제로 쓴 돈이라 함께 센다 (D-29)."""
    tot = 0.0
    for key, c in rec["cond"].items():
        tot += usd(key.split("|")[2], c["tokens_in"], c["tokens_out"])
    for r in rec["rewrites"].values():
        tot += usd("rewriter", r["tokens_in"], r["tokens_out"])
    for k, (a, b) in rec.get("gen_tokens", {}).items():
        tot += usd(axis_tier(rec, k) if k in ("D1", "N1") else "judge", a, b)
    for old in rec.get("superseded", []):
        if "cost_credits" in old:          # 지금 등급에 없는 모델(예: 빠진 qwen3.7-plus)의 칸 — 옮길 때 그 단가로 계산해 둔 값
            tot += old["cost_credits"]
            continue
        tot += record_cost({"cond": old.get("cond", {}), "rewrites": old.get("rewrites", {}),
                            "gen_tokens": old.get("gen_tokens", {}), "d1_tier": old.get("d1_tier", "rewriter")})
    return tot


def axis_tier(rec, k):
    """LLM 축(D1·N1)을 만든 등급 — 예전 D1 은 재작성기(Flash)가 만들었다"""
    t = rec.get("axis_tier", {}).get(k)
    return t or (rec.get("d1_tier", "rewriter") if k == "D1" else "expensive")


def spend_map(rec):
    """{키 번호(문자열): 크레딧} — 문항 하나를 여러 키로 나눠 채웠을 수 있다. 예전 기록은 key 하나에 전부"""
    if rec.get("spend"):
        return dict(rec["spend"])
    return {str(rec.get("key", 0)): record_cost(rec)}


def spent_in_files(*paths):
    tot = 0.0
    for p in paths:
        if pathlib.Path(p).exists():
            tot += sum(record_cost(r) for r in load_runs(p))
    return tot


def spent_by_key(*paths):
    """{키 번호: 사용 크레딧} — 기록에 key 가 없으면 0번(첫 키)"""
    out = {}
    for p in paths:
        if pathlib.Path(p).exists():
            for r in load_runs(p):
                for k, c in spend_map(r).items():
                    out[int(k)] = out.get(int(k), 0.0) + c
    return out
