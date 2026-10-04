"""
v5 · Part B 배분 시뮬레이션  (추가 API 호출 없음)

  입력    runs.jsonl — 학습 절반으로 p(θ,a)·c(θ,a) 를 추정, 시험 절반의 실측값으로 평가
  유형 θ  (변형, 길이 구간)        행동 a  {그대로, 재작성} × {저가, 중가, 고가}  + 건너뛰기
  하루    질문 N개가 순서대로 도착, 중요도 w ∈ {1,2,3}, 한도 B

  정책
    oracle      사후 최적 — 하루치 실측을 다 알고 푼 다중선택 배낭 DP (상한)
    bp_aware    재풀이 입찰가격 — LP 완화의 예산 쌍대변수 λ 로 w·p − λ·c 최대 행동
    bp_blind    같은 정책, 단 결함을 모름 (변형을 평균낸 표, 길이 구간만 앎)   ← 핵심 대조군
    greedy      w·p / c 최대 행동
    threshold   긴 질문 → 고가, 짧은 질문 → 저가, 재작성 안 함
    all_exp     항상 고가 그대로        all_cheap  항상 저가 그대로

  결함 정보의 가치 = bp_aware − bp_blind  (같은 하루, 같은 예산)

  평가값  가치 = w × p_test(θ, a)  — 시험 절반의 유형별 실측 정답률 (운이 아니라 기댓값으로 채점)
          비용 = 그 문항의 실측 비용
  사후 최적은 '하루에 어떤 질문이 올지'만 미리 안다. 정답을 맞힐지 운까지 아는 상한은 쓰지 않는다.
  학습 표는 결함별 칸 추정이 흔들리지 않게 길이 구간 평균 쪽으로 축소한다 (SHRINK 문항 분량)
  자동 검사: 어떤 정책이든 사후 최적을 넘으면 코드 오류로 보고 중단 (D-03)

실행: python3 partc.py [runs.jsonl] [--days 200] [--resolve 5] [--mix mix.json]
"""
import sys, json, math, random, itertools
import numpy as np
from scipy.optimize import linprog
import perturb as P
from common import TIERS, ACTS, load_runs, load_for_analysis, cell

args = sys.argv[1:]
PATH = args[0] if args and not args[0].startswith("--") else "runs.jsonl"
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
DAYS, RESOLVE, N = opt("--days", 200), opt("--resolve", 5), opt("--n", 40)
BUDGETS = [1.0, 0.6, 0.35, 0.2]                 # '전부 고가' 하루 비용 대비
SIGMAS = [0.0, 0.1, 0.2, 0.3]                   # 결함 판별 오류율
WS = (1, 2, 3)
SEED = 20261004
V = P.VARIANTS
A = [(a, t) for a in ACTS for t in TIERS]       # 6 행동
BUCKETS = ("lo", "hi")
SHRINK = opt("--shrink", 10)
MIX = {v: 1 / len(V) for v in V}
if "--mix" in args:
    MIX = json.load(open(opt("--mix", "")))
    s = sum(MIX.values()); MIX = {v: MIX.get(v, 0) / s for v in V}

R = load_for_analysis(PATH)
# LLM 축(D1·N1)은 조작 확인을 통과한 문항만 (D-29, D-35). 예전 D1 기록은 정답 보존만 있다
ok = lambda r, v: v in r["prompts"] and (v not in P.LLM_AXES or
                                         r["checks"].get(f"{v}_valid", r["checks"].get(f"{v}_same_answer", False)))
train = [r for r in R if r["split"] == "train"]
test = [r for r in R if r["split"] == "test"]
if not train or not test:
    raise SystemExit("학습/시험 문항이 모두 있어야 합니다 (파일럿이 너무 작으면 --limit 을 늘리세요)")
# 조작 확인을 통과한 문항이 학습·시험 한쪽에라도 없으면 그 변형은 표를 만들 수 없다 → 빼고 알린다
_drop = [v for v in V if not any(ok(r, v) for r in train) or not any(ok(r, v) for r in test)]
if _drop:
    print(f"  ⚠ 학습·시험 양쪽에 쓸 문항이 없는 변형은 시뮬레이션에서 뺍니다: {_drop}")
    V = [v for v in V if v not in _drop]
    _s = sum(MIX[v] for v in V)
    MIX = {v: MIX[v] / _s for v in V}

# ───────────────────────────────── 비용 정수화 — 정책과 사후 최적이 같은 단위를 쓴다
med_cheap = float(np.median([cell(r, "base", "raw", "cheap")[2] for r in R]))
UNIT = med_cheap / 20 if med_cheap > 0 else 1e-6
icost = lambda usd: max(1, math.ceil(usd / UNIT))

# ───────────────────────────────── 학습 표: p, c (유형별 평균)
def raw_table(rows):
    p, c, n = {}, {}, {}
    for b in BUCKETS:
        for v in V:
            rs = [r for r in rows if r["len_bucket"] == b and ok(r, v)] or [r for r in rows if ok(r, v)]
            for (a, t) in A:
                cs = [cell(r, v, a, t) for r in rs]
                p[(v, b, a, t)] = float(np.mean([x[0] for x in cs]))
                c[(v, b, a, t)] = float(np.mean([x[2] for x in cs])) / UNIT
                n[(v, b, a, t)] = len(cs)
    return p, c, n


def blind(p, c):
    """결함 무시: 변형을 도착 비율로 평균 — 길이 구간만 안다"""
    pb, cb = {}, {}
    for b in BUCKETS:
        for (a, t) in A:
            pb[(b, a, t)] = sum(MIX[v] * p[(v, b, a, t)] for v in V)
            cb[(b, a, t)] = sum(MIX[v] * c[(v, b, a, t)] for v in V)
    return pb, cb


_p, C_AW, _n = raw_table(test if "--selftest" in args else train)   # --selftest: 논리 점검용(시험표로 결정)
P_BL, C_BL = blind(_p, C_AW)
P_AW = {k: (_n[k] * _p[k] + SHRINK * P_BL[k[1:]]) / (_n[k] + SHRINK) for k in _p}   # 축소 추정
P_TEST, _, _ = raw_table(test)                   # 평가용 — 정책은 보지 않는다
PB = {b: np.mean([r["len_bucket"] == b for r in train]) for b in BUCKETS}

# 시험 문항의 평가값 (유형별 실측 정답률, 문항의 실측 정수 비용)
TEST = {(r["id"], v): {(a, t): (P_TEST[(v, r["len_bucket"], a, t)], icost(cell(r, v, a, t)[2])) for (a, t) in A}
        for r in test for v in V if ok(r, v)}
TEST_BY_V = {v: [r for r in test if ok(r, v)] for v in V}


# ───────────────────────────────── LP 완화 → 예산 쌍대변수 λ
def make_lp(types, pf, cf):
    """types: [(key, prob, w)], pf/cf: key,a,t → p, c.  반환: 목적계수, 비용계수, 행동선택 제약행렬"""
    nT, nA = len(types), len(A)
    obj = np.array([pr * w * pf(k, a, t) for (k, pr, w) in types for (a, t) in A])
    cst = np.array([pr * cf(k, a, t) for (k, pr, w) in types for (a, t) in A])
    one = np.zeros((nT, nT * nA))
    for i in range(nT):
        one[i, i * nA:(i + 1) * nA] = 1
    return obj, cst, one


def lam(lp, n_left, b_left):
    obj, cst, one = lp
    if n_left <= 0:
        return 0.0
    res = linprog(-n_left * obj, A_ub=np.vstack([n_left * cst, one]),
                  b_ub=np.concatenate([[b_left], np.ones(one.shape[0])]),
                  bounds=(0, None), method="highs")
    if res.status != 0:
        return 0.0
    return max(0.0, -float(res.ineqlin.marginals[0]))


T_AW = [((v, b), MIX[v] * PB[b] / 3, w) for v in V for b in BUCKETS for w in WS]
LP_AW = make_lp(T_AW, lambda k, a, t: P_AW[(k[0], k[1], a, t)], lambda k, a, t: C_AW[(k[0], k[1], a, t)])
T_BL = [(b, PB[b] / 3, w) for b in BUCKETS for w in WS]
LP_BL = make_lp(T_BL, lambda k, a, t: P_BL[(k, a, t)], lambda k, a, t: C_BL[(k, a, t)])


# ───────────────────────────────── 하루 생성
def make_day(rng):
    day = []
    for _ in range(N):
        v = rng.choices(V, weights=[MIX[x] for x in V])[0]
        r = rng.choice(TEST_BY_V[v])
        day.append({"id": r["id"], "v": v, "b": r["len_bucket"], "w": rng.choice(WS)})
    return day


def oracle(day, B):
    dp = np.zeros(B + 1)
    for q in day:
        new = dp.copy()
        for at in A:
            val, c = TEST[(q["id"], q["v"])][at]
            if c <= B and val > 0:
                cand = np.full(B + 1, -np.inf)
                cand[c:] = dp[:B + 1 - c] + q["w"] * val
                new = np.maximum(new, cand)
        dp = new
    return float(dp.max())


def simulate(day, B, choose, min_cost):
    left, val, stuck = B, 0.0, None
    for i, q in enumerate(day):
        if stuck is None and left < min_cost:
            stuck = i + 1
        at = choose(q, left, len(day) - i)
        if at is None:
            continue
        good, c = TEST[(q["id"], q["v"])][at]
        if c > left:                              # 실제 비용이 남은 한도를 넘음 → 막힘
            left = 0
            continue
        left -= c
        val += q["w"] * good
    return val, (stuck or len(day) + 1)


def affordable(est, left):
    return [at for at in A if est(at) <= left]


def pol_fixed(at_fn, cf):
    def choose(q, left, n_left):
        at = at_fn(q)
        return at if cf(q, at) <= left else None
    return choose


def pol_greedy(q, left, n_left):
    cand = affordable(lambda at: C_AW[(q["obs"], q["b"], *at)], left)
    if not cand:
        return None
    return max(cand, key=lambda at: q["w"] * P_AW[(q["obs"], q["b"], *at)] / C_AW[(q["obs"], q["b"], *at)])


def pol_bp(aware):
    state = {"lam": 0.0, "k": 0}
    def choose(q, left, n_left):
        if state["k"] % RESOLVE == 0:
            state["lam"] = lam(LP_AW if aware else LP_BL, n_left, left)
        state["k"] += 1
        pf = (lambda at: P_AW[(q["obs"], q["b"], *at)]) if aware else (lambda at: P_BL[(q["b"], *at)])
        cf = (lambda at: C_AW[(q["obs"], q["b"], *at)]) if aware else (lambda at: C_BL[(q["b"], *at)])
        cand = affordable(cf, left)
        if not cand:
            return None
        best = max(cand, key=lambda at: q["w"] * pf(at) - state["lam"] * cf(at))
        return best if q["w"] * pf(best) - state["lam"] * cf(best) > 0 else None
    return choose


def observe(day, sigma, rng):
    for q in day:
        q["obs"] = q["v"] if rng.random() >= sigma else rng.choice([x for x in V if x != q["v"]])


# ───────────────────────────────── 실행
def main():
    rng = random.Random(SEED)
    cbl = lambda q, at: C_BL[(q["b"], *at)]
    min_cost = min(C_BL[(b, *at)] for b in BUCKETS for at in A)
    fixed = {
        "all_exp": pol_fixed(lambda q: ("raw", "expensive"), cbl),
        "all_cheap": pol_fixed(lambda q: ("raw", "cheap"), cbl),
        "threshold": pol_fixed(lambda q: ("raw", "expensive" if q["b"] == "hi" else "cheap"), cbl),
    }
    names = ["oracle", "bp_aware", "bp_blind", "greedy", "threshold", "all_exp", "all_cheap"]
    res = {}
    for d in range(DAYS):
        day = make_day(rng)
        full = sum(TEST[(q["id"], q["v"])][("raw", "expensive")][1] for q in day)
        for bf in BUDGETS:
            B = max(1, int(full * bf))
            best = oracle(day, B)
            if best <= 0:
                continue
            base_vals = {k: simulate(day, B, f, min_cost) for k, f in fixed.items()}
            for q in day:
                q["obs"] = q["v"]
            base_vals["bp_blind"] = simulate(day, B, pol_bp(False), min_cost)
            for s in SIGMAS:
                observe(day, s, random.Random(SEED + d * 97 + int(s * 100)))
                vals = dict(base_vals)
                vals["bp_aware"] = simulate(day, B, pol_bp(True), min_cost)
                vals["greedy"] = simulate(day, B, pol_greedy, min_cost)
                for k, (v, stuck) in vals.items():
                    if v > best + 1e-9:
                        raise SystemExit(f"[중단] {k} 가 사후 최적을 넘었습니다 (day {d}, B {bf}, σ {s}): "
                                         f"{v} > {best}. 코드 오류 — D-03")
                    res.setdefault((bf, s), {}).setdefault(k, []).append((v / best, v, stuck))
                res[(bf, s)].setdefault("oracle", []).append((1.0, best, N + 1))
        if (d + 1) % 50 == 0:
            print(f"  모의 하루 {d + 1}/{DAYS}")

    L = "=" * 86
    print(L); print(f"Part B — 사후 최적 대비 달성률 (평균, 모의 하루 {DAYS}일 · 하루 {N}문항)"); print(L)
    out = {"days": DAYS, "n": N, "resolve": RESOLVE, "mix": MIX, "unit_usd": UNIT, "results": []}
    for bf in BUDGETS:
        for s in SIGMAS:
            r = res.get((bf, s), {})
            if not r:
                continue
            row = {k: float(np.mean([x[0] for x in r[k]])) for k in names if k in r}
            stuck = {k: float(np.mean([x[2] <= N for x in r[k]])) for k in names if k in r}
            diff = np.array([a[1] - b[1] for a, b in zip(r["bp_aware"], r["bp_blind"])])
            bs = [np.mean(diff[np.random.default_rng(i).integers(0, len(diff), len(diff))]) for i in range(1000)]
            voi = (float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
            out["results"].append({"budget": bf, "sigma": s, "ratio": row, "stuck_rate": stuck,
                                   "voi_weighted_correct": voi})
            if s == 0.0:
                print(f"\n  예산 {bf:.0%}  " + "  ".join(f"{k} {row[k]:.0%}" for k in names if k in row))
            print(f"    σ={s:.1f}  결함 인식 {row['bp_aware']:.1%} · 결함 무시 {row['bp_blind']:.1%}"
                  f"  → 결함 정보의 가치 {voi[0]:+.2f} [{voi[1]:+.2f}, {voi[2]:+.2f}] 가중정답/일"
                  f"  · 한도 막힘 비율(전부 고가) {stuck['all_exp']:.0%}")
    json.dump(out, open("partc_sim.json", "w"), ensure_ascii=False, indent=2)
    print("\n저장 → partc_sim.json   (자동 검사: 사후 최적 초과 0건)")


if __name__ == "__main__":
    main()
