"""
v5 · Part A 분석

  ⓪ 개요   ① 조작 확인   ② G1 파일럿 판정
  ③ 실측표 (변형 × 행동)   ④ H1 확증 분석   ⑤ H2 재작성 순효과   ⑥ 사람 작성 질문
  → table1.csv (포스터 표 1) · cells.csv (문항 단위, Part B 입력 확인용)

실행: python3 analyze.py [runs.jsonl]
"""
import sys, csv, math
import numpy as np
from scipy import stats
import perturb as P
from common import TIERS, ACTS, load_runs, load_for_analysis, cell, PRICES_ARE_DEFAULT

PATH = sys.argv[1] if len(sys.argv) > 1 else "runs.jsonl"
R = load_for_analysis(PATH)
rng = np.random.default_rng(20261004)
L = "=" * 86
TN = {"cheap": "저가", "mid": "중가", "expensive": "고가"}
AN = {"raw": "그대로", "rw": "재작성"}


def boot_ci(x, f=np.mean, B=2000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    s = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, m, run = [0.0] * len(ps), len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def ok_var(r, v):
    """이 문항의 변형 v 를 분석에 쓸 수 있나. LLM 축(D1·N1)은 조작 확인을 통과한 것만 (D-29, D-35).
    예전 D1 기록은 정답 보존만 있다. 기록에 아직 없는 변형(나중에 추가된 축)은 뺀다"""
    if v not in r["prompts"]:
        return False
    if v not in P.LLM_AXES:
        return True
    c = r["checks"]
    return c.get(f"{v}_valid", c.get(f"{v}_same_answer", False))


def ok_d1(r):
    return ok_var(r, "D1")


# ───────────────────────────────────────────────────────── ⓪
print(L); print(f"⓪ 개요 — {PATH}"); print(L)
print(f"  문항 {len(R)}개 (train {sum(r['split']=='train' for r in R)} / test {sum(r['split']=='test' for r in R)})"
      f" · 출처 {sorted({r['source'] for r in R})}")
if R:
    print(f"  모델 {R[0]['models']}")
if PRICES_ARE_DEFAULT:
    print("  ⚠ 단가가 자리표시자입니다. PRICE_CHEAP / PRICE_MID / PRICE_EXPENSIVE 를 실제 값으로 넣으세요.")

# ───────────────────────────────────────────────────────── ①
print(); print(L); print("① 조작 확인 — 변형이 의도대로 됐는가"); print(L)
keys = ["translate_numbers_kept"] + [f"{v}_numbers_kept" for v in P.VARIANTS if v != "base"] + [f"{v}_{k}" for v in P.LLM_AXES for k in ("changed", "same_answer", "valid")] + ["N1_added_number"]
rate = {}
for k in keys:
    v = [r["checks"][k] for r in R if k in r["checks"]]
    rate[k] = sum(v) / len(v) if v else float("nan")
    print(f"  {k:<26} 통과 {sum(v):>3}/{len(v):<3} = {rate[k]:.0%}")
leaks = [r["rewrites"][v]["leak"] for r in R for v in r["rewrites"]]
leak_rate = sum(leaks) / len(leaks) if leaks else float("nan")
print(f"  {'재작성 정답 누설':<24} {sum(leaks):>3}/{len(leaks):<3} = {leak_rate:.0%}")
print("  ※ 번역 숫자 보존 실패는 영어 원문이 숫자를 단어로 쓴 경우(two 등)일 수 있다 — 표본을 눈으로 확인")

# ───────────────────────────────────────────────────────── ②
print(); print(L); print("② G1 파일럿 판정"); print(L)
base_acc = np.mean([cell(r, "base", "raw", "cheap")[0] for r in R]) if R else float("nan")
g = [("저가 기준선 정답률 ≤ 85% (천장 회피)", base_acc <= 0.85, f"{base_acc:.0%}"),
     *[(f"{v} 유효(조작 확인 통과) ≥ 70%", rate.get(f"{v}_valid", 0) >= 0.70, f"{rate.get(f'{v}_valid', float('nan')):.0%}")
       for v in P.LLM_AXES],
     ("재작성 누설 ≤ 5%", leak_rate <= 0.05, f"{leak_rate:.0%}")]
for name, ok, val in g:
    print(f"  {'통과' if ok else '실패'}  {name:<34} 실측 {val}")
print("  → 모두 통과면 같은 runs.jsonl 에 이어서 전체 실행. 실패 항목은 계획서 '위험과 대안' 참고")

# ───────────────────────────────────────────────────────── ③
print(); print(L); print("③ 실측표 — 정답률 / 평균 출력 토큰 / 평균 한도 비용(1천 회당, 단가 단위)"); print(L)
hdr = "".join(f"{TN[t]+'·'+AN[a]:>20}" for a in ACTS for t in TIERS)
print(f"  {'변형':<10}{hdr}")
rows_out = []
for v in P.VARIANTS:
    rs = [r for r in R if ok_var(r, v)]
    if not rs:
        continue
    line = f"  {P.LABEL[v]:<10}"
    for a in ACTS:
        for t in TIERS:
            cs = [cell(r, v, a, t) for r in rs]
            acc = np.mean([c[0] for c in cs]); tok = np.mean([c[1] for c in cs]); cost = np.mean([c[2] for c in cs])
            line += f"  {acc:>5.0%} {tok:>5.0f} {cost*1000:>7.1f}"   # 칸마다 띄워 숫자가 붙지 않게
            rows_out.append({"variant": v, "label": P.LABEL[v], "action": a, "tier": t, "n": len(cs),
                             "acc": round(float(acc), 4), "tokens_out": round(float(tok), 1),
                             "cost_usd": float(cost)})
    print(line)
with open("table1.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
with open("cells.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["id", "split", "len_bucket", "variant", "action", "tier", "correct", "tokens_out", "cost_usd"])
    for r in R:
        for v in r["prompts"]:
            for a in ACTS:
                for t in TIERS:
                    w.writerow([r["id"], r["split"], r["len_bucket"], v, a, t, *cell(r, v, a, t)])
print("  → table1.csv, cells.csv 저장  (LLM 축은 조작 확인 통과 문항만)")

# ───────────────────────────────────────────────────────── ④
print(); print(L); print("④ H1 확증 분석 — 각 축 최고 강도 vs 기준선 (저가 · 그대로)"); print(L)
tests = []
for v in ("S1_lo", "S1_md", "S1_hi", "E2_hi", "D2", "D1", "N1"):
    rs = [r for r in R if ok_var(r, v)]
    if not rs:
        continue
    b = [cell(r, "base", "raw", "cheap") for r in rs]
    x = [cell(r, v, "raw", "cheap") for r in rs]
    n10 = sum(1 for p, q in zip(b, x) if p[0] and not q[0])      # 기준선만 맞힘
    n01 = sum(1 for p, q in zip(b, x) if q[0] and not p[0])
    pval = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    lr = [math.log(max(q[1], 1) / max(p[1], 1)) for p, q in zip(b, x)]
    lo, hi = boot_ci(lr)
    tests.append((v, len(rs), np.mean([q[0] for q in x]) - np.mean([p[0] for p in b]), n10, n01, pval,
                  math.exp(np.mean(lr)) if lr else float("nan"), math.exp(lo), math.exp(hi)))
adj = holm([t[5] for t in tests])
print(f"  {'축':<10}{'n':>4}{'Δ정답률':>9}{'기준만':>7}{'변형만':>7}{'p(Holm)':>10}{'출력토큰 배율 [95% CI]':>26}")
for t, pa in zip(tests, adj):
    print(f"  {P.LABEL[t[0]]:<10}{t[1]:>4}{t[2]*100:>+8.1f}p{t[3]:>7}{t[4]:>7}{pa:>10.3f}"
          f"{t[6]:>11.2f}x [{t[7]:.2f}, {t[8]:.2f}]")

# ───────────────────────────────────────────────────────── ⑤
print(); print(L); print("⑤ H2 재작성 순효과 — 재작성 − 그대로 (저가 모델, 재작성 비용 포함)"); print(L)
print(f"  {'변형':<10}{'Δ정답률 [95% CI]':>26}{'Δ비용 1천회당 [95% CI]':>34}")
for v in P.VARIANTS:
    rs = [r for r in R if ok_var(r, v)]
    if not rs:
        continue
    d_acc = [cell(r, v, "rw", "cheap")[0] - cell(r, v, "raw", "cheap")[0] for r in rs]
    d_cost = [(cell(r, v, "rw", "cheap")[2] - cell(r, v, "raw", "cheap")[2]) * 1000 for r in rs]
    a0, a1 = boot_ci(d_acc); c0, c1 = boot_ci(d_cost)
    print(f"  {P.LABEL[v]:<10}{np.mean(d_acc)*100:>+9.1f}p [{a0*100:+.1f}, {a1*100:+.1f}]"
          f"{np.mean(d_cost):>+14.3f} [{c0:+.3f}, {c1:+.3f}]")

# ───────────────────────────────────────────────────────── ⑥
print(); print(L); print("⑥ 사람이 쓴 질문 — 합성 곡선의 어디에 놓이는가 (저가 · 그대로)"); print(L)
for hk in ("HUMAN_A", "HUMAN_B"):
    rs = [r for r in R if hk in r["prompts"]]
    if not rs:
        print(f"  {hk}: 아직 없음 (human_TODO.csv)"); continue
    h = [cell(r, hk, "raw", "cheap") for r in rs]; b = [cell(r, "base", "raw", "cheap") for r in rs]
    print(f"  {hk}  n={len(rs)}  정답률 {np.mean([x[0] for x in h]):.0%} (같은 문항 기준선 {np.mean([x[0] for x in b]):.0%})"
          f"  출력토큰 {np.mean([x[1] for x in h]):.0f} (기준선 {np.mean([x[1] for x in b]):.0f})")
print("\n다음: python3 partc.py " + PATH)
