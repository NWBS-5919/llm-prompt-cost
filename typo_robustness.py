"""
D-38 ② 오타 효과의 일반성 — 등급 밖의 모델 3개에 기준선·오타 5/10/20% 를 그대로 보내 정답률 변화를 잰다.

질문 문장은 runs.jsonl 에 저장된 것을 그대로 쓴다(같은 번역·같은 오타). 채점은 regrade.py 와 같다:
'정답: 숫자' 가 있으면 규칙, 없으면 qwen3.8-flash 로 최종 답만 추출.
결과 → runs_typo.jsonl (이어서 실행 가능). 분석은 --report.

실행
  python3 typo_robustness.py --report      # 기존 3등급 + 추가 모델 표 (무과금)
  python3 typo_robustness.py               # 2번 키로 실행
"""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parent)
import json, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import binomtest
import llm
import common
import regrade as RG

# 단가: ChatKHU 문서(2026-09-30) 크레딧 / 100만 토큰
MODELS = {"solar-mini4": (100, 400), "google/gemma-4-31B-it": (130, 380), "deepseek-v4-flash": (200, 400)}
VARS = ["base", "S1_lo", "S1_md", "S1_hi"]
OUT = _pl.Path("runs_typo.jsonl")
LIMIT_CREDITS = float(_os.environ.get("TYPO_LIMIT") or 600)   # 넘기 전에 멈춘다
SYSTEM = "You are a helpful assistant."
_LOCK = threading.Lock()


def load_out():
    return {r["id"]: r for r in common.load_runs(OUT)} if OUT.exists() else {}


def save(out):
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(OUT)


def run():
    llm.check_models()
    llm.use_key(int(_os.environ.get("REGRADE_KEY") or 1))
    src = common.load_for_analysis("runs.jsonl")
    out = load_out()
    jobs = []
    for r in src:
        o = out.setdefault(r["id"], {"id": r["id"], "gold": r["gold"], "split": r["split"], "cond": {}})
        for m in MODELS:
            for v in VARS:
                if f"{v}|raw|{m}" not in o["cond"]:
                    jobs.append((r, m, v))
    print(f"  남은 호출 {len(jobs)} (모델 {len(MODELS)} × 변형 {len(VARS)} × 문항 {len(src)})")
    spent = [0.0]
    failed = []

    def job(j):
        r, m, v = j
        if spent[0] > LIMIT_CREDITS:
            return None
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["prompts"][v]}]
        try:                                             # 한 칸이 실패해도 멈추지 않는다 — 다시 실행하면 그 칸만
            text, a, b, u = llm._real(m, msgs, 0.0)
            pi, po = MODELS[m]
            cost = (a * pi + b * po) / 1e6
            x = RG.rule_value(text)
            grader = "rule"
            if x is None:
                x, ea, eb = RG.llm_value(text)
                cost += common.usd(RG.ROLE, ea, eb)
                grader = "llm"
        except Exception as e:
            with _LOCK:
                failed.append(str(e)[:120])
            return None
        cell = {"answer": text, "correct": RG.same(x, r["gold"]), "extracted": x, "grader": grader,
                "tokens_in": a, "tokens_out": b, "usage": u}
        with _LOCK:
            spent[0] += cost
            out[r["id"]]["cond"][f"{v}|raw|{m}"] = cell
        return cell

    with ThreadPoolExecutor(max_workers=llm.WORKERS) as pool:
        futs = [pool.submit(job, j) for j in jobs]
        for n, _ in enumerate(as_completed(futs), 1):    # 끝난 순서대로 — 50칸마다 저장
            if n % 50 == 0:
                with _LOCK:
                    save(out)
                print(f"  {n}/{len(jobs)} · 약 {spent[0]:,.0f} 크레딧 · 실패 {len(failed)}", flush=True)
    save(out)
    print(f"  끝 · 약 {spent[0]:,.0f} 크레딧 · 실패 {len(failed)} · 저장 → {OUT}")
    if failed:
        print(f"  실패한 칸은 다시 실행하면 그것만 다시 한다. 예: {failed[0]}")
    if spent[0] > LIMIT_CREDITS:
        print(f"  [예산 도달] TYPO_LIMIT={LIMIT_CREDITS:.0f} 에서 멈춤 — 다시 실행하면 이어서")


def report():
    src = {r["id"]: r for r in common.load_for_analysis("runs.jsonl")}
    out = load_out()
    rows = [("저가 gemma-3-27b", lambda r, v: r["cond"][f"{v}|raw|cheap"]["correct"], src),
            ("중가 qwen3.8-flash", lambda r, v: r["cond"][f"{v}|raw|mid"]["correct"], src),
            ("고가 qwen3.8-max", lambda r, v: r["cond"][f"{v}|raw|expensive"]["correct"], src)]
    for m in MODELS:
        rows.append((m, (lambda mm: lambda r, v: r["cond"][f"{v}|raw|{mm}"]["correct"])(m),
                     {i: r for i, r in out.items() if i in src and all(f"{v}|raw|{m}" in r["cond"] for v in VARS)}))
    print(f"{'모델':22s} {'n':>3s} " + " ".join(f"{x:>8s}" for x in ("기준선", "오타5%", "오타10%", "오타20%")) +
          "   기준만/변형만   p(기준 vs 20%)")
    sig = 0
    for name, f, rs in rows:
        if not rs:
            print(f"{name:22s}   (아직 없음)"); continue
        R = list(rs.values())
        acc = [sum(f(r, v) for r in R) / len(R) for v in VARS]
        b = sum(f(r, "base") and not f(r, "S1_hi") for r in R)
        c = sum(f(r, "S1_hi") and not f(r, "base") for r in R)
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        sig += (p < 0.05 and b > c)
        print(f"{name:22s} {len(R):>3d} " + " ".join(f"{a:>8.0%}" for a in acc) + f"   {b:>4d}/{c:<4d}     {p:.4f}")
    print(f"\n  유의한 하락(p<0.05) 모델 수: {sig}/{len(rows)}  → D-38 ② 규칙: 4개 이상이면 '대부분에 비싸다', 아니면 '모델마다 다르다'")


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
