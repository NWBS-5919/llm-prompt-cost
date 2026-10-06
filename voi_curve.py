"""
D-39 정보 가치 vs 추정 오차 — 보정(학습) 문항 수를 10·20·30·49 로 바꿔 가며 '실제로 얻는' 결함 정보의 가치를 잰다.
상한은 정책이 시험 문항의 실제 정답률표로 결정할 때(--selftest)의 가치. API 를 부르지 않는다.

실행
  python3 voi_curve.py            # 표 출력 + results/voi/voi_curve.json
"""
import os as _os, pathlib as _pl
HERE = _pl.Path(__file__).resolve().parent
_os.chdir(HERE)
import json, subprocess, sys, tempfile, statistics
from concurrent.futures import ThreadPoolExecutor
import voi_map as V

SEVERE = [0.02, 0.10]
SIZES = [10, 20, 30]
SEEDS = range(5)
BUDGETS = "0.2,0.35"
OUT = HERE / "results" / "voi"


def run(tag, s, extra):
    d = _pl.Path(tempfile.mkdtemp(prefix="voic_"))
    (d / "mix.json").write_text(json.dumps(V.mix_for(s)))
    cmd = [sys.executable, str(HERE / "partc.py"), str(HERE / "runs.jsonl"), "--mix", str(d / "mix.json"),
           "--budgets", BUDGETS, "--sigmas", "0"] + extra
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    res = json.loads((d / "partc_sim.json").read_text())["results"]
    return tag, s, {x["budget"]: x for x in res}


def main():
    jobs = []
    for s in SEVERE:
        jobs.append(("상한", s, ["--selftest"]))
        jobs.append((49, s, []))
        for n in SIZES:
            for seed in SEEDS:
                jobs.append((n, s, ["--train-n", str(n), "--train-seed", str(seed)]))
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda j: run(*j), jobs))
    table = []
    for s in SEVERE:
        print(f"\n심한 오타 {s:.0%} — 결함 정보의 가치 (가중정답/일, 결함 인식 − 결함 무시) · 괄호: 결함 인식 달성률")
        for b in (0.2, 0.35):
            line = f"  예산 {b:.0%}: "
            for tag in (10, 20, 30, 49, "상한"):
                xs = [r[b] for t, ss, r in res if t == tag and ss == s]
                vois = [x["voi_weighted_correct"][0] for x in xs]
                aw = [x["ratio"]["bp_aware"] for x in xs]
                m = statistics.mean(vois)
                sd = statistics.stdev(vois) if len(vois) > 1 else 0.0
                line += f"  {str(tag) + ('개' if tag != '상한' else ''):>4s} {m:+5.2f}" + (f"±{sd:.2f}" if sd else "") + f" ({statistics.mean(aw):.1%})"
                table.append({"severe": s, "budget": b, "train_n": tag, "voi_mean": m, "voi_sd": sd,
                              "voi_runs": vois, "aware_ratio": statistics.mean(aw),
                              "all_cheap_ratio": xs[0]["ratio"]["all_cheap"], "blind_ratio": statistics.mean(x["ratio"]["bp_blind"] for x in xs)})
            print(line)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "voi_curve.json").write_text(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"\n저장 → {OUT / 'voi_curve.json'}  (10·20·30개는 무작위 표본 5번 평균 ± 표준편차)")


if __name__ == "__main__":
    main()
