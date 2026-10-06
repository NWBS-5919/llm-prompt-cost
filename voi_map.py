"""
D-38 ④ 결정 지도 — 하루 질문 중 심한 오타(오타 20%) 비율 × 예산 격자에서 결함 정보의 가치를 계산한다. API 를 부르지 않는다.

비율마다 partc.py 를 --mix 로 따로 돌리고(동시에), 결과를 results/voi/ 에 모은다.
나머지 질문 구성은 D-38 ④ 기준 구성을 (1 − 심한 오타 비율)에 맞게 비례 조정한다.

실행
  python3 voi_map.py              # 표 출력 + results/voi/voi_map.json
"""
import os as _os, pathlib as _pl
HERE = _pl.Path(__file__).resolve().parent
_os.chdir(HERE)
import json, subprocess, sys, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor

BASE_MIX = {"base": 0.55, "S1_lo": 0.12, "S1_md": 0.06, "E2_lo": 0.05, "E2_hi": 0.02,
            "D2": 0.08, "D1": 0.06, "N1": 0.04}
SEVERE = [0.0, 0.02, 0.05, 0.10, 0.20]
OUT = HERE / "results" / "voi"


def mix_for(s):
    tot = sum(BASE_MIX.values())
    m = {k: v / tot * (1 - s) for k, v in BASE_MIX.items()}
    m["S1_hi"] = s
    return m


def run_one(s):
    d = _pl.Path(tempfile.mkdtemp(prefix="voi_"))
    (d / "mix.json").write_text(json.dumps(mix_for(s)))
    r = subprocess.run([sys.executable, str(HERE / "partc.py"), str(HERE / "runs.jsonl"), "--mix", str(d / "mix.json")],
                       cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    dst = OUT / f"partc_sim_severe{int(s * 100):02d}.json"
    shutil.copy(d / "partc_sim.json", dst)
    return s, json.loads(dst.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(len(SEVERE)) as ex:
        res = dict(ex.map(run_one, SEVERE))
    table = []
    print("결함 정보의 가치 (결함 인식 − 결함 무시, 가중정답/일, σ=0) · 괄호: 결함 인식 − 전부 저가 (달성률 %p)\n")
    budgets = sorted({x["budget"] for x in res[SEVERE[0]]["results"]}, reverse=True)
    print(f"{'심한 오타 비율':>12s} " + " ".join(f"{'예산 ' + str(int(b * 100)) + '%':>24s}" for b in budgets))
    for s in SEVERE:
        cells = []
        for b in budgets:
            x = next(r for r in res[s]["results"] if r["budget"] == b and r["sigma"] == 0.0)
            v, lo, hi = x["voi_weighted_correct"]
            gap = (x["ratio"]["bp_aware"] - x["ratio"]["all_cheap"]) * 100
            cells.append(f"{v:+5.2f} [{lo:+5.2f},{hi:+5.2f}] ({gap:+4.1f})")
            table.append({"severe": s, "budget": b, "voi": v, "lo": lo, "hi": hi,
                          "aware_minus_all_cheap_pp": gap, "ratio": x["ratio"]})
        print(f"{s:>12.0%} " + " ".join(f"{c:>24s}" for c in cells))
    (OUT / "voi_map.json").write_text(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"\n저장 → {OUT / 'voi_map.json'}")


if __name__ == "__main__":
    main()
