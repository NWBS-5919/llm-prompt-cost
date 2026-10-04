"""
D-33 등급 재구성 — 기존 기록을 새 등급 이름으로 옮긴다. API 를 부르지 않는다.

  예전  cheap = qwen3.8-flash   mid = qwen3.7-plus   expensive = qwen3.8-max
  이제  cheap = google/gemma-3-27b-it   mid = qwen3.8-flash   expensive = qwen3.8-max

  - 예전 cheap(Flash) 칸 → mid 로 이름만 바꾼다 (같은 모델, 같은 응답)
  - 예전 mid(Plus) 칸 → 문항 기록의 superseded 에 보관하고, Plus 단가로 계산한 비용을 함께 적는다
  - 키별 사용량(spend)을 지금 단가로 확정해 둔다 — 단가표가 바뀌어도 이미 쓴 크레딧은 그대로
  - 새 cheap(gemma) 칸과 새 변형은 run_main.py 가 채운다

반드시 env.txt 를 바꾸기 **전에** 실행한다 (옛 단가로 비용을 확정해야 하므로). 원본은 <파일명>_등급v1백업.jsonl.

실행
  python3 migrate_tiers.py               # runs.jsonl + runs_test*.jsonl
"""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parent)
import json, shutil, sys, time
import common

OLD = {"cheap": "qwen3.8-flash", "mid": "qwen3.7-plus", "expensive": "qwen3.8-max"}
NEW = {"cheap": "google/gemma-3-27b-it", "mid": "qwen3.8-flash", "expensive": "qwen3.8-max"}


def migrate(rec):
    if rec["models"].get("mid") != OLD["mid"]:
        return False                                  # 이미 옮겼거나 다른 구성
    rec["spend"] = common.spend_map(rec)              # 옛 단가로 확정
    plus = {k: c for k, c in rec["cond"].items() if k.endswith("|mid")}
    cost = sum(common.usd("mid", c["tokens_in"], c["tokens_out"]) for c in plus.values())
    rec.setdefault("superseded", []).append({
        "reason": "D-33 등급 재구성: qwen3.7-plus 를 등급에서 뺌 (Flash 와 정답률이 같고 더 비쌈)",
        "model": OLD["mid"], "cond": plus, "cost_credits": cost,
        "models_before": dict(rec["models"]), "replaced_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    cond = {}
    for k, c in rec["cond"].items():
        if k.endswith("|mid"):
            continue
        cond[k[:-len("cheap")] + "mid" if k.endswith("|cheap") else k] = c
    rec["cond"] = cond
    rec["models"].update(NEW)
    return True


def main():
    if os_env_mid() != OLD["mid"]:
        raise SystemExit(f"[중단] env.txt 의 MODEL_MID 가 {OLD['mid']} 일 때(바꾸기 전에) 실행하세요 — 옛 단가로 비용을 확정해야 합니다")
    paths = [_pl.Path("runs.jsonl"), *sorted(_pl.Path(".").glob("runs_test*.jsonl"))]
    for p in paths:
        if not p.exists():
            continue
        recs = common.load_runs(p)
        if p.name == "runs.jsonl":
            backup = p.with_name(p.stem + "_등급v1백업.jsonl")
            if not backup.exists():
                shutil.copy(p, backup)
                print(f"  원본 백업 → {backup.name}")
            before = sum(common.record_cost(r) for r in recs)
            n = sum(migrate(r) for r in recs)
            after = sum(sum(common.spend_map(r).values()) for r in recs)
            print(f"  {p.name}: {n}/{len(recs)} 문항 옮김 · 확정 사용량 {before:,.1f} → {after:,.1f} 크레딧 (같아야 정상)")
        else:                                           # 연결 테스트 기록은 사용량만 확정 (분석에 안 씀)
            for r in recs:
                r["spend"] = common.spend_map(r)
            print(f"  {p.name}: 사용량 확정 {sum(sum(r['spend'].values()) for r in recs):,.1f} 크레딧")
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(p)
    print("\n다음: env.txt 를 새 등급으로 바꾼다 (MODEL_CHEAP, MODEL_MID, PRICE_CHEAP, PRICE_MID, PRICE_REWRITER)")


def os_env_mid():
    return _os.environ.get("MODEL_MID", "")


if __name__ == "__main__":
    main()
