"""
이미 돌린 문항의 D1 만 새 방식(D-29: 고가 모델 생성 + 조작 확인 + 최대 3회 재시도)으로 다시 만든다.

D1 이 아닌 변형과 그 답변은 건드리지 않는다. 예전 D1 기록(질문·재작성·답변·토큰)은 지우지 않고
문항 기록의 superseded 에 옮겨 둔다 — 실제로 쓴 크레딧이라 common.record_cost 가 계속 센다.
처음 실행할 때 원본을 <파일명>_D1v1백업.jsonl 로 복사해 둔다.

실행
  python3 redo_d1.py                 # runs.jsonl
  python3 redo_d1.py runs_fake.jsonl --fake
"""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parent)   # 어디서 실행해도 이 폴더 기준으로
import json, sys, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import llm
import common
import run_main as RM

llm.FAKE = "--fake" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PATH = _pl.Path(args[0] if args else "runs.jsonl")


def save(recs):
    """한 문항 끝날 때마다 통째로 다시 쓴다 — 임시 파일에 쓰고 바꿔치기해서 중간에 끊겨도 원본이 깨지지 않게"""
    tmp = PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(PATH)


def redo(rec, item):
    old = {"prompt": rec["prompts"]["D1"],
           "cond": {k: c for k, c in rec["cond"].items() if k.startswith("D1|")},
           "rewrites": {"D1": rec["rewrites"]["D1"]},
           "gen_tokens": {k: rec["gen_tokens"][k] for k in ("D1", "check_D1") if k in rec["gen_tokens"]},
           "d1_tier": rec.get("d1_tier", "rewriter"),
           "checks": {k: v for k, v in rec["checks"].items() if k.startswith("D1_")},
           "replaced_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    gen = {"gen_tokens": {}, "checks": {}}
    text = RM.make_d1(gen, rec["prompts"]["base"], item)
    ans = {"rewrites": {}, "cond": {}}
    RM.run_answers(ans, item, [("D1", text)])

    rec["prompts"]["D1"] = text
    rec["rewrites"]["D1"] = ans["rewrites"]["D1"]
    rec["cond"].update(ans["cond"])
    rec["gen_tokens"].update(gen["gen_tokens"])
    rec["checks"].update(gen["checks"])
    rec["d1_tier"], rec["d1_attempts"] = gen["d1_tier"], gen["d1_attempts"]
    rec.setdefault("superseded", []).append(old)
    return rec


def main():
    if not llm.FAKE:
        llm.check_models()
    recs = common.load_runs(PATH)
    backup = PATH.with_name(PATH.stem + "_D1v1백업.jsonl")
    if not backup.exists():
        shutil.copy(PATH, backup)
        print(f"  원본 백업 → {backup.name}")
    items = {it["id"]: it for it in json.loads(_pl.Path("items.json").read_text(encoding="utf-8"))}
    todo = [i for i, r in enumerate(recs) if "D1_valid" not in r["checks"]]
    print(f"  D1 을 다시 만들 문항 {len(todo)}/{len(recs)}")
    before = sum(common.record_cost(r) for r in recs)
    llm.use_key(0) if llm.KEYS else None
    with ThreadPoolExecutor(max_workers=1 if llm.FAKE else RM.ITEM_WORKERS) as ex:
        futs = {ex.submit(redo, recs[i], items[recs[i]["id"]]): i for i in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            r = recs[futs[fut]] = fut.result()
            save(recs)
            c = r["checks"]
            print(f"  [{n:>3}/{len(todo)}] {r['id']}  시도 {r['d1_attempts']}  "
                  f"변형됨 {c['D1_changed']}  숫자보존 {c['D1_numbers_kept']}  정답보존 {c['D1_same_answer']}"
                  f"  → 유효 {c['D1_valid']}")
    after = sum(common.record_cost(r) for r in recs)
    ok = sum(r["checks"].get("D1_valid", False) for r in recs)
    print(f"\n저장 → {PATH}  · D1 유효 {ok}/{len(recs)} = {ok / max(len(recs), 1):.0%}  · 이번에 쓴 크레딧 {after - before:,.0f}")


if __name__ == "__main__":
    main()
