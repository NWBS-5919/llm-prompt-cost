"""
D-37 채점 보정 — '정답: 숫자' 형식이 없는 답을 LLM 으로 최종 답만 추출해 다시 채점한다. 답변은 다시 부르지 않는다.

문제: grade.py 는 '정답: 숫자' 가 없으면 답의 마지막 숫자를 답으로 본다. 형식 지시가 없는 조건(지시 제거, 사람 질문)에서
모델은 "답: 일주일에 평균 21개씩 … 둘이 합쳐 125개" 처럼 쓰고, 채점기는 125 를 읽는다 (m017 HUMAN_B, 정답 21).

방법
  - '정답: 숫자' 형식이 있는 칸은 예전 규칙 그대로 (결과 변화 없음)
  - 형식이 없는 칸만 LLM 에게 "답변이 최종 답으로 내놓은 수 하나" 를 뽑게 한다 (문제는 주지 않는다 — 풀지 못하게)
  - 검증: 형식이 있는 칸 150개를 LLM 으로도 뽑아 규칙 추출과의 일치율을 보고한다
  - 예전 판정은 칸마다 grade_v1 로 남긴다. 원본은 <파일명>_채점v1백업.jsonl

실행
  python3 regrade.py --dry               # 대상 칸 수와 예상 비용만 (무과금)
  python3 regrade.py                     # runs.jsonl (2번 키)
  python3 regrade.py runs_fake.jsonl --fake
"""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parent)
import json, re, sys, shutil, random, threading
from concurrent.futures import ThreadPoolExecutor
import llm
import common

llm.FAKE = "--fake" in sys.argv
DRY = "--dry" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PATH = _pl.Path(args[0] if args else "runs.jsonl")
KEY = int(_os.environ.get("REGRADE_KEY") or 1)          # 1번 키는 10월 한도를 다 썼다
ROLE = "mid"                                             # qwen3.8-flash — 추출만 하므로 싼 모델로 충분, 검증으로 확인
TAIL = 4000                                              # 최종 답은 끝에 있다. 긴 답은 끝부분만
N_CHECK = 0 if "--skip-check" in sys.argv else 150          # 2026-10-06 첫 실행에서 150/150 일치 확인
CACHE = _pl.Path("logs") / "extract_cache.jsonl"         # 추출 결과를 한 칸씩 바로 남긴다 — 끊겨도 이어서 (분당 120회 제한, D-37)

EXPL = re.compile(r"(?:최종\s*)?(?:정답|답)\s*[:：]?\s*\**\s*(-?\d[\d,]*\.?\d*)")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
EXTRACT_SYS = ("너는 채점 보조 도구다. [답변]은 어떤 수학 문제에 대한 AI의 답변이다. "
               "답변이 최종 답으로 내놓은 수 하나만 출력하라. 단위·기호·쉼표·설명 없이 숫자만 쓴다. "
               "답변이 최종 답을 내놓지 않았거나 여러 값 중 하나로 정하지 않았으면 NONE 이라고만 출력하라. "
               "문제를 직접 풀거나 답변의 계산을 고치지 마라.")
_LOCK = threading.Lock()


def rule_value(ans):
    m = EXPL.findall(ans or "")
    return m[-1].replace(",", "") if m else None


def same(x, gold):
    try:
        return x is not None and abs(float(x) - float(gold)) < 1e-4
    except ValueError:
        return False


def llm_value(ans, item=None):
    """returns (값 또는 None, 입력 토큰, 출력 토큰)"""
    text = (ans or "")[-TAIL:]
    out, a, b, _ = llm.chat(ROLE, [{"role": "system", "content": EXTRACT_SYS},
                                   {"role": "user", "content": f"[답변]\n{text}"}],
                            item=item, tag="extract")
    if "NONE" in out.upper():
        return None, a, b
    m = NUM.findall(out)
    return (m[0].replace(",", "").rstrip(".") if m else None), a, b


def main():
    recs = common.load_runs(PATH)
    ex = common.excluded()
    todo, explicit = [], []
    for i, r in enumerate(recs):
        if r["id"] in ex:
            continue
        for k, c in r["cond"].items():
            if c.get("grader"):                           # 이미 보정한 칸
                continue
            (explicit if rule_value(c["answer"]) is not None else todo).append((i, k))
    chars = sum(len((recs[i]["cond"][k]["answer"] or "")[-TAIL:]) for i, k in todo)
    est = common.usd(ROLE, chars / 1.6 + 120 * len(todo), 6 * len(todo))   # 한국어 약 1.6자/토큰 (대략)
    print(f"  형식 있는 칸 {len(explicit)} (규칙 그대로) · 형식 없는 칸 {len(todo)} (LLM 추출) · 검증 {N_CHECK}칸")
    print(f"  예상 비용 약 {est * (len(todo) + N_CHECK) / max(len(todo), 1):,.0f} 크레딧")
    if DRY:
        return
    if not llm.FAKE:
        llm.check_models()
        llm.use_key(KEY)
        print(f"  {KEY + 1}번 키 사용")
    backup = PATH.with_name(PATH.stem + "_채점v1백업.jsonl")
    if not backup.exists():
        shutil.copy(PATH, backup)
        print(f"  원본 백업 → {backup.name}")

    rng = random.Random(20261006)
    check = rng.sample(explicit, min(N_CHECK, len(explicit)))
    spent = {}
    cache = {}
    CACHE.parent.mkdir(exist_ok=True)
    if CACHE.exists() and not llm.FAKE:
        for l in open(CACHE, encoding="utf-8"):
            d = json.loads(l)
            cache[(d["id"], d["cell"])] = d
    print(f"  이전 실행에서 저장된 추출 {len(cache)}칸")
    failed = []

    def job(ik):
        i, k = ik
        key = (recs[i]["id"], k)
        if key in cache:
            return cache[key]["value"]
        try:
            v, a, b = llm_value(recs[i]["cond"][k]["answer"], item={"gold": recs[i]["gold"]})
        except Exception as e:                            # 한 칸이 실패해도 멈추지 않는다 — 다시 실행하면 그 칸만
            with _LOCK:
                failed.append((key, str(e)[:120]))
            return "ERR"
        with _LOCK:
            t = recs[i].setdefault("regrade_tokens", [0, 0])
            t[0] += a; t[1] += b
            spent[i] = spent.get(i, 0.0) + common.usd(ROLE, a, b)
            if not llm.FAKE:
                with open(CACHE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": key[0], "cell": k, "value": v, "a": a, "b": b}, ensure_ascii=False) + "\n")
        return v

    with ThreadPoolExecutor(max_workers=1 if llm.FAKE else llm.WORKERS) as pool:
        if check:
            got_check = list(pool.map(job, check))
            agree = sum(same(v, rule_value(recs[i]["cond"][k]["answer"])) for (i, k), v in zip(check, got_check))
            print(f"\n  검증: 형식 있는 칸 {len(check)}개에서 LLM 추출 = 규칙 추출 {agree}/{len(check)} ({agree / max(len(check), 1):.1%})")
            for (i, k), v in list(zip(check, got_check)):
                if not same(v, rule_value(recs[i]["cond"][k]["answer"])):
                    print(f"    불일치 {recs[i]['id']} {k}: 규칙 {rule_value(recs[i]['cond'][k]['answer'])} / LLM {v}")
        got = []
        for n, v in enumerate(pool.map(job, todo), 1):
            got.append(v)
            if n % 200 == 0:
                print(f"  {n}/{len(todo)} · 실패 {len(failed)}", flush=True)
    if failed:
        print(f"\n[중단] {len(failed)}칸 추출 실패 — 기록은 바꾸지 않았다. 잠시 뒤 다시 실행하면 실패한 칸만 다시 한다")
        for key, e in failed[:3]:
            print(f"    {key}: {e}")
        return

    changed = {"오답→정답": 0, "정답→오답": 0}
    for (i, k), v in zip(todo, got):
        c = recs[i]["cond"][k]
        c.setdefault("grade_v1", c["correct"])
        new = same(v, recs[i]["gold"])
        if new != c["correct"]:
            changed["오답→정답" if new else "정답→오답"] += 1
        c.update({"correct": new, "extracted": v, "grader": "llm"})
    for i, k in explicit:
        c = recs[i]["cond"][k]
        c.setdefault("grade_v1", c["correct"])
        c.update({"extracted": rule_value(c["answer"]), "grader": "rule"})
    for i, cost in spent.items():                          # 키별 사용량에 추출 비용을 더한다
        sp = recs[i].setdefault("spend", common.spend_map(recs[i]))
        sp[str(KEY)] = sp.get(str(KEY), 0.0) + cost

    tmp = PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(PATH)
    print(f"\n  판정이 바뀐 칸: 오답→정답 {changed['오답→정답']} · 정답→오답 {changed['정답→오답']} (LLM 추출 {len(todo)}칸 중)")
    print(f"  추출 비용 {sum(spent.values()):,.1f} 크레딧 · 저장 → {PATH}")


if __name__ == "__main__":
    main()
