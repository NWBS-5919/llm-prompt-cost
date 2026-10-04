"""
v5 · Part A 실행

문항마다
  1. 영어 원문 → 판정 모델이 한국어 기준선으로 번역 (+ 숫자 보존 검사)
  2. 변형 9개 (perturb.VARIANTS)  + 사람 작성(있으면 HUMAN_A, HUMAN_B)
  3. LLM 축(D1·N1) 조작 확인 — 통과할 때까지 최대 3번 (D-29, D-35)
  4. 변형마다 재작성 1회 (+ 정답 누설 검사)
  5. 변형마다 행동 6개 = {그대로, 재작성} × {cheap, mid, expensive}, 1턴

실행
  python3 run_main.py --fake               # 무과금 배관 점검 (runs_fake.jsonl)
  python3 run_main.py --limit 30           # 파일럿: 앞 30문항
  python3 run_main.py                      # 전체 — 같은 파일에 이어서 실행된다
                                           # 기존 문항도 빠진 변형·등급 칸만 채운다 (D-33)
  --out 파일명                             # 기본 runs.jsonl (가짜는 runs_fake.jsonl)
"""
import json, sys, csv, time, pathlib, copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import llm
from llm import chat, USAGE, USAGE_BY, TIERS
import common
from grade import grade
import perturb as P

HERE = pathlib.Path(__file__).parent
llm.FAKE = "--fake" in sys.argv
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
OUT = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
       else ("runs_fake.jsonl" if llm.FAKE else "runs.jsonl"))
SYSTEM = "You are a helpful assistant."
# 키 하나당 이번 달 최대 크레딧(env.txt 의 BUDGET). 넘을 것 같으면 다음 키로, 키가 없으면 멈춘다
import os
BUDGET = float(os.environ.get("BUDGET") or 0)
# 동시에 돌리는 문항 수 (문항 안의 동시 호출 수는 llm.WORKERS)
ITEM_WORKERS = max(1, int(os.environ.get("ITEM_WORKERS") or 3))


def spent():
    return sum(common.usd(k if k in common.PRICES else "expensive", a, b)
               for k, (a, b) in USAGE_BY.items())


SESSION_BY_KEY = {}          # 이번 실행에서 키별로 쓴 양 (문항 단위로 확정)

TRANSLATE_SYS = ("다음 수학 문제를 한국어로 자연스럽게 옮겨라. 숫자와 조건을 하나도 빠뜨리지 말고, "
                 "숫자는 아라비아 숫자로 쓴다. 문제를 풀지 마라. 번역문만 출력한다.")
REWRITE_SYS = ("너는 사용자의 질문을 AI가 더 잘 처리할 수 있게 다듬는 도구다. "
               "오타를 고치고, 질문과 무관한 내용은 지우고, 의도를 분명히 하고, 풀이 과정과 최종 답 형식을 요청하는 문장을 붙인다. "
               "질문의 숫자와 조건은 절대 바꾸지 않는다. 질문에 답하거나 풀지 말고, 다듬은 질문만 출력한다.")
CHECK_SYS = ("두 수학 문제가 같은 최종 정답을 갖는지 판정하라. "
             "두 번째 문제만 보고도 첫 번째와 같은 답을 구할 수 있으면 '같다', 아니면 '다르다'. 한 단어로만 답한다.")


def load_human():
    out = {}
    p = HERE / "human_TODO.csv"
    if p.exists():
        with open(p, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                for w in ("A", "B"):
                    t = (row.get(f"writer_{w}") or "").strip()
                    if t:
                        out.setdefault(row["id"], {})[f"HUMAN_{w}"] = t
    return out


def answer(item, text, tier, var, act):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    out, a, b, u = chat(tier, msgs, item=item, tag=f"answer:{var}:{tier}:{act}")
    return {"answer": out, "correct": bool(grade(item, out)),
            "tokens_in": a, "tokens_out": b, "usage": u}


def new_record(item):
    return {"id": item["id"], "key": llm.KEY_IDX, "split": item["split"], "len_bucket": item["len_bucket"],
            "source": item["source"], "gold": item["gold"],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "models": dict(llm.MODELS),
            "prompts": {}, "gen_tokens": {}, "checks": {}, "rewrites": {}, "cond": {}}


def missing(rec, human):
    """이 기록에 아직 없는 것 — 변형, 사람 질문, (변형 × 행동 × 등급) 칸. 비어 있으면 끝난 문항"""
    out = [v for v in P.VARIANTS if v not in rec["prompts"]]
    out += [k for k in human.get(rec["id"], {}) if k not in rec["prompts"]]
    vs = [v for v in rec["prompts"]] + out
    out += [f"{v}|{a}|{t}" for v in vs for a in common.ACTS for t in TIERS if f"{v}|{a}|{t}" not in rec["cond"]]
    return out


def run_item(item, human, rec=None):
    """새 문항이면 처음부터, 이미 있는 기록이면 빠진 것만 채운다 (등급·변형을 늘려도 기존 응답을 다시 부르지 않게, D-33)"""
    rec = rec or new_record(item)
    if "base" not in rec["prompts"]:
        ko, a, b, _ = chat("judge", [{"role": "system", "content": TRANSLATE_SYS},
                                     {"role": "user", "content": item["question"]}],
                           item=item, tag="translate")
        rec["gen_tokens"]["translate"] = [a, b]
        rec["checks"]["translate_numbers_kept"] = P.numbers_kept(item["question"], ko)
        rec["prompts"]["base"] = ko.strip() + P.INSTR
    base = rec["prompts"]["base"]

    for v in P.VARIANTS:
        if v in rec["prompts"]:
            continue
        if v in P.LLM_AXES:
            rec["prompts"][v] = make_llm_variant(rec, v, base, item)
            continue
        t, a, b = P.apply_variant(v, base, item)
        rec["prompts"][v] = t
        if a or b:
            rec["gen_tokens"][v] = [a, b]
        rec["checks"][f"{v}_numbers_kept"] = P.numbers_kept(base, t)
    for k, t in human.get(item["id"], {}).items():
        rec["prompts"].setdefault(k, t)
    run_answers(rec, item, list(rec["prompts"].items()))
    rec["models"] = dict(llm.MODELS)
    return rec


FEEDBACK = {
    "unchanged": {"D1": "직전 시도는 아무것도 빼지 않았다. 반드시 한두 군데를 빼라.",
                  "N1": "직전 시도는 아무 문장도 넣지 않았다. 반드시 문장 하나를 끼워 넣어라."},
    "numbers": {"D1": "직전 시도는 숫자가 든 조건을 지웠다. 숫자가 있는 문장은 그대로 두고 이름·장소·상황 설명만 빼라.",
                "N1": "직전 시도는 원래 숫자를 지우거나 바꿨다. 원래 문장은 그대로 두고 새 문장만 끼워 넣어라."},
    "no_new_number": {"N1": "직전 시도가 넣은 문장에 숫자가 없다. 계산에 쓰이지 않는 숫자를 하나 넣어라."},
    "answer": {"D1": "직전 시도는 문제의 정답을 바꿨다. 계산에 쓰이지 않는 정보만 빼라.",
               "N1": "직전 시도가 넣은 문장이 정답을 바꿨다. 계산과 무관한 내용이어야 한다."},
}


def make_llm_variant(rec, v, base, item):
    """LLM 축(D1·N1)을 만들고 조작 확인을 통과할 때까지 최대 P.LLM_TRIES 번 다시 만든다 (D-29, D-35).
    통과 = 기준선과 다름 + 원래 숫자 모두 보존 (+ N1 은 새 숫자 하나 이상) + 심판이 같은 정답이라고 판정.
    못 넘으면 {v}_valid=False 로 남겨 분석에서 뺀다."""
    gen, chk = [0, 0], [0, 0]
    feedback, changed, nums, added, same = "", False, False, True, False
    for attempt in range(1, P.LLM_TRIES + 1):
        t, a, b = P.llm_generate(v, base, item, feedback)
        gen[0] += a; gen[1] += b
        changed, nums, same = t.strip() != base.strip(), P.numbers_kept(base, t), False
        added = v != "N1" or bool(P.added_numbers(base, t))
        why = ("unchanged" if not changed else "numbers" if not nums else "no_new_number" if not added else None)
        if why:
            feedback = FEEDBACK[why][v]
            continue
        verdict, a, b, _ = chat("judge", [{"role": "system", "content": CHECK_SYS},
                                          {"role": "user", "content": f"[문제 1]\n{base}\n\n[문제 2]\n{t}"}],
                                item=item, tag=f"check_{v}")
        chk[0] += a; chk[1] += b
        same = ("같" in verdict) and ("다르" not in verdict)
        if same:
            break
        feedback = FEEDBACK["answer"][v]
    rec["gen_tokens"][v] = gen
    rec["gen_tokens"][f"check_{v}"] = chk
    rec.setdefault("axis_tier", {})[v] = P.AXIS_ROLE          # 비용 환산용 (common.record_cost)
    rec.setdefault("axis_attempts", {})[v] = attempt
    if v == "D1":                                              # 예전 이름 (redo_d1.py 와 예전 기록)
        rec["d1_tier"], rec["d1_attempts"] = P.AXIS_ROLE, attempt
    checks = {f"{v}_numbers_kept": nums, f"{v}_same_answer": same, f"{v}_changed": changed,
              f"{v}_valid": changed and nums and added and same}
    if v == "N1":
        checks["N1_added_number"] = added
    rec["checks"].update(checks)
    return t


def make_d1(rec, base, item):
    return make_llm_variant(rec, "D1", base, item)


def run_answers(rec, item, prompts):
    """prompts 의 변형마다 재작성 1회 + 행동 6개 중 기록에 없는 것만 동시에 보내고 rec 에 기록한다"""
    def rewrite(v, text):
        rw, a, b, u = chat("rewriter", [{"role": "system", "content": REWRITE_SYS},
                                        {"role": "user", "content": text}],
                           item=item, tag=f"rewrite:{v}")
        rw = rw.strip()
        leak = (item["gold"] in P.numbers(rw)) and (item["gold"] not in P.numbers(text))
        return {"text": rw, "tokens_in": a, "tokens_out": b, "leak": leak}

    need = lambda v, a, t: f"{v}|{a}|{t}" not in rec["cond"]
    # 재작성과 답변은 서로 독립이라 동시에 보낸다. 재작성본 답변(rw)만 그 변형의 재작성이 끝난 뒤에 보낸다.
    # 이미 있는 재작성본은 다시 만들지 않는다. 가짜 모드는 공유 난수(_rng)의 순서가 바뀌지 않도록 1개씩 돌린다.
    with ThreadPoolExecutor(max_workers=1 if llm.FAKE else llm.WORKERS) as ex:
        f_rw = {ex.submit(rewrite, v, text): v for v, text in prompts if v not in rec["rewrites"]}
        f_cond = {(v, "raw", tier): ex.submit(answer, item, text, tier, v, "raw")
                  for v, text in prompts for tier in TIERS if need(v, "raw", tier)}
        rewrites = {v: rec["rewrites"][v] for v, _ in prompts if v in rec["rewrites"]}
        for v in list(rewrites):
            for tier in TIERS:
                if need(v, "rw", tier):
                    f_cond[(v, "rw", tier)] = ex.submit(answer, item, rewrites[v]["text"], tier, v, "rw")
        for fut in as_completed(f_rw):
            v = f_rw[fut]
            rewrites[v] = fut.result()
            for tier in TIERS:
                if need(v, "rw", tier):
                    f_cond[(v, "rw", tier)] = ex.submit(answer, item, rewrites[v]["text"], tier, v, "rw")
        cond = {k: fut.result() for k, fut in f_cond.items()}

    # 칸 순서를 (변형 → 행동 → 등급)으로 고정해 다시 쓴다 — 예전 칸과 새 칸이 섞여도 같은 순서
    old = rec["cond"]
    rec["rewrites"] = {v: rewrites[v] for v, _ in prompts}
    rec["cond"] = {}
    for v, _ in prompts:
        for tier in TIERS:
            for a in common.ACTS:
                k = f"{v}|{a}|{tier}"
                rec["cond"][k] = old[k] if k in old else cond[(v, a, tier)]
    for k, c in old.items():             # 지금 등급·변형에 없는 예전 칸도 버리지 않는다
        rec["cond"].setdefault(k, c)


def save(path, order, recs):
    """기록 전체를 다시 쓴다 — 임시 파일에 쓰고 바꿔치기해서 중간에 끊겨도 원본이 깨지지 않게"""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(recs[i], ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    if not llm.FAKE:
        llm.check_models()
    items = json.loads((HERE / "items.json").read_text(encoding="utf-8"))
    human = load_human()
    # 사람 작성 질문이 아직 없는 문항은 맨 뒤로 — 그 사이에 human_TODO.csv 를 채우면 반영된다
    todo_ids = set()
    hp = HERE / "human_TODO.csv"
    if hp.exists():
        with open(hp, encoding="utf-8-sig") as f:
            todo_ids = {row["id"] for row in csv.DictReader(f)}
    items.sort(key=lambda it: it["id"] in todo_ids and it["id"] not in human)
    if LIMIT:
        items = items[:LIMIT]
    # 문제 자체가 성립하지 않는 문항(D-31)은 분석에서 빠지므로 크레딧을 더 쓰지 않는다. 이미 있는 기록은 그대로 둔다
    ex_ids = common.excluded()
    if ex_ids and not llm.FAKE:
        print(f"  제외 문항 {sorted(i for i in ex_ids if any(it['id'] == i for it in items))} 은 건너뜀 (data/excluded_items.json)")
    items = [it for it in items if llm.FAKE or it["id"] not in ex_ids]
    out_path = HERE / OUT
    recs = {r["id"]: r for r in common.load_runs(out_path)} if out_path.exists() else {}
    order = list(recs)
    # 기록의 등급 모델과 지금 설정이 다르면 섞이지 않게 멈춘다 — 등급을 바꿨으면 migrate_tiers.py 를 먼저 (D-33)
    for r in recs.values():
        bad = [t for t in TIERS if r["models"].get(t) != llm.MODELS[t] and any(k.endswith("|" + t) for k in r["cond"])]
        if bad:
            raise SystemExit(f"[중단] {r['id']} 의 {bad} 등급 모델이 env.txt 와 다릅니다 "
                             f"(기록 {[r['models'].get(t) for t in bad]} / 설정 {[llm.MODELS[t] for t in bad]}). "
                             f"python3 migrate_tiers.py 로 기록을 먼저 옮기세요")

    t0 = time.time()
    n_run = 0
    # 키별 이번 달 사용 — 보관한 연결 테스트 기록(runs_test*.jsonl)까지 센다. 같은 파일을 두 번 세지 않게 집합으로
    logs = {out_path, *HERE.glob("runs_test*.jsonl")}
    prior = common.spent_by_key(*map(str, logs))
    used = lambda k: prior.get(k, 0.0) + SESSION_BY_KEY.get(k, 0.0)
    per_item = [0.0]

    exhausted = set()            # 게이트웨이가 크레딧 부족으로 거절한 키

    def room(k, n):
        """k번 키로 n 문항을 더 돌릴 예산이 있나"""
        return llm.FAKE or (k not in exhausted and (not BUDGET or used(k) + per_item[0] * 1.2 * n <= BUDGET))

    # 새 문항 + 빠진 칸이 있는 기존 문항 (등급·변형을 늘리면 기존 문항도 다시 대상이 된다)
    queue = deque(it for it in items if it["id"] not in recs or missing(recs[it["id"]], human))
    n_done = sum(1 for it in items if it["id"] in recs and not missing(recs[it["id"]], human))
    if not llm.FAKE:
        if recs:
            per_item[0] = sum(common.record_cost(r) for r in recs.values()) / len(recs)
        for k in range(len(llm.KEYS)):
            print(f"  {k + 1}번 키: 이번 달 사용(기록 기준) {used(k):,.0f} / 예산 {BUDGET:,.0f} 크레딧")
        print(f"  할 일: 새 문항 {sum(it['id'] not in recs for it in queue)} · 칸 보충 {sum(it['id'] in recs for it in queue)}")
        llm.use_key(0)

    # 문항 여러 개를 동시에 돌린다 — 한 문항의 시간은 가장 긴 답 하나가 정하므로, 그동안 다른 문항을 진행한다 (D-30).
    # 키는 진행 중인 문항이 하나도 없을 때만 바꾼다. 그래야 문항별로 어느 키로 얼마를 썼는지(spend)가 정확하다.
    inflight, err = {}, None
    with ThreadPoolExecutor(max_workers=1 if llm.FAKE else ITEM_WORKERS) as ex:
        while queue or inflight:
            while queue and err is None and len(inflight) < (1 if llm.FAKE else ITEM_WORKERS):
                if room(llm.KEY_IDX, len(inflight) + 1):
                    it = queue.popleft()
                    old = recs.get(it["id"])
                    inflight[ex.submit(run_item, it, human, copy.deepcopy(old) if old else None)] = it
                    continue
                if inflight:
                    break                      # 키를 바꾸기 전에 진행 중인 문항이 끝나길 기다린다
                nxt = next((k for k in range(llm.KEY_IDX + 1, len(llm.KEYS)) if room(k, 1)), None)
                if nxt is None:
                    print(f"\n[예산 도달] 모든 키의 예산을 썼습니다 — 남은 {len(queue)}문항은 돌리지 않고 멈춥니다"
                          f" (크레딧을 사면 env.txt 의 BUDGET 을 올리고 다시 실행)")
                    queue.clear()
                    break
                print(f"\n[키 전환] {llm.KEY_IDX + 1}번 키 예산 소진 → {nxt + 1}번 키로 이어서 실행")
                llm.use_key(nxt)
            if not inflight:
                break
            fin, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in fin:
                item = inflight.pop(fut)
                try:
                    rec = fut.result()
                except llm.CreditExhausted as e:
                    if llm.KEY_IDX not in exhausted:
                        print(f"\n[크레딧 소진] {llm.KEY_IDX + 1}번 키: {e}")
                        exhausted.add(llm.KEY_IDX)
                    queue.appendleft(item)     # 다음 키로 이 문항부터 다시
                    continue
                except Exception as e:         # 진행 중인 문항은 마저 저장하고 멈춘다 (이미 쓴 크레딧을 버리지 않게)
                    if err is None:
                        err = e
                        print(f"\n[오류] {item['id']}: {e!r} — 진행 중인 문항을 저장하고 멈춥니다")
                        queue.clear()
                    continue
                old = recs.get(item["id"])
                cost = common.record_cost(rec) - (common.record_cost(old) if old else 0.0)
                rec["spend"] = common.spend_map(old) if old else {}
                k = str(llm.KEY_IDX)               # 진행 중엔 키를 바꾸지 않으므로 지금 키가 이 문항에 쓴 키
                rec["spend"][k] = rec["spend"].get(k, 0.0) + cost
                SESSION_BY_KEY[llm.KEY_IDX] = SESSION_BY_KEY.get(llm.KEY_IDX, 0.0) + cost
                n_run += 1
                per_item[0] = max(per_item[0], cost) if n_run == 1 else (per_item[0] * (n_run - 1) + cost) / n_run
                if item["id"] not in recs:
                    order.append(item["id"])
                recs[item["id"]] = rec
                save(out_path, order, recs)
                n_done += 1
                c = rec["cond"]
                vs = [v for v in P.VARIANTS if v in rec["prompts"]]
                acc = {t: sum(c[f"{v}|raw|{t}"]["correct"] for v in vs) for t in TIERS}
                flags = [k for k, ok in rec["checks"].items() if not ok]
                print(f"  [{n_done:>3}/{len(items)}] {item['id']} {item['split']:<5} "
                      f"정답(raw, {len(vs)}변형) 저{acc['cheap']} 중{acc['mid']} 고{acc['expensive']}"
                      f"  이 문항 {cost:,.0f}  누적 {spent():,.0f} 크레딧" + (f"  ⚠ {flags}" if flags else ""))
    if err is not None:
        raise err

    print(f"\n저장 → {out_path}  ({time.time() - t0:.0f}초)")
    print(f"이번 실행 호출 {USAGE['calls']}회 · 입력 {USAGE['tokens_in']:,} · 출력 {USAGE['tokens_out']:,} 토큰"
          f" · 추정 사용 {spent():,.0f} 크레딧")
    print(f"다음: python3 analyze.py {OUT}")


if __name__ == "__main__":
    main()
