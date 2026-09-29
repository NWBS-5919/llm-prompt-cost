"""
파일럿 v4 · 실행  (엉성함 8축 분해판)

구조는 앵커와 개방형이 동일하다.

    기준선   잘 쓴 한국어 질문
    축 변형  기준선을 한 가지 방식으로만 망가뜨린 것 (S1~E2)
    자연복합 사람이 실제로 쓴 것 — 여러 축이 뒤섞인 현실

      앵커     기준선 ko_expert(모델 번역) · 자연복합 ko_human(팀원 작성)
               보너스 en_expert — 언어 효과 측정용
      개방형   기준선 refined(재작성기) · 자연복합 original(사용자풍 원문)

핵심 비교
    축별 대가   기준선 vs 각 축          → 어느 엉성함이 제일 비싼가
    언어 대가   en_expert vs ko_expert   → 한국어라서 치르는 값
    합성 타당성 축들의 합 vs 자연복합      → 합성이 사람을 재현하는가

실행
  python3 run_pilot.py --fake     # 무과금 배관 점검
  python3 run_pilot.py            # 실제 실행
"""
import json, sys, csv, time, pathlib
import llm
from llm import chat, USAGE
from grade import grade
import judge as J
import perturb as P

HERE = pathlib.Path(__file__).parent
llm.FAKE = "--fake" in sys.argv
MAX_TURNS = 2
FOLLOWUP = "답이 충분하지 않습니다. 다시 확인해서 제대로 답해 주세요."
SYSTEM = "You are a helpful assistant."

KO_INSTR = "\n\n단계별로 풀고, 최종 답만 마지막 줄에 '정답: ' 형식으로 써라."
EN_INSTR = "\n\nLet's think step by step."

TRANSLATE_SYS = ("다음 문제를 한국어로 자연스럽게 옮겨라. "
                 "숫자와 조건을 하나도 빠뜨리지 말고, 문제를 풀지 마라. 번역문만 출력한다.")
REWRITE_SYS = ("너는 사용자의 질문을 AI가 더 잘 처리할 수 있게 다듬는 도구다. "
               "질문의 의도를 절대 바꾸지 말고, 목적·원하는 형식·분량·빠진 맥락만 보강한다. "
               "질문에 답하지 말고, 다듬은 질문만 출력한다.")


def build_variants(item, human_map):
    """{조건명: 프롬프트}, 기준선 이름, 자연복합 이름, 심판에게 줄 의도, 준비 비용

    준비 비용(번역·재작성 호출의 토큰)을 함께 돌려준다. 재작성 토큰은 Part B의
    총비용에 들어가므로 버리면 손익분기를 사후에 계산할 수 없다.

    축 변형은 **질문 본문에만** 적용하고 채점 지시문은 그 뒤에 붙인다.
    지시문까지 망가뜨리면 모델이 '정답:' 형식을 안 써서 grade.py 가 폴백으로
    떨어지고, 축의 대가와 채점 실패가 섞인다. 지시문 제거는 D2 전용 축이다.
    """
    if item["kind"] == "anchor":
        ko, tin, tout = chat("judge", [{"role": "system", "content": TRANSLATE_SYS},
                                       {"role": "user", "content": item["question"]}],
                             item=item, tag="translate")
        body = ko.strip()
        setup = {"translate": {"tokens_in": tin, "tokens_out": tout}}
        v = {"en_expert": item["question"] + EN_INSTR, "ko_expert": body + KO_INSTR}
        for a in P.ANCHOR_AXES:
            # D2(지시 부재) = 지시문을 붙이지 않는다. 나머지는 본문만 변형 후 지시문 유지
            v[a] = body if a == "D2" else P.apply_axis(a, body, item) + KO_INSTR
        v["ko_human"] = human_map.get(item["id"], item["question"][:60])
        return v, "ko_expert", "ko_human", item["question"], setup

    original = item["user_prompt"]
    if item["task"] == "summarize":
        original = item["doc"] + "\n\n" + original
    ref, tin, tout = chat("rewriter", [{"role": "system", "content": REWRITE_SYS},
                                       {"role": "user", "content": original}],
                          item=item, tag="rewrite")
    setup = {"rewrite": {"tokens_in": tin, "tokens_out": tout}}
    base = ref.strip()
    v = {"refined": base}
    for a in P.OPEN_AXES:
        v[a] = P.apply_axis(a, base, item)
    v["original"] = original
    return v, "refined", "original", item["user_prompt"], setup


def run_condition(item, text, model_key, cond=""):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    tin = tout = 0
    correct = None
    final = ""
    turns = 1
    for t in range(MAX_TURNS):
        # cond 는 --fake 가 축을 구분하는 데만 쓰인다. 실제 호출에는 영향이 없다.
        out, a, b = chat(model_key, msgs, item=item, tag=f"cond:{cond}")
        tin += a; tout += b; final = out; turns = t + 1
        if item["kind"] != "anchor":
            break
        correct = grade(item, out)
        if correct or t == MAX_TURNS - 1:
            break
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": FOLLOWUP}]
    return {"answer": final, "correct": correct, "turns": turns,
            "tokens_in": tin, "tokens_out": tout, "tokens_total": tin + tout}


def main():
    items = json.loads((HERE / "items.json").read_text(encoding="utf-8"))
    human = {}
    csvp = HERE / "novice_TODO.csv"
    if csvp.exists():
        with open(csvp, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("novice_ko", "").strip():
                    human[row["id"]] = row["novice_ko"].strip()

    anchors = [i for i in items if i["kind"] == "anchor"]
    missing = [i["id"] for i in anchors if i["id"] not in human]
    if missing and not llm.FAKE:
        print(f"[중단] novice_TODO.csv 에 {len(missing)}개가 비어 있습니다: {missing}")
        sys.exit(1)

    out_path = HERE / "runs.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")} \
        if out_path.exists() else set()

    with open(out_path, "a", encoding="utf-8") as f:
        for n, item in enumerate(items, 1):
            if item["id"] in done:
                print(f"  [{n:>2}/{len(items)}] {item['id']:<10} 건너뜀(완료)")
                continue

            variants, base_k, nat_k, intent, setup = build_variants(item, human)
            rec = {"id": item["id"], "task": item["task"], "kind": item["kind"],
                   "base": base_k, "natural": nat_k, "intent": intent,
                   "axes": list(P.ACTIVE_AXES),
                   "prompts": variants,
                   "prompt_len": {k: len(v) for k, v in variants.items()},
                   "setup": setup,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "cond": {}, "judge": {}}

            for mk in ("cheap", "expensive"):
                for c, text in variants.items():
                    rec["cond"][f"{mk}/{c}"] = run_condition(item, text, mk, c)

            for mk in ("cheap", "expensive"):
                a = rec["cond"][f"{mk}/{nat_k}"]["answer"]
                b = rec["cond"][f"{mk}/{base_k}"]["answer"]
                w, ok = J.pairwise(intent, a, b, item)
                rec["judge"][mk] = {
                    "pair": [nat_k, base_k],
                    "winner": {"first": nat_k, "second": base_k, "tie": "tie"}[w],
                    "order_consistent": ok}
                if item["kind"] == "open" and item.get("rubric"):
                    rec["judge"][mk]["rubric"] = {
                        c: J.rubric_score(intent, rec["cond"][f"{mk}/{c}"]["answer"],
                                          item["rubric"], item)
                        for c in variants}

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            c = rec["cond"]
            tb = c[f"cheap/{base_k}"]["tokens_total"]
            tn = c[f"cheap/{nat_k}"]["tokens_total"]
            extra = ""
            if item["kind"] == "anchor":
                ok = sum(1 for k in variants if c[f"cheap/{k}"]["correct"])
                extra = f"정답 {ok}/{len(variants)}조건"
            print(f"  [{n:>2}/{len(items)}] {item['id']:<10} {item['task']:<10} "
                  f"조건 {len(variants):>2}  토큰 기준{tb:>5}/자연{tn:>5}  {extra}")

    print(f"\n저장 완료 → {out_path}")
    print(f"총 호출 {USAGE['calls']}회 · 입력 {USAGE['tokens_in']:,} · "
          f"출력 {USAGE['tokens_out']:,} 토큰")
    print("다음: python3 analyze.py")


if __name__ == "__main__":
    main()
