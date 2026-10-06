"""
D-40 사람 질문 판정의 두 번째 판정자 — qwen3.8-max 가 같은 기준으로 60개를 따로 판정한다 (첫 판정자는 Claude 직접, 결과 보기 전에 고정).

기준 (검토/사람질문_판정_안내.md 와 같음)
  ① 같은 정답이 나오는가: 사람 질문만 보고 풀었을 때 원래 문제와 같은 답이면 '예'. 답에 필요 없는 정보가 빠진 건 '예'.
     답에 필요한 숫자·조건이 빠지거나 바뀌었거나, 다른 계산으로 자연스럽게 읽히면 '아니오'.
  ② 오타 개수: 맞춤법·자모 오류와 깨진 글자만. 띄어쓰기·구어체·줄임말은 세지 않는다.

출력 → 검토/사람질문_판정_qwen.json (번역이 들어 있어 저장소에 올리지 않는다)
실행: python3 human_judge.py            (2번 키, 60회 호출)
      python3 human_judge.py --report   (두 판정의 일치도와 분석)
"""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parent)
import csv, json, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
import llm
import common

OUT = _pl.Path("검토/사람질문_판정_qwen.json")
MINE = _pl.Path("검토/사람질문_판정_Claude.json")
SYS = ("너는 수학 문제 질문의 품질을 판정하는 평가자다. [원래 문제]와 [정답], 사람이 AI에게 물으려고 쓴 [사람 질문]이 주어진다.\n"
       "① same: [사람 질문]만 보고 풀었을 때 [정답]과 같은 답이 나오면 '예'. 답에 필요 없는 정보가 빠진 것은 '예'. "
       "답에 필요한 숫자나 조건이 빠지거나 바뀌었거나, 다른 계산으로 자연스럽게 읽힐 수 있으면 '아니오'.\n"
       "② typos: [사람 질문]의 맞춤법·자모 오류와 깨진 글자(예: 기호 대신 '_')의 개수. 띄어쓰기, 구어체 말투, 줄임말은 세지 않는다.\n"
       '반드시 JSON 한 줄로만 답하라: {"same": "예" 또는 "아니오", "typos": 정수, "reason": "한 문장"}')
_LOCK = threading.Lock()


def items():
    rows = list(csv.DictReader(open("human_TODO.csv", encoding="utf-8-sig")))
    gold = {i["id"]: i["gold"] for i in json.load(open("items.json"))}
    ex = common.excluded()
    for r in rows:
        if r["id"] in ex:
            continue
        for w in ("A", "B"):
            t = (r.get(f"writer_{w}") or "").strip()
            if t:
                yield {"id": r["id"], "writer": w, "en": r["original_question"], "gold": gold[r["id"]], "human": t}


def judge(x):
    msg = f"[원래 문제]\n{x['en']}\n\n[정답]\n{x['gold']}\n\n[사람 질문]\n{x['human']}"
    out, a, b, _ = llm.chat("expensive", [{"role": "system", "content": SYS}, {"role": "user", "content": msg}],
                            tag="human_judge")
    m = re.search(r"\{.*\}", out, re.S)
    d = json.loads(m.group(0)) if m else {}
    return {"id": x["id"], "writer": x["writer"], "same_answer": int(d.get("same") == "예"),
            "typos": int(d.get("typos", -1)), "note": d.get("reason", out[:120]), "tokens": [a, b]}


def run():
    llm.check_models()
    llm.use_key(int(_os.environ.get("REGRADE_KEY") or 1))
    xs = list(items())
    with ThreadPoolExecutor(4) as pool:
        res = list(pool.map(judge, xs))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    cost = sum(common.usd("expensive", *r["tokens"]) for r in res)
    print(f"  {len(res)}건 저장 → {OUT} · 약 {cost:.0f} 크레딧")


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def report():
    mine = {(x["id"], x["writer"]): x for x in json.loads(MINE.read_text())}
    qw = {(x["id"], x["writer"]): x for x in json.loads(OUT.read_text())}
    keys = sorted(mine)
    a = [mine[k]["same_answer"] for k in keys]
    b = [qw[k]["same_answer"] for k in keys]
    po, k = kappa(a, b)
    print(f"① 같은 정답: Claude 예 {sum(a)}/60 · qwen 예 {sum(b)}/60 · 일치 {po:.0%} · κ={k:.2f}")
    for kk in keys:
        if mine[kk]["same_answer"] != qw[kk]["same_answer"]:
            print(f"   불일치 {kk}: Claude {'예' if mine[kk]['same_answer'] else '아니오'} / qwen {'예' if qw[kk]['same_answer'] else '아니오'} — {qw[kk]['note']}")
    ta = [mine[k]["typos"] for k in keys]
    tb = [qw[k]["typos"] for k in keys]
    print(f"② 오타: Claude 합 {sum(ta)} · qwen 합 {sum(tb)} · 개수 일치 {sum(x == y for x, y in zip(ta, tb))}/60")
    for kk, x, y in zip(keys, ta, tb):
        if x != y:
            print(f"   {kk}: Claude {x} / qwen {y} — {qw[kk]['note']}")
    analysis(mine, qw)


AMBIG = {"m016", "m046", "m061", "m065", "m090", "m097"}      # 해석이 갈리는 문항 (민감도 분석용, D-38 ①)


def analysis(mine, qw):
    """D-38 ③ — 현실 오타율, 그리고 두 판정자 모두 '같은 정답' 인 사람 질문만 기준선과 비교"""
    from scipy.stats import binomtest
    R = {r["id"]: r for r in common.load_for_analysis("runs.jsonl")}
    print("\n③ 현실 오타율 (평소 말투로 쓴 질문) — 비교: 합성 오타 5% / 10% / 20%")
    for w in ("A", "B"):
        ks = [k for k in mine if k[1] == w]
        hang = sum(len(re.findall(r"[가-힣]", R[k[0]]["prompts"][f"HUMAN_{w}"])) for k in ks)
        typ = sum(mine[k]["typos"] for k in ks)
        print(f"   작성자 {w}: 질문 {len(ks)} · 오타 {typ}개 · 한글 {hang}자 → 글자당 {typ / hang:.2%} · 오타 있는 질문 {sum(mine[k]['typos'] > 0 for k in ks)}")
    for title, drop in (("④ 같은 정답인 사람 질문만 (두 판정자 모두 '예')", set()),
                        ("⑤ 민감도: ④ 에서 해석이 갈리는 6문항도 뺌", AMBIG)):
        print(f"\n{title} — 그대로 질문, 기준선 대비 (McNemar)")
        for w in ("A", "B"):
            ks = [k for k in mine if k[1] == w and mine[k]["same_answer"] and qw[k]["same_answer"] and k[0] not in drop]
            line = f"   작성자 {w} n={len(ks)}:"
            for t, nm in (("cheap", "저가"), ("mid", "중가"), ("expensive", "고가")):
                h = [R[k[0]]["cond"][f"HUMAN_{w}|raw|{t}"]["correct"] for k in ks]
                bb = [R[k[0]]["cond"][f"base|raw|{t}"]["correct"] for k in ks]
                lose = sum(y and not x for x, y in zip(h, bb)); gain = sum(x and not y for x, y in zip(h, bb))
                p = binomtest(lose, lose + gain, 0.5).pvalue if lose + gain else 1.0
                line += f"  {nm} {sum(h) / len(h):.0%} vs {sum(bb) / len(bb):.0%} ({lose}/{gain}, p={p:.2f})"
            print(line)


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
