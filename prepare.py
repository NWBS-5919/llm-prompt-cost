"""
파일럿 v2 · 준비
  앵커(정답 있음) 10개 + 개방형(정답 없음) 10개 = 20문항

  앵커   : 벤치마크 원문(잘 쓴 질문) → 팀원이 '대충 쓴 질문'을 작성   [→ CSV]
  개방형 : 실제 사용자풍 질문(대충)  → 재작성기가 '다듬은 질문'을 생성 [→ 실행 시]

실행:  python3 prepare.py
"""
import json, random, csv, pathlib
from openended_items import OPEN_ENDED

SEED = 20260922
N_GSM = N_MBPP = 5
HERE = pathlib.Path(__file__).parent
rng = random.Random(SEED)


def load_gsm8k():
    out = []
    for l in open(HERE / "gsm8k_test.jsonl", encoding="utf-8"):
        r = json.loads(l)
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        try:
            float(gold)
        except ValueError:
            continue
        out.append({"task": "gsm8k", "question": r["question"].strip(), "gold": gold})
    return out


def load_mbpp():
    out = []
    for l in open(HERE / "mbpp.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if not r.get("test_list"):
            continue
        out.append({"task": "mbpp", "question": r["text"].strip(),
                    "tests": r["test_list"], "setup": r.get("test_setup_code") or "",
                    "ref_code": r.get("code", "")})
    return out


def stratified(pool, n, key=lambda x: len(x["question"])):
    pool = sorted(pool, key=key)
    half = len(pool) // 2
    return rng.sample(pool[:half], n // 2) + rng.sample(pool[half:], n - n // 2)


def main():
    anchors = stratified(load_gsm8k(), N_GSM) + stratified(load_mbpp(), N_MBPP)
    for i, it in enumerate(anchors):
        it["id"] = f"{it['task']}-{i:02d}"
        it["kind"] = "anchor"

    openended = []
    for it in OPEN_ENDED:
        it = dict(it)
        it["kind"] = "open"
        openended.append(it)

    items = anchors + openended
    (HERE / "items.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(HERE / "novice_TODO.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "task", "original_question", "novice_ko"])
        for it in anchors:
            w.writerow([it["id"], it["task"], it["question"], ""])

    print(f"문항 {len(items)}개 저장 → items.json")
    print(f"  앵커(정답 있음)  {len(anchors)}개 : 수학 {N_GSM} · 코딩 {N_MBPP}")
    print(f"  개방형(정답 없음) {len(openended)}개 : "
          f"요약 5 · 조언 2 · 계획 2 · 글쓰기 1")
    print()
    print("사람이 할 일은 하나뿐입니다.")
    print("  novice_TODO.csv 의 novice_ko 열에, 앵커 10문항을 '평소 말투'로 다시 쓰세요.")
    print("  · 원문을 번역하지 말고, 내가 AI에 물어본다면 어떻게 칠지 그대로")
    print("  · '단계별로 풀어라' 같은 지시문을 넣지 않기")
    print("  · 맞춤법 고치지 않기")
    print()
    print("개방형 10문항은 이미 '엉성한 질문' 상태이므로 작성할 필요가 없습니다.")
    print("  (실행 시 재작성기가 다듬은 버전을 자동 생성합니다)")


if __name__ == "__main__":
    main()
