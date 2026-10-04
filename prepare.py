"""
v5 · 문항 준비

  수학 N문항 (기본 120, GSM-Symbolic P1 은 원문이 100개라 100)
    1순위  data/gsm_symbolic_p1.jsonl  — apple/GSM-Symbolic 의 p1/test.jsonl 을 이 이름으로 저장
    대안    gsm8k_test.jsonl 중 문장이 긴 절반
  같은 원문에서 나온 변형은 하나만 쓴다 (instance 0 우선)

  출력
    items.json        문항 + split(train/test) + 길이 구간(len_bucket: lo/hi)
    human_TODO.csv    시험 절반에서 30문항 — 두 사람이 각자 writer_A / writer_B 열을 채운다

실행:  python3 prepare.py            # 120문항
       python3 prepare.py --n 60     # 문항 수 지정
※ GSM-Symbolic 은 CC BY-NC-ND 4.0 — 번역·변형본을 외부에 배포하지 말 것 (내부 평가용만)
"""
import json, random, csv, pathlib, sys

SEED = 20261004
HERE = pathlib.Path(__file__).parent
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 120
N_HUMAN = 30


def _gold(ans):
    g = ans.split("####")[-1].strip().replace(",", "")
    float(g)
    return g


def load_symbolic(path):
    seen, out = set(), []
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows.sort(key=lambda r: (r.get("instance", 0), str(r.get("id", ""))))
    for r in rows:
        key = r.get("original_id", r.get("id"))
        if key in seen:
            continue
        try:
            out.append({"question": r["question"].strip(), "gold": _gold(r["answer"]),
                        "source": "gsm_symbolic_p1", "orig": str(key)})
            seen.add(key)
        except (KeyError, ValueError):
            continue
    return out


def load_gsm8k_long(path):
    out = []
    for i, l in enumerate(open(path, encoding="utf-8")):
        r = json.loads(l)
        try:
            out.append({"question": r["question"].strip(), "gold": _gold(r["answer"]),
                        "source": "gsm8k", "orig": str(i)})
        except ValueError:
            continue
    out.sort(key=lambda x: len(x["question"]))
    return out[len(out) // 2:]


def main():
    global N
    rng = random.Random(SEED)
    sym = HERE / "data" / "gsm_symbolic_p1.jsonl"
    if sym.exists():
        pool = load_symbolic(sym)
        print(f"GSM-Symbolic P1 사용 ({len(pool)}개 원문)")
    else:
        pool = load_gsm8k_long(HERE / "gsm8k_test.jsonl")
        print(f"[대안] data/gsm_symbolic_p1.jsonl 없음 → GSM8K 긴 절반 사용 ({len(pool)}개)")
    if not pool:
        raise SystemExit("[중단] 문항을 하나도 못 읽었습니다. answer 열에 '#### 숫자' 형식이 있는지 확인하세요.")
    if len(pool) < N:
        print(f"  원문이 {len(pool)}개뿐이라 {N} → {len(pool)}문항으로 줄입니다")
        N = len(pool)

    items = rng.sample(pool, N)                 # 순서도 무작위 — 앞 30개가 파일럿이 된다
    med = sorted(len(x["question"]) for x in items)[N // 2]
    test_idx = set(rng.sample(range(N), N // 2))
    for i, it in enumerate(items):
        it["id"] = f"m{i:03d}"
        it["task"] = "gsm8k"
        it["kind"] = "anchor"
        it["split"] = "test" if i in test_idx else "train"
        it["len_bucket"] = "hi" if len(it["question"]) >= med else "lo"

    (HERE / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    human = [it for it in items if it["split"] == "test"][:N_HUMAN]
    hp = HERE / "human_TODO.csv"
    if hp.exists():
        print(f"human_TODO.csv 가 이미 있어 덮어쓰지 않습니다 (작성분 보호)")
    else:
        with open(hp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "original_question", "writer_A", "writer_B"])
            for it in human:
                w.writerow([it["id"], it["question"], "", ""])

    print(f"문항 {N}개 → items.json  (train {N - len(test_idx)} / test {len(test_idx)})")
    print(f"사람 작성 {len(human)}문항 → human_TODO.csv")
    print("  · writer_A 는 상연, writer_B 는 팀원. 서로 안 보고 각자 쓴다")
    print("  · 번역하지 말고, 내가 AI한테 물으면 이렇게 친다를 그대로 / 지시문 넣지 않기 / 맞춤법 고치지 않기")


if __name__ == "__main__":
    main()
