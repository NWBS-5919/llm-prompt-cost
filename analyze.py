"""
파일럿 v4 · 분석  (축별 대가 분해)

  ① 축별 대가 — 어느 엉성함이 제일 비싼가  ★ 이번 판의 핵심
  ② 언어의 대가 — 한국어라서 치르는 값
  ③ 합성 vs 자연 — 합성 변형이 사람이 쓴 것을 재현하는가
  ④ 심판 검증
  ⑤ 개방형 체크리스트 (축별)
  ⑥ 판정

실행: python3 analyze.py
"""
import json, pathlib
from scipy import stats
import perturb as P

HERE = pathlib.Path(__file__).parent
rows = [json.loads(l) for l in open(HERE / "runs.jsonl", encoding="utf-8")]
anchor = [r for r in rows if r["kind"] == "anchor"]
openx = [r for r in rows if r["kind"] == "open"]
L = "=" * 82


def m(rs, cond, mk, field):
    v = [r["cond"][f"{mk}/{cond}"][field] for r in rs
         if f"{mk}/{cond}" in r["cond"] and r["cond"][f"{mk}/{cond}"][field] is not None]
    return sum(v) / len(v) if v else float("nan")


# ─────────────────────────────────────────────────────────── ①
print(L); print("① 축별 대가 — 기준선 대비 (cheap 모델)"); print(L)
print("  부호 규약 — '대가' 는 기준선 대비 나빠진 정도다. **양수 = 손해, 음수 = 오히려 나음.**")
print()
print(f"{'축':<6}{'가족':<6}{'이름':<16}{'정답률':>8}{'대가':>9}"
      f"{'토큰':>8}{'배율':>8}{'턴':>7}")
print("-" * 82)

MK = "cheap"
base_a = m(anchor, "ko_expert", MK, "correct")
baset_a = m(anchor, "ko_expert", MK, "tokens_total")
print(f"{'기준':<6}{'—':<6}{'한국어·정제':<16}{base_a:>7.0%}{'—':>9}"
      f"{baset_a:>8.0f}{'1.00x':>8}{m(anchor,'ko_expert',MK,'turns'):>7.2f}")

axis_rows = []
for a in P.ANCHOR_AXES:
    acc = m(anchor, a, MK, "correct")
    tok = m(anchor, a, MK, "tokens_total")
    trn = m(anchor, a, MK, "turns")
    cost = (base_a - acc) * 100                      # 양수 = 손해
    axis_rows.append((a, cost, tok / baset_a if baset_a else 0))
    print(f"{a:<6}{P.FAMILY[a]:<6}{P.LABEL[a]:<16}{acc:>7.0%}"
          f"{cost:>+8.0f}p{tok:>8.0f}{tok/baset_a if baset_a else 0:>7.2f}x{trn:>7.2f}")

nat_a = m(anchor, "ko_human", MK, "correct")
natt_a = m(anchor, "ko_human", MK, "tokens_total")
nat_cost = (base_a - nat_a) * 100
print(f"{'복합':<6}{'—':<6}{'사람이 쓴 것':<16}{nat_a:>7.0%}{nat_cost:>+8.0f}p"
      f"{natt_a:>8.0f}{natt_a/baset_a if baset_a else 0:>7.2f}x"
      f"{m(anchor,'ko_human',MK,'turns'):>7.2f}")

print()
print("  가족별 평균 대가 (양수 = 손해)")
for fam in ("표면", "결핍", "과잉"):
    sel = [(d, t) for a, d, t in axis_rows if P.FAMILY[a] == fam]
    if sel:
        print(f"    {fam}  정답률 대가 {sum(d for d,_ in sel)/len(sel):>+5.1f}%p  "
              f"토큰 {sum(t for _,t in sel)/len(sel):>4.2f}배")

worst_acc = max(axis_rows, key=lambda x: x[1]) if axis_rows else None
worst_tok = max(axis_rows, key=lambda x: x[2]) if axis_rows else None
if worst_acc:
    print(f"\n  ▸ 정답률을 가장 깎은 축   {worst_acc[0]} {P.LABEL[worst_acc[0]]}"
          f"  — {worst_acc[1]:.0f}%p 하락")
    print(f"  ▸ 토큰을 가장 늘린 축     {worst_tok[0]} {P.LABEL[worst_tok[0]]}"
          f"  — {worst_tok[2]:.2f}배")

# ─────────────────────────────────────────────────────────── ②
print(); print(L); print("② 언어의 대가 — 영어 정제 vs 한국어 정제"); print(L)
for mk in ("cheap", "expensive"):
    ae, ak = m(anchor, "en_expert", mk, "correct"), m(anchor, "ko_expert", mk, "correct")
    te, tk = m(anchor, "en_expert", mk, "tokens_total"), m(anchor, "ko_expert", mk, "tokens_total")
    x = sum(1 for r in anchor if r["cond"][f"{mk}/en_expert"]["correct"]
            and not r["cond"][f"{mk}/ko_expert"]["correct"])
    y = sum(1 for r in anchor if not r["cond"][f"{mk}/en_expert"]["correct"]
            and r["cond"][f"{mk}/ko_expert"]["correct"])
    p = stats.binomtest(x, x + y, 0.5).pvalue if (x + y) else 1.0
    print(f"  [{mk:<9}] 정답률 한국어 {ak:.0%} → 영어 {ae:.0%} ({(ae-ak)*100:+.0f}%p, p={p:.3f})"
          f" · 토큰 {tk/te if te else 0:.2f}x (한국어/영어)")

# ─────────────────────────────────────────────────────────── ③
print(); print(L); print("③ 합성 vs 자연 — 합성 변형이 사람이 쓴 것을 재현하는가"); print(L)
worst = min((m(anchor, a, MK, "correct") for a in P.ANCHOR_AXES), default=float("nan"))
print(f"  사람이 쓴 것        정답률 {nat_a:.0%} · 토큰 {natt_a/baset_a if baset_a else 0:.2f}x")
print(f"  단일 축 중 최악     정답률 {worst:.0%}")
print(f"  기준선              정답률 {base_a:.0%}")
print()
if nat_a < worst:
    print("  → 사람이 쓴 것이 어떤 단일 축보다 나쁘다. 여러 축이 겹쳐 있다는 뜻이며,")
    print("    '합성 축의 단순 합으로는 사람의 엉성함을 재현하지 못한다'가 결과가 된다.")
else:
    print("  → 사람이 쓴 것이 단일 축 수준이다. 합성 변형으로 대체 가능하다는 근거.")

# ─────────────────────────────────────────────────────────── ④
print(); print(L); print("④ 심판 검증 — 정답을 아는 문항에서 심판이 정답 쪽을 골랐는가"); print(L)
tot = hit = 0
flip = sum(1 for r in anchor for mk in ("cheap", "expensive")
           if not r["judge"][mk]["order_consistent"])
for r in anchor:
    for mk in ("cheap", "expensive"):
        lo, hi = r["judge"][mk]["pair"]
        cl, ch = r["cond"][f"{mk}/{lo}"]["correct"], r["cond"][f"{mk}/{hi}"]["correct"]
        if cl == ch:
            continue
        tot += 1
        if r["judge"][mk]["winner"] == (hi if ch else lo):
            hit += 1
print(f"  판정 가능한 쌍 {tot}개 중 {hit}개 일치 → 심판 정확도 "
      f"{(hit/tot if tot else float('nan')):.0%}" if tot else "  판정 가능한 쌍 없음")
print(f"  위치 편향(순서 뒤집으면 판정이 바뀜) {flip}/{len(anchor)*2} = "
      f"{flip/max(len(anchor)*2,1):.0%}")
print("  ≥85% 진행 · 70~85% 심판 손보기 · <70% 개방형 수치를 논문에 싣지 말 것")

# ─────────────────────────────────────────────────────────── ⑤
print(); print(L); print("⑤ 개방형 — 축별 체크리스트 충족률 (cheap)"); print(L)


def rub(cond):
    v = []
    for r in openx:
        s = r["judge"][MK].get("rubric", {}).get(cond)
        if s:
            v.append(sum(s) / len(s))
    return sum(v) / len(v) if v else float("nan")


bt = m(openx, "refined", MK, "tokens_total")
print(f"{'축':<6}{'이름':<16}{'체크리스트':>11}{'Δ':>8}{'토큰':>9}{'배율':>8}")
print("-" * 82)
rb = rub("refined")
print(f"{'기준':<6}{'재작성본':<16}{rb:>10.0%}{'—':>8}{bt:>9.0f}{'1.00x':>8}")
for a in P.OPEN_AXES:
    s, t = rub(a), m(openx, a, MK, "tokens_total")
    print(f"{a:<6}{P.LABEL[a]:<16}{s:>10.0%}{(s-rb)*100:>+7.0f}p"
          f"{t:>9.0f}{t/bt if bt else 0:>7.2f}x")
no, to = rub("original"), m(openx, "original", MK, "tokens_total")
print(f"{'복합':<6}{'사용자 원문':<16}{no:>10.0%}{(no-rb)*100:>+7.0f}p"
      f"{to:>9.0f}{to/bt if bt else 0:>7.2f}x")

# ─────────────────────────────────────────────────────────── ⑥
print(); print(L); print("⑥ 판정"); print(L)
print("  1차 기준은 **축**이다. 축 변형은 기준선에 결정론적 규칙·고정 프롬프트를 적용해")
print("  만들므로 작성자 편향이 원리적으로 불가능하다. 사람이 쓴 문장은 보조 증거로 읽는다.")
print()

if axis_rows:
    best = max(axis_rows, key=lambda x: x[1])        # 정답률 대가가 가장 큰 축
    tok_max = max(axis_rows, key=lambda x: x[2])     # 토큰 배율이 가장 큰 축
    live = [a for a, d, t in axis_rows if d >= 3 or t >= 1.15]

    print(f"  정답률 대가 최대   {best[0]} {P.LABEL[best[0]]}  {best[1]:+.1f}%p")
    print(f"  토큰 배율 최대     {tok_max[0]} {P.LABEL[tok_max[0]]}  {tok_max[2]:.2f}x")
    print(f"  신호가 있는 축     {len(live)}/{len(axis_rows)}개 "
          f"{live if live else '— 없음'}")
    print()

    if best[1] >= 10:
        print(f"  → ★ 어떤 엉성함이 비싼지가 뚜렷하다. 이것을 주 결과로 보고한다.")
    elif best[1] >= 3 or tok_max[2] >= 1.15:
        print(f"  → ◎ 정답률이든 토큰이든 한쪽에 신호가 있다. 그 축을 주 결과로 보고한다.")
    else:
        print(f"  → ▲ 축별 정답률 신호가 약하다. 토큰·개방형 체크리스트로 무게를 옮긴다.")
        print(f"     '최신 모델은 엉성한 질문에 강건하다'도 그 자체로 결과다.")

print()
print(f"  [보조] 사람이 쓴 것의 대가 {nat_cost:+.1f}%p")
if axis_rows:
    worst_axis = min(m(anchor, a, MK, "correct") for a in P.ANCHOR_AXES)
    if nat_a < worst_axis:
        print("     어떤 단일 축보다 나쁘다 → 여러 축이 겹쳐 있다는 뜻.")
        print("     '합성 축의 단순 합으로는 사람의 엉성함을 재현하지 못한다'가 결과가 된다.")
    else:
        print("     단일 축 수준이다 → 합성 변형으로 대체 가능하다는 근거.")
print("     주의 — 작성자가 저자 본인이므로 요구특성 편향이 있다. 주 결과로 쓰지 말 것.")

print()
print("  축을 더 살릴 때 고르는 기준")
print("   · 정답률 대가 3%p 이상 또는 토큰 배율 1.15x 이상")
print("   · 가족별로 최소 한 개씩 남겨 세 가족을 모두 대표하게 한다")
print("   · 신호가 없는 축도 결과다 — '오타는 대가가 없다'는 서비스 메시지가 된다")
