#!/bin/bash
# 1문항만 실제로 돌려 연결·모델·토큰을 확인한다 (약 52회 호출)
cd "$(dirname "$0")"
rm -f runs_test.jsonl
python3 run_main.py --limit 1 --out runs_test.jsonl || { echo; echo "실패 — 위 에러 메시지를 Claude 에게 붙여 주세요"; read -n 1 -s -r; exit 1; }
python3 - <<'PY'
import json, common
r = json.loads(open("runs_test.jsonl").readline())
print("\n── 연결 테스트 결과 (1문항) ──")
print("번역 기준선:", r["prompts"]["base"][:120].replace("\n", " "), "...")
for t in ("cheap", "mid", "expensive"):
    c = r["cond"][f"base|raw|{t}"]
    print(f"{t:<9} 정답 {c['correct']}  입력 {c['tokens_in']}  출력 {c['tokens_out']}  usage {json.dumps(c['usage'], ensure_ascii=False)[:160]}")
tot = sum(common.cell(r, v, a, t)[2] for v in r["prompts"] for a in common.ACTS for t in common.TIERS)
print(f"\n이 문항 하나의 답변 비용 ≈ {tot:.1f}  → 100문항 ≈ {tot*100:,.0f} (env.txt 단가 단위 = ChatKHU 크레딧, 번역·재작성 제외 추정)")
PY
echo; echo "위 결과 전체를 Claude 에게 붙여 주세요."; read -n 1 -s -r -p "아무 키나 누르면 닫힙니다"
