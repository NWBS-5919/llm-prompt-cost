#!/bin/bash
# 무과금 배관 점검 — API 를 부르지 않는다
cd "$(dirname "$0")"
rm -f runs_fake.jsonl
python3 run_main.py --fake && python3 analyze.py runs_fake.jsonl > /dev/null && python3 partc.py runs_fake.jsonl --days 10 | tail -1
echo; echo "위에 '자동 검사: 사후 최적 초과 0건' 이 보이면 정상"; read -n 1 -s -r -p "아무 키나 누르면 닫힙니다"
