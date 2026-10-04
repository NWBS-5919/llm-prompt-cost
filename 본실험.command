#!/bin/bash
# G1 통과 후 더블클릭: 나머지 문항까지 이어서 실행 → 분석 → Part B
cd "$(dirname "$0")"
python3 run_main.py && python3 analyze.py runs.jsonl | tee 본실험_결과.txt && python3 partc.py runs.jsonl | tee -a 본실험_결과.txt
echo; echo "끝. 본실험_결과.txt 를 Claude 에게 붙여 주세요."; read -n 1 -s -r -p "아무 키나 누르면 닫힙니다"
