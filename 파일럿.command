#!/bin/bash
# 더블클릭하면 파일럿(30문항)을 돌리고 결과를 파일럿_결과.txt 로 저장한다
cd "$(dirname "$0")"
python3 run_main.py --limit 30 && python3 analyze.py runs.jsonl | tee 파일럿_결과.txt
echo; echo "끝. 파일럿_결과.txt 를 Claude 에게 붙여 주세요."; read -n 1 -s -r -p "아무 키나 누르면 닫힙니다"
