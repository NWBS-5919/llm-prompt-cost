"""고가 모델의 사고(thinking) 모드를 끄는 옵션 찾기 — 짧은 호출 몇 번 (최대 약 60 크레딧)"""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parent)   # 어디서 실행해도 이 폴더 기준으로
import json, llm

Q = ("파벨은 16살이다. 6년 후에 그의 여동생 아이샤는 현재 파벨 나이의 3배가 된다. "
     "5년 후 파벨과 아이샤의 나이의 평균은 얼마인가?\n\n단계별로 풀고, 최종 답만 마지막 줄에 '정답: ' 형식으로 써라.")
MSG = [{"role": "user", "content": Q}]

print("── 연결 테스트 때 저가·중가도 사고 모드였나 (기록에서, 무료)")
r = json.loads(open("runs_test.jsonl").readline())
for t in ("cheap", "mid", "expensive"):
    u = r["cond"][f"base|raw|{t}"]["usage"]
    print(f"  {t:<9} 출력 {u.get('completion_tokens')}  그중 사고 {(u.get('completion_tokens_details') or {}).get('reasoning_tokens')}")

TRIES = [
    ("qwen3.8-max", "옵션 없음", {}),
    ("qwen3.8-max", "enable_thinking=false", {"enable_thinking": False}),
    ("qwen3.8-max", "chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
    ("qwen3.8-max", "thinking disabled", {"thinking": {"type": "disabled"}}),
    ("qwen3.8-max", "reasoning_effort=none", {"reasoning_effort": "none"}),
    ("qwen3.7-max", "옵션 없음 (대체 후보)", {}),
]
print("\n── 고가 모델에 옵션별로 한 번씩")
for model, name, extra in TRIES:
    try:
        text, a, b, u = llm._post(model, MSG, 0.0, extra)
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        ok = "34" in text.split("정답")[-1]
        print(f"  {model:<12} {name:<24} 출력 {b:>5}  사고 {rt}  정답 {'O' if ok else 'X'}")
    except Exception as e:
        print(f"  {model:<12} {name:<24} 실패: {str(e)[:120]}")
print("\n위 결과를 Claude 에게 붙여 주세요.")
