"""채점기 — GSM8K는 최종 숫자 비교, MBPP는 단위 테스트 실행"""
import re, subprocess, sys, tempfile, os, json

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text: str):
    """모델 답에서 최종 숫자를 뽑는다. '정답: 18', '**18**', '$18.00' 모두 허용."""
    if not text:
        return None
    # 1) '정답/답/answer' 뒤에 오는 숫자를 최우선
    for pat in (r"(?:최종\s*)?(?:정답|답)\s*[:：]?\s*\**\s*(-?\d[\d,]*\.?\d*)",
                r"(?:final\s*)?answer\s*(?:is)?\s*[:：]?\s*\**\s*(-?\d[\d,]*\.?\d*)"):
        m = re.findall(pat, text, flags=re.I)
        if m:
            return m[-1].replace(",", "")
    # 2) 없으면 마지막 숫자
    m = NUM.findall(text)
    return m[-1].replace(",", "") if m else None


def grade_gsm8k(output: str, gold: str) -> bool:
    got = extract_number(output)
    if got is None:
        return False
    try:
        return abs(float(got) - float(gold)) < 1e-4
    except ValueError:
        return False


CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    if not text:
        return ""
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len)
    # 코드블록이 없으면 def 가 나오는 줄부터 끝까지
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.lstrip().startswith(("def ", "import ", "from ", "class ")):
            return "\n".join(lines[i:])
    return text


def grade_mbpp(output: str, tests, setup: str = "", timeout: int = 8) -> bool:
    """생성 코드를 별도 프로세스에서 실행. 외부 모델 출력을 실행하므로
    반드시 격리된 환경(도커/가상머신 또는 개인 실습용 계정)에서 돌릴 것."""
    code = extract_code(output)
    if not code.strip():
        return False
    prog = "\n".join([code, "", setup or "", ""] + list(tests))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def grade(item, output: str) -> bool:
    if item["task"] == "gsm8k":
        return grade_gsm8k(output, item["gold"])
    return grade_mbpp(output, item["tests"], item.get("setup", ""))


if __name__ == "__main__":
    # 자가 점검
    assert grade_gsm8k("계산하면 정답: 18 입니다.", "18")
    assert grade_gsm8k("...따라서 **1,250**원", "1250")
    assert grade_gsm8k("The final answer is 42", "42")
    assert not grade_gsm8k("잘 모르겠습니다", "7")
    ok = grade_mbpp("```python\ndef add(a,b):\n    return a+b\n```",
                    ["assert add(1,2)==3"])
    bad = grade_mbpp("```python\ndef add(a,b):\n    return a-b\n```",
                     ["assert add(1,2)==3"])
    assert ok and not bad
    inf = grade_mbpp("```python\ndef f():\n    while True: pass\n```",
                     ["assert f()==1"], timeout=3)
    assert not inf
    print("채점기 자가 점검 통과 (숫자 추출 4건, 코드 실행 3건)")
