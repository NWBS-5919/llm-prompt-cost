"""
심판 — 정답이 없는 과제를 채점한다.

  pairwise : 두 답변 중 어느 쪽이 나은지. 위치 편향을 없애려 순서를 바꿔 두 번 묻는다.
  rubric   : 체크리스트 항목별 0/1 채점.

앵커 문항(정답 있음)에도 pairwise 를 돌려, 심판 판정이 실제 정답과
얼마나 일치하는지 측정한다. 그 일치율이 개방형 결과의 신뢰도가 된다.
"""
import json, re
from llm import chat

PAIR_SYS = (
    "너는 두 AI 답변을 비교하는 엄격한 평가자다. "
    "사용자의 원래 의도를 얼마나 충족했는지만 본다. "
    "길다고 좋은 것이 아니며, 불필요하게 장황하면 감점한다. "
    "반드시 A, B, 무승부 중 한 단어로만 답한다."
)

PAIR_TMPL = """[사용자가 원한 것]
{intent}

[답변 A]
{a}

[답변 B]
{b}

어느 쪽이 더 나은 답변인가? A, B, 무승부 중 하나만 쓰시오."""

RUBRIC_SYS = (
    "너는 체크리스트로 답변을 채점하는 평가자다. "
    "각 항목을 충족하면 1, 아니면 0을 매긴다. "
    "설명 없이 JSON 배열만 출력한다. 예: [1,0,1,1]"
)

RUBRIC_TMPL = """[사용자가 원한 것]
{intent}

[답변]
{ans}

[체크리스트]
{checks}

각 항목에 0 또는 1을 매겨 JSON 배열로만 출력하시오."""


def _parse_verdict(t):
    t = (t or "").strip()
    if re.search(r"무승부|tie|draw", t, re.I):
        return "tie"
    if re.match(r"^\W*A\b", t, re.I) or re.search(r"답변\s*A", t):
        return "A"
    if re.match(r"^\W*B\b", t, re.I) or re.search(r"답변\s*B", t):
        return "B"
    return "tie"


def pairwise(intent, ans_first, ans_second, item=None):
    """ans_first 가 이기면 'first', 지면 'second', 비기면 'tie'.
    순서를 바꿔 두 번 물어 일관될 때만 승패를 인정한다."""
    v1 = _parse_verdict(chat("judge", [
        {"role": "system", "content": PAIR_SYS},
        {"role": "user", "content": PAIR_TMPL.format(
            intent=intent, a=ans_first, b=ans_second)}], item=item, tag="judge_pair")[0])
    v2 = _parse_verdict(chat("judge", [
        {"role": "system", "content": PAIR_SYS},
        {"role": "user", "content": PAIR_TMPL.format(
            intent=intent, a=ans_second, b=ans_first)}], item=item, tag="judge_pair")[0])
    # 1차: A=first,B=second   2차: A=second,B=first
    w1 = {"A": "first", "B": "second", "tie": "tie"}[v1]
    w2 = {"A": "second", "B": "first", "tie": "tie"}[v2]
    if w1 == w2:
        return w1, True            # 순서를 바꿔도 같은 판정 → 신뢰
    return "tie", False            # 뒤집힘 → 위치 편향, 무승부 처리


def rubric_score(intent, answer, rubric, item=None):
    checks = "\n".join(f"{i+1}. {c}" for i, c in enumerate(rubric))
    txt, _, _ = chat("judge", [
        {"role": "system", "content": RUBRIC_SYS},
        {"role": "user", "content": RUBRIC_TMPL.format(
            intent=intent, ans=answer, checks=checks)}], item=item, tag="judge_rubric")
    m = re.search(r"\[[^\]]*\]", txt or "")
    if not m:
        return None
    try:
        arr = json.loads(m.group())
        arr = [1 if int(x) else 0 for x in arr][:len(rubric)]
        while len(arr) < len(rubric):
            arr.append(0)
        return arr
    except Exception:
        return None
