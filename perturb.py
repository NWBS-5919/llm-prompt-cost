"""
v5 · 결함 5축 — 기준선을 한 가지 방식으로만 망가뜨린다.

  S1 오타·띄어쓰기 붕괴     정보 보존형   규칙   강도 lo(한글 5%) / md(10%) / hi(20%)  — 용량-반응 (D-34)
  E2 무관 붙여넣기          정보 보존형   규칙   강도 lo(블록 1개) / hi(블록 3개)
  D2 지시 제거              지시 결핍     규칙
  D1 맥락 누락              정보 손실형   LLM   + 조작 확인(run_main.make_llm_variant)
  N1 그럴듯한 무관 정보     정보 과잉형   LLM   + 조작 확인 — GSM-NoOp 방식 (D-35)

규칙 3축은 시드가 고정이라 누구나 똑같이 재현된다.
"""
import random, re
from llm import chat

SEED = 20260922

VARIANTS = ["base", "S1_lo", "S1_md", "S1_hi", "E2_lo", "E2_hi", "D2", "D1", "N1"]
AXIS = {"base": "기준", "S1_lo": "S1", "S1_md": "S1", "S1_hi": "S1", "E2_lo": "E2", "E2_hi": "E2",
        "D2": "D2", "D1": "D1", "N1": "N1"}
KIND = {"S1": "보존형", "E2": "보존형", "D2": "지시 결핍", "D1": "손실형", "N1": "과잉형"}
LABEL = {"base": "기준선", "S1_lo": "오타 5%", "S1_md": "오타 10%", "S1_hi": "오타 20%", "E2_lo": "잡음 약",
         "E2_hi": "잡음 강", "D2": "지시 제거", "D1": "맥락 누락", "N1": "그럴듯한 무관정보"}
S1_RATE = {"lo": 0.05, "md": 0.10, "hi": 0.20}
LLM_AXES = ("D1", "N1")          # LLM 이 만들고 조작 확인을 거치는 축 — 통과한 문항만 분석에 쓴다
E2_BLOCKS = {"lo": 1, "hi": 3}

INSTR = "\n\n단계별로 풀고, 최종 답만 마지막 줄에 '정답: ' 형식으로 써라."

# ────────────────────────────────────────────────── 숫자 검사
NUM = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text):
    return {n.replace(",", "") for n in NUM.findall(text or "")}


def numbers_kept(src, dst):
    """src 의 숫자가 dst 에 모두 남아 있는가"""
    return numbers(src) <= numbers(dst)


# ────────────────────────────────────────────────── 한글 자모
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
VOWEL_SWAP = {"ㅏ": "ㅓ", "ㅓ": "ㅏ", "ㅗ": "ㅜ", "ㅜ": "ㅗ",
              "ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅕ": "ㅑ", "ㅑ": "ㅕ"}


def _is_hangul(ch):
    return 0xAC00 <= ord(ch) <= 0xD7A3


def _decompose(ch):
    c = ord(ch) - 0xAC00
    return CHO[c // 588], JUNG[(c % 588) // 28], JONG[c % 28]


def _compose(cho, jung, jong=" "):
    return chr(0xAC00 + (CHO.index(cho) * 21 + JUNG.index(jung)) * 28 + JONG.index(jong))


def s1_typo(text, rate, seed=SEED):
    """받침 누락 · 모음 교체 · 자모 분리 · 띄어쓰기 붕괴. 숫자는 건드리지 않는다."""
    rng = random.Random(seed + len(text) + int(rate * 1000))
    out = []
    for ch in text:
        if not _is_hangul(ch) or rng.random() > rate:
            out.append(ch)
            continue
        cho, jung, jong = _decompose(ch)
        r = rng.random()
        if r < 0.35 and jong != " ":
            out.append(_compose(cho, jung))
        elif r < 0.65 and jung in VOWEL_SWAP:
            out.append(_compose(cho, VOWEL_SWAP[jung], jong))
        elif r < 0.85:
            out.append(_compose(cho, jung))
            out.append(jong.strip())
        else:
            out.append(ch)
    s = "".join(out)
    space_drop = 0.15 + rate                       # 강할수록 띄어쓰기도 더 무너진다
    parts = s.split(" ")
    return "".join(p + ("" if rng.random() < space_drop else " ") for p in parts).strip()


NOISE_BLOCKS = [
    "홈 | 로그인 | 회원가입 | 고객센터 | 이용약관 | 개인정보처리방침\n"
    "배너 닫기 오늘 하루 보지 않기 쿠키 설정 전체 동의 필수만 동의",
    "[이전 대화]\nA: 아까 말한 그 카페 이름이 뭐였지\nB: 아 거기 이름 기억 안 나네\n"
    "A: 담에 가면 찍어놔야겠다\nB: ㅇㅇ 그러자",
    "Copyright All rights reserved. 본 문서의 무단 전재 및 재배포를 금합니다.\n"
    "관련 기사 더보기 · 인기 검색어 · 댓글 보기",
    "공유하기 카카오톡 페이스북 링크복사\n이 글이 도움이 되었나요? 좋아요 싫어요",
]


def e2_paste_noise(text, n_blocks, seed=SEED):
    """잡음 블록을 앞뒤에 붙인다. 블록에는 숫자가 없어 채점을 방해하지 않는다."""
    rng = random.Random(seed + 11 + len(text))
    blocks = rng.sample(NOISE_BLOCKS, n_blocks)
    head, tail = [], []
    for b in blocks:
        (head if rng.random() < 0.5 else tail).append(b)
    return "\n\n".join(head + [text] + tail)


def d2_no_instruction(text):
    return text.replace(INSTR, "").strip()


D1_SYS = ("너는 실험용으로 질문을 일부러 망가뜨리는 도구다. "
          "요청된 한 가지 방식으로만 바꾸고, 다른 것은 건드리지 않는다. "
          "질문에 답하지 말고, 바꾼 질문만 출력한다.")
D1_SPEC = ("질문에서 맥락 정보 한두 개를 슬쩍 빼라. 단위, 대상, 상황 설명 중에서 고른다. "
           "숫자는 하나도 빼지 말고, 문제를 풀 수 없게 만들지 마라. 마지막 줄의 풀이 형식 지시는 그대로 둔다. "
           "숫자가 들어 있는 문장이나 구절은 지우지 말고, 계산에 쓰이지 않는 말만 뺀다. "
           "반드시 한 군데 이상은 빼야 한다.")
# GSM-NoOp(Mirzadeh 외, 2024): 관련 있어 보이지만 계산에 쓰이지 않는 문장 하나가 강한 모델도 크게 떨어뜨린다.
# 사람이 자기 사정을 덧붙이며 묻는 습관을 흉내 낸다. 원래 숫자와 문장은 그대로 두고 새 숫자를 하나 넣는다.
N1_SPEC = ("질문 중간에 문장 하나를 끼워 넣어라. 그 문장은 문제 상황과 관련 있어 보이고 숫자를 하나 담고 있지만, "
           "최종 정답 계산에는 전혀 영향을 주지 않아야 한다(예: 물건 일부가 조금 작았다, 누군가 그 가게를 좋아한다). "
           "원래 문장과 숫자는 한 글자도 바꾸거나 지우지 마라. 마지막 줄의 풀이 형식 지시는 그대로 둔다.")
SPECS = {"D1": D1_SPEC, "N1": N1_SPEC}
# 저가(Flash)로 만들었더니 파일럿 9문항 중 3번은 아무것도 안 뺐고 3번은 숫자 조건을 지웠다 → 고가 모델로 만든다 (D-29)
AXIS_ROLE = "expensive"
D1_ROLE = AXIS_ROLE              # 예전 이름 (기록의 d1_tier)
LLM_TRIES = 3


def llm_generate(v, base, item=None, feedback=""):
    """LLM 축 한 번 생성. feedback 은 직전 시도가 왜 탈락했는지 — temperature 0 이라 그대로 다시 부르면 같은 답이 나온다"""
    spec = SPECS[v] + (f"\n\n(주의: {feedback})" if feedback else "")
    out, a, b, _ = chat(AXIS_ROLE, [{"role": "system", "content": D1_SYS},
                                    {"role": "user", "content": spec + "\n\n---\n" + base}],
                        item=item, tag=f"axis_{v}")
    return out.strip(), a, b


def added_numbers(src, dst):
    """dst 에만 있는 숫자 — N1 은 하나 이상이어야 한다"""
    return numbers(dst) - numbers(src)


def apply_variant(v, base, item=None):
    """returns (text, tokens_in, tokens_out) — 규칙 축은 토큰 0"""
    if v == "base":
        return base, 0, 0
    if v.startswith("S1"):
        return s1_typo(base, S1_RATE[v[-2:]]), 0, 0
    if v.startswith("E2"):
        return e2_paste_noise(base, E2_BLOCKS[v[-2:]]), 0, 0
    if v == "D2":
        return d2_no_instruction(base), 0, 0
    if v in LLM_AXES:
        return llm_generate(v, base, item)  # 검증·재시도는 run_main.make_llm_variant
    raise ValueError(v)


if __name__ == "__main__":
    base = ("친구 4명이 피자 4판을 총 64달러에 시켰다. 그중 두 판이 30달러였다면, "
            "나머지 두 판은 각각 얼마인가? 두 판의 가격은 같다." + INSTR)
    for v in VARIANTS:
        if v in LLM_AXES:
            continue
        t = apply_variant(v, base)[0]
        print(f"[{v} {LABEL[v]}]  숫자 보존: {numbers_kept(base, t)}\n{t}\n")
    print("※ D1·N1 은 LLM 호출이 필요해 여기서는 생략")
