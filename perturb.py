"""
엉성함의 축 — 잘 쓴 질문을 한 가지 방식으로만 망가뜨린다.

  가족 1 · 표면 잡음   정보는 다 있고 글자만 망가짐
      S1 오타·자모분리   (규칙)
      S2 어순 도치       (규칙)
      S3 구어체·줄임말   (LLM)
  가족 2 · 정보 결핍   글자는 멀쩡한데 정보가 없음
      D1 맥락 생략       (LLM)
      D2 지시 부재       (규칙)
      D3 지시어 모호     (LLM, 개방형 전용)
  가족 3 · 정보 과잉   쓸데없는 게 붙음
      E1 사족·잡담       (LLM)
      E2 무관 붙여넣기   (규칙)

규칙 기반은 시드를 고정하면 누구나 똑같이 재현된다. 논문에서 이게 강점이다.
"""
import random, re
from llm import chat

SEED = 20260922

# 앵커는 D3(지시어 모호)를 쓰지 않는다 — 수학 문제에 "그거"가 어색하다
ANCHOR_AXES = ["S1", "S2", "S3", "D1", "D2", "E1", "E2"]
OPEN_AXES = ["S1", "S2", "S3", "D1", "D2", "D3", "E1", "E2"]

FAMILY = {"S1": "표면", "S2": "표면", "S3": "표면",
          "D1": "결핍", "D2": "결핍", "D3": "결핍",
          "E1": "과잉", "E2": "과잉"}

LABEL = {"S1": "오타·자모분리", "S2": "어순 도치", "S3": "구어체·줄임말",
         "D1": "맥락 생략", "D2": "지시 부재", "D3": "지시어 모호",
         "E1": "사족·잡담", "E2": "무관 붙여넣기"}

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


# ────────────────────────────────────────────────── S1 오타·자모분리
def s1_typo(text, rate=0.13, seed=SEED):
    """받침 누락 · 모음 교체 · 자모 분리 · 띄어쓰기 붕괴"""
    rng = random.Random(seed + len(text))
    out = []
    for ch in text:
        if not _is_hangul(ch) or rng.random() > rate:
            out.append(ch)
            continue
        cho, jung, jong = _decompose(ch)
        r = rng.random()
        if r < 0.35 and jong != " ":
            out.append(_compose(cho, jung))                      # 받침 누락
        elif r < 0.65 and jung in VOWEL_SWAP:
            out.append(_compose(cho, VOWEL_SWAP[jung], jong))     # 모음 교체
        elif r < 0.85:
            out.append(_compose(cho, jung))                       # 자모 분리
            out.append(jong.strip())
        else:
            out.append(ch)
    s = "".join(out)
    # 띄어쓰기 붕괴 — 공백 일부 제거
    parts = s.split(" ")
    s = "".join(p + ("" if rng.random() < 0.35 else " ") for p in parts).strip()
    return s


# ────────────────────────────────────────────────── S2 어순 도치
def s2_scramble(text, seed=SEED):
    """절 단위로 쪼개 순서를 섞는다. 한국어는 어순이 자유로워 의미는 대체로 살아남는다."""
    rng = random.Random(seed + 7)
    sents = [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]
    out = []
    for s in sents:
        chunks = [c.strip() for c in re.split(r",\s*|(?<=고)\s+|(?<=면)\s+|(?<=데)\s+", s) if c.strip()]
        if len(chunks) < 2:
            out.append(s)
            continue
        rng.shuffle(chunks)
        out.append(", ".join(chunks))
    rng.shuffle(out)
    return " ".join(out)


# ────────────────────────────────────────────────── D2 지시 부재
INSTRUCTIONS = [
    "\n\n단계별로 풀고, 최종 답만 마지막 줄에 '정답: ' 형식으로 써라.",
    "\n\nLet's think step by step.",
]


def d2_no_instruction(text):
    for ins in INSTRUCTIONS:
        text = text.replace(ins, "")
    # 일반적인 형식 요구 문장도 제거
    text = re.sub(r"[^.?!\n]*?(형식|분량|글자|줄로|단계별|요약해|정리해)[^.?!\n]*[.?!]\s*$",
                  "", text).strip()
    return text


# ────────────────────────────────────────────────── E2 무관 붙여넣기
NOISE_BLOCKS = [
    "홈 | 로그인 | 회원가입 | 고객센터 | 이용약관 | 개인정보처리방침\n"
    "배너 닫기 오늘 하루 보지 않기 쿠키 설정 전체 동의 필수만 동의",
    "[이전 대화]\nA: 아까 말한 그 카페 이름이 뭐였지\nB: 아 거기 이름 기억 안 나네\n"
    "A: 담에 가면 찍어놔야겠다\nB: ㅇㅇ 그러자",
    "Copyright 2026 All rights reserved. 본 문서의 무단 전재 및 재배포를 금합니다.\n"
    "관련 기사 더보기 · 인기 검색어 1위 2위 3위 · 댓글 0",
]


def e2_paste_noise(text, seed=SEED):
    rng = random.Random(seed + 11)
    block = rng.choice(NOISE_BLOCKS)
    return (block + "\n\n" + text) if rng.random() < 0.5 else (text + "\n\n" + block)


# ────────────────────────────────────────────────── LLM 기반 축
LLM_SPEC = {
    "S3": ("질문을 20대가 채팅으로 치듯 구어체·줄임말로 바꿔라. "
           "정보는 하나도 빼지 말고 말투만 바꾼다. 'ㅇㅇ', '~임', '~음' 같은 표현을 쓴다."),
    "D1": ("질문에서 맥락 정보 한두 개를 슬쩍 빼라. 단위, 대상, 상황, 조건 중에서 고른다. "
           "단 문제 자체가 풀 수 없게 되면 안 된다. 추론으로 메울 수 있는 정도만 뺀다."),
    "D3": ("질문의 대상을 '그거', '이거', '저번에 그거' 같은 모호한 지시어로 바꿔라. "
           "무엇을 말하는지 불분명해지게 만든다."),
    "E1": ("질문 앞뒤에 관련 없는 잡담과 사족을 붙여라. 인사, 근황, 미안함, 감사 같은 것들. "
           "원래 질문 내용은 그대로 두고 분량만 두세 배로 늘린다."),
}

LLM_SYS = ("너는 실험용으로 질문을 일부러 망가뜨리는 도구다. "
           "요청된 한 가지 방식으로만 바꾸고, 다른 것은 건드리지 않는다. "
           "질문에 답하지 말고, 바꾼 질문만 출력한다.")


def llm_axis(axis, text, item=None):
    out, _, _ = chat("rewriter", [
        {"role": "system", "content": LLM_SYS},
        {"role": "user", "content": LLM_SPEC[axis] + "\n\n---\n" + text}],
        item=item, tag=f"axis_{axis}")
    return out.strip()


# ────────────────────────────────────────────────── 진입점
def apply_axis(axis, text, item=None):
    if axis == "S1":
        return s1_typo(text)
    if axis == "S2":
        return s2_scramble(text)
    if axis == "D2":
        return d2_no_instruction(text)
    if axis == "E2":
        return e2_paste_noise(text)
    return llm_axis(axis, text, item)


if __name__ == "__main__":
    base = ("친구 4명이 피자 4판을 총 64달러에 시켰다. 그중 두 판이 30달러였다면, "
            "나머지 두 판은 각각 얼마인가? 두 판의 가격은 같다."
            "\n\n단계별로 풀고, 최종 답만 마지막 줄에 '정답: ' 형식으로 써라.")
    print("[기준]\n" + base + "\n")
    for a in ("S1", "S2", "D2", "E2"):
        print(f"[{a} {LABEL[a]}]\n{apply_axis(a, base)}\n")
    print("※ S3/D1/D3/E1 은 LLM 호출이 필요해 여기서는 생략")
