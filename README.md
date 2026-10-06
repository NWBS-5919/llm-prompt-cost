# llm-prompt-cost

**사람들이 AI에 엉성하게 물어서 치르는 대가를 재고, 질문의 결함을 보고 언제 다듬고 어느 모델을 쓸지 정한다.**

경희대 산업경영공학과 학부 2인 · 대한산업공학회 2026 추계학술대회 포스터
(초록 2026-10-20 · 발표자료 2026-11-10 · 발표 2026-12-03~04 부산) · GPU 불필요

---

## 현재 상태 (2026-10-06)

**본실험 완료 · 채점 보정 · 오타 일반성 점검 · 정보 가치 분해까지 끝.** 남은 일: 사람 질문 판정(두 사람) → 그림 3장 → 초록(10/18 목표).
전체 결과 [`docs/RESULTS.md`](docs/RESULTS.md) · 진행 기록과 봉착했던 문제 [`docs/PROGRESS.md`](docs/PROGRESS.md) ·
결정 근거 [`docs/DECISIONS.md`](docs/DECISIONS.md) (D-1~D-39, 남은 분석은 D-38 에 결과 보기 전에 고정).

---

## 결론

> **엉성한 질문의 대가는 작고 국소적이며, 그것을 고치는 비용이 대부분 더 크다.
> 결함 정보에는 가치가 있지만, 그 가치를 실현하려면 결함 유형별 성능을 정확히 알아야 하고 그게 병목이다.**

**1. 표면 결함 중 심한 오타만 대가가 있다.** 한글 글자 20% 에 자모 오타를 넣으면 **6개 모델 중 5개**(네 회사)에서 정답률이 유의하게
떨어진다(−8~−19%p, p ≤ 0.022). 기준선이 낮은 모델일수록 크게 떨어지고, 가장 강한 모델(qwen3.8-max)만 버틴다(−6%p, p=0.11).
가벼운 오타(5·10%), 웹 잡문, 지시 생략, 맥락 생략, 그럴듯한 무관 정보는 유의하지 않고, 엉성해도 답이 길어지지 않는다.

| 모델 | 기준선 | 오타 20% |
|---|---|---|
| solar-mini4 · gemma-3-27b | 91% · 86% | **72% · 72%** |
| qwen3.8-flash · gemma-4-31B · deepseek-v4-flash | 93 · 97 · 97% | **81 · 88 · 89%** |
| qwen3.8-max | 97% | 91% |

**2. 고치는 비용이 더 크다.** 질문 재작성은 모든 결함에서 비용을 +60~90% 늘리고(유의) 정답률 이득은 어디서도 유의하지 않다.
재작성이 질문 뜻을 바꾸는 경우도 있다(단위 누락).

**3. 결함 정보의 가치 — 상한은 있지만 실현이 어렵다.** 예산이 빠듯할 때(전부 고가 대비 20%) 결함을 아는 정책의 이론상 가치는
+2~3 가중정답/일(달성률 +3~4%p)이지만, 보정 문항 49개로 추정해 결정하면 −1~+1 로 사라진다. 보정 문항이 늘수록 커지지만 49개에서도 상한의 절반.

> 포스터 메시지(안): 엉성한 한국어 질문, 고칠 가치가 있는가 — 심한 오타만 비싸고, 다듬는 비용은 그 이득보다 크다.
> 결함을 보고 모델을 고르는 정책의 가치는 보정 자료의 양에 달려 있다.

한계(현실 오타율 미측정, 수학 한 종류, 연구자 2명이 쓴 사람 질문, 파일럿 후 등급 교체 등)는 [`docs/RESULTS.md`](docs/RESULTS.md).

---

## 실험 설계

```
GSM-Symbolic P1 영어 원문 ──번역(claude-sonnet-5-5)──▶ 한국어 기준선
                                                        │
          ┌───────────── 변형 9개 ──────────────────────┤
          │  기준선 · 오타 5/10/20% · 웹 잡문 약/강        │  규칙 (고정 시드)
          │  지시 제거                                    │
          │  맥락 누락(D1) · 그럴듯한 무관 정보(N1)         │  고가 모델 생성 → 조작 확인 → 최대 3회
          │  + 사람이 직접 쓴 질문 A·B (30문항)            │
          ▼
  행동 6개 = {그대로, 재작성(qwen3.8-flash)} × {gemma-3-27b, qwen3.8-flash, qwen3.8-max}
          ▼
  runs.jsonl (응답 원본·토큰·정오) → analyze.py (Part A·B) → partc.py (Part C: 사후 최적 DP · 입찰가격 정책)
```

- 모든 호출 `temperature=0`, 사고(thinking) 모드 끔, 1턴.
- LLM 축은 ① 기준선과 다름 ② 원래 숫자 보존(N1은 새 숫자 추가) ③ 심판이 같은 정답이라 판정 — 셋을 통과한 것만 분석.
- 확증 분석은 저가 모델·정답률·축 7개를 McNemar + Holm 보정으로. 그 밖은 탐색적 분석(신뢰구간).
- Part C는 추가 API 호출 없이 실측표를 입력으로 쓴다. 사후 최적을 넘는 정책 결과가 나오면 자동으로 멈춘다.

---

## 실행

```bash
pip3 install numpy scipy certifi
cp env.example.txt env.txt                  # 키·모델·단가를 채운다 (env.txt 는 커밋 금지)
# data/gsm_symbolic_p1.jsonl 을 HuggingFace apple/GSM-Symbolic 의 p1/test.jsonl 에서 받는다
python3 prepare.py                          # 100문항 → items.json, human_TODO.csv

python3 run_main.py --fake && python3 analyze.py runs_fake.jsonl   # 무과금 배관 점검
python3 run_main.py --limit 30 && python3 analyze.py               # 파일럿 → G1 판정
python3 run_main.py && python3 analyze.py && python3 partc.py      # 본실험 (같은 파일에 이어서, 빠진 칸만 채움)
```

더블클릭용: `점검.command` → `연결테스트.command` → `파일럿.command` → `본실험.command`.
속도: `ITEM_WORKERS=6 MIN_INTERVAL=0.3 python3 run_main.py` (본실험은 99문항 43분).

## 코드

| 파일 | 역할 |
|---|---|
| `llm.py` | OpenAI 호환 호출(스트리밍) · 키 2개 자동 전환 · 재시도 · 동시 호출 잠금 · `--fake` 가짜 모델 |
| `perturb.py` | 결함 5축 · 변형 9개 생성 (규칙 + LLM) |
| `run_main.py` | 문항 동시 실행 · LLM 축 조작 확인 · 빠진 칸만 보충 · 키별 예산 자동 정지 |
| `grade.py` | 정답 채점 |
| `common.py` | 기록 로드 · 제외 문항 · 비용 환산 · 키별 사용량 |
| `analyze.py` | 조작 확인 · G1 판정 · 실측표 · H1 확증 · H2 재작성 · 사람 질문 → `table1.csv`, `cells.csv` |
| `partc.py` | 사후 최적 DP · LP 재풀이 입찰가격 정책 · 결함 정보의 가치 → `partc_sim.json` |
| `prepare.py` | 문항 100개 · 학습/시험 분할 |
| `regrade.py` | 채점 보정 — '정답:' 형식 없는 답의 최종 숫자를 LLM 으로 추출 (D-37) |
| `typo_robustness.py` | 오타 일반성 — 등급 밖 모델 3개에 기준선·오타 3단계 (D-38 ②) → `runs_typo.jsonl` |
| `voi_map.py` · `voi_curve.py` | 결정 지도(심한 오타 비율 × 예산) · 보정 문항 수별 정보 가치 (D-38 ④, D-39) |
| `figures.py` | 포스터 그림 3장 → `results/figures/` (API 호출 없음) |
| `redo_d1.py` · `migrate_tiers.py` | 진행 중 설계 변경 때 쓴 일회성 이전 스크립트 (D-29, D-33) |
| `사고모드테스트.py` | 사고 모드 끄는 옵션 시험 (D-27) |

의존성은 표준 라이브러리 + `numpy`, `scipy` (+ macOS 인증서용 `certifi`).

## 결과 파일

| 파일 | 내용 |
|---|---|
| `results/figures/` | 포스터 그림 3장 (`figures.py`): 오타 덤벨 · 재작성 비용-효과 평면 · 정보 가치 상한 vs 실현 |
| `results/본실험_결과.txt` | analyze.py 전체 출력 (99문항) |
| `results/partc_결과.txt` · `results/partc_sim.json` | Part C 시뮬레이션 |
| `results/오타일반성_결과.txt` | 6개 모델 오타 용량-반응 |
| `results/voi/` | 결정 지도 (`voi_map.json`, 비율별 `partc_sim_severe*.json`) · 보정 문항 수 곡선 (`voi_curve.json`) |
| `results/파일럿_결과.txt` | 등급 재구성 후 파일럿 G1 판정 (29문항, 채점 보정 전) |
| `results/table1.csv` · `results/cells.csv` | 변형 × 행동 실측표 · 문항 단위 칸 (질문 문장은 없음) |

## 저장소에 올리지 않는 것

- **`runs.jsonl`, `items.json`, `human_TODO.csv`, `data/`** — GSM-Symbolic 은 CC BY-NC-ND 4.0 이라
  한국어 번역·변형본을 외부에 배포하지 않는다(D-15). 원 기록은 팀 내부에 보관한다.
- **`env.txt`** — API 키가 들어 있다.

## 반드시 지킬 것

- `runs.jsonl` 을 지우지 않는다. 다시 돌려야 하면 파일명을 바꿔 보관한다.
- 모든 수치는 저장된 실행 기록에서만 낸다. 외부 논문의 수치를 옮겨오지 않는다.
- 실험 코드를 고치면 이유를 `docs/DECISIONS.md` 에 한 줄 남긴다.
