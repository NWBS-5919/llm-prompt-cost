# llm-prompt-cost

경희대 산업경영공학과 학부 포스터 논문 + 창업 프로토타입 프로젝트.
**대한산업공학회 2026 추계 포스터** · 초록 10/20 · 발표자료 11/10 · 발표 12/3~4 (부산) · 팀 2인.
범위를 늘리는 제안은 하지 말 것. 설계 기준은 `docs/DECISIONS.md` D-14 이후 (특히 D-27~D-36).

---

## 한 줄

일반 사용자가 AI에 **엉성하게 묻기 때문에 치르는 대가**를 측정하고,
질문의 결함을 보고 **언제 다듬고 어느 모델을 쓸지**를 사용 한도 안에서 최적화한다.

작업 전에 읽을 것: `README.md`(현황·결과 요약) → `docs/RESULTS.md`(결과·한계·남은 일) →
`docs/PROGRESS.md`(봉착했던 문제와 해결) → `docs/DECISIONS.md`(결정 근거).

---

## 지금 단계

**본실험·채점 보정·오타 일반성·정보 가치 분해 완료 (2026-10-06).** 결과는 `results/`, 정리는 `docs/RESULTS.md`.
남은 일: 사람 질문 판정(두 사람) → 그림 3장 → 초록. 남은 분석의 판단 규칙은 D-38 에 고정돼 있다 — 바꾸지 말 것.

```bash
python3 analyze.py runs.jsonl      # Part A·B → table1.csv, cells.csv (확정본은 results/ 로 복사)
python3 partc.py runs.jsonl        # Part C → partc_sim.json
python3 run_main.py                # 빠진 칸이 있으면 그것만 채운다 (새 변형·등급을 추가했을 때)
```

`runs.jsonl`, `items.json`, `human_TODO.csv`, `data/` 는 저장소에 없다 (라이선스, D-15). 팀 내부 보관본을 받아서 쓴다.

---

## 환경

설정은 `env.txt` (코드가 자동으로 읽음, 커밋 금지). `env.example.txt` 를 복사해 키를 채운다.
게이트웨이는 ChatKHU(BAZE) OpenAI 호환 `/v1/gateway`, 키는 계정별 월 한도 3,937 크레딧.

| 변수 | 지금 값 |
|---|---|
| `MODEL_CHEAP` / `MID` / `EXPENSIVE` | `google/gemma-3-27b-it` / `qwen3.8-flash` / `qwen3.8-max` |
| `MODEL_REWRITER` / `MODEL_JUDGE` | `qwen3.8-flash` / `claude-sonnet-5-5` |
| `PRICE_*` | 크레딧 / 100만 토큰. `PRICE_CHEAP` 은 임시값(Gemma 4 단가) |
| `LLM_API_KEY_2` | 1번 키 예산을 다 쓰면 자동 전환 |
| `BUDGET` | 키당 최대 크레딧. 넘기 전에 멈춤 |
| `EXTRA_BODY` | 사고 모드 끔 (D-27) |
| `MAX_TOKENS` | 답변 길이 상한 8192 (D-36) |
| `WORKERS` / `ITEM_WORKERS` / `MIN_INTERVAL` | 문항 안 동시 호출 8 / 동시 문항 3 / 호출 간격 0.6초. 본실험은 6 · 0.3 으로 돌렸다 |

---

## 코드 지도

| 파일 | 역할 |
|---|---|
| `llm.py` | OpenAI 호환 호출(스트리밍) · 키 전환 · 재시도 · 잠금 · `--fake` 가짜 모델 |
| `perturb.py` | 결함 5축, 변형 9개 (S1 오타 5/10/20%, E2 잡문 약/강, D2 지시 제거, D1 맥락 누락, N1 무관 정보) |
| `run_main.py` | 문항 동시 실행 · LLM 축 조작 확인(최대 3회) · 빠진 칸만 보충 · 키별 예산 → `runs.jsonl` |
| `grade.py` | 정답 채점 |
| `common.py` | 기록 로드 · 제외 문항(`data/excluded_items.json`) · 비용 환산 · 키별 사용량(`spend`) |
| `analyze.py` | 조작 확인 · G1 · 실측표 · H1 확증(McNemar+Holm) · H2 재작성 · 사람 질문 |
| `partc.py` | 사후 최적 DP · LP 재풀이 입찰가격 · 결함 정보의 가치 |
| `prepare.py` | GSM-Symbolic P1 100문항 · 학습/시험 분할 |
| `regrade.py` | 채점 보정 — 형식 없는 답만 LLM 추출 (D-37). 새 칸이 생기면 다시 돌린다 |
| `typo_robustness.py` | 오타 일반성 (등급 밖 모델 3개) → `runs_typo.jsonl` (D-38 ②) |
| `voi_map.py` · `voi_curve.py` | 결정 지도 · 보정 문항 수별 정보 가치 (D-38 ④, D-39) |
| `redo_d1.py` · `migrate_tiers.py` | 일회성 기록 이전 (D-29, D-33). 다시 돌릴 일 없음 |
| `judge.py` · `openended_items.py` | v4 개방형(요약) 과제용. v5 에서는 쓰지 않음 |

---

## 반드시 지킬 규칙

**결과 파일을 지우지 않는다.** `runs.jsonl` 에 모든 응답 원본·토큰·판정이 있다. 설계를 바꿔 예전 칸을 대체할 때도
지우지 않고 문항 기록의 `superseded` 로 옮긴다(비용 집계에 계속 포함).

**모든 수치는 저장된 실행 자료에서만 산출한다.** 외부 논문의 성능 수치를 옮겨오지 않는다.

**기록과 설정의 모델이 다르면 섞지 않는다.** `run_main.py` 가 멈춘다. 등급을 바꾸면 이전 스크립트로 기록을 먼저 옮긴다.

**LLM 축은 조작 확인을 통과한 것만 분석한다.** (`{v}_valid`)

**문제 결함 문항은 결과가 아니라 문제 문장을 근거로 제외한다.** `data/excluded_items.json` 에 이유와 함께.

**달성률이 100%를 넘으면 중단한다.** `partc.py` 가 자동으로 멈춘다(D-03, D-19).

**LLM 호출에 temperature=0, 사고 모드 끔.** 재현성과 등급 간 공정성 때문이다.

**GSM-Symbolic 번역·변형본을 외부에 배포하지 않는다.** (CC BY-NC-ND 4.0)

**API 키를 커밋·공유하지 않는다.** 예전에 공유 링크와 압축 파일로 노출된 적이 있다.

---

## 자주 터지는 곳

| 증상 | 확인할 것 |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` | `pip3 install certifi` |
| HTTP 403 `1010` | User-Agent (llm.py 가 처리) |
| HTTP 524 | 120초 넘는 응답 — 스트리밍으로 처리됨. 계속되면 `MAX_TOKENS` 확인 |
| 한 문항이 끝나지 않음 | 답변 폭주. `MAX_TOKENS` 가 걸려 있는지 |
| 출력 토큰이 비정상적으로 많음 | `usage.completion_tokens_details.reasoning_tokens` — 사고 모드가 켜졌는지 |
| `[중단] … 등급 모델이 env.txt 와 다릅니다` | 기록과 설정의 모델이 다름. 이전 스크립트 먼저 |
| HTTP 429 `120 requests per minute` | 키당 분당 120회. 같은 키로 두 작업을 동시에 돌리지 말 것, `MIN_INTERVAL` 0.6 이상 |
| 정답을 맞혔는데 오답 처리 | '정답:' 형식이 없는 답 — `regrade.py` 로 보정 |
| 기록상 사용량 < ChatKHU 사용량 | 실패·시험 호출은 기록에 안 남는다(약 8%). `BUDGET` 을 한도보다 넉넉히 낮게 |

---

## 작업 스타일

- 한국어로 답한다.
- 수치를 지어내지 않는다. 모르면 "아직 측정 안 됨"이라고 쓴다.
- 실험 코드를 고칠 때는 **왜 고쳤는지를 `docs/DECISIONS.md` 에 한 줄 남긴다.**
- 새 의존성을 함부로 추가하지 않는다. 표준 라이브러리 + `numpy`, `scipy` (+ `certifi`).
