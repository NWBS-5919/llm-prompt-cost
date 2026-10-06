"""
포스터 그림 3장 → results/figures/  (API 를 부르지 않는다. 수치는 모두 저장된 기록에서 계산)

  fig1_typo.png     오타 20% — 6개 모델의 기준선 → 오타 20% 정답률 (덤벨)
  fig2_rewrite.png  재작성의 비용-효과 평면 — 저가 모델, 변형 9개 (재작성 − 그대로)
  fig3_voi.png      결함 정보의 가치 — 보정 문항 수별 실현 가치 vs 상한 (예산 20%)

실행: python3 figures.py
"""
import os as _os, pathlib as _pl
HERE = _pl.Path(__file__).resolve().parent
_os.chdir(HERE)
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
import common, perturb as P

OUT = HERE / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# 색 — dataviz 기준 팔레트(밝은 면). 파랑 두 단계는 검증 통과(ordinal), 나머지는 글자·축 잉크
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_L, BLUE_D, ACCENT, BLUE_PALE = "#6da7ec", "#184f95", "#2a78d6", "#b7d3f6"
plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo", "axes.unicode_minus": False, "font.size": 10.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
})


def title(fig, main, sub):
    fig.text(0.02, 0.97, main, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, sub, fontsize=10, color=INK2, va="top")


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


R = common.load_for_analysis("runs.jsonl")


# ───────────────────────────── 그림 1
def fig1():
    typo = {r["id"]: r for r in common.load_runs("runs_typo.jsonl")}
    models = [("gemma-3-27b (저가)", lambda r, v: r["cond"][f"{v}|raw|cheap"]["correct"], R),
              ("qwen3.8-flash (중가)", lambda r, v: r["cond"][f"{v}|raw|mid"]["correct"], R),
              ("qwen3.8-max (고가)", lambda r, v: r["cond"][f"{v}|raw|expensive"]["correct"], R)]
    for m, label in (("solar-mini4", "solar-mini4"), ("google/gemma-4-31B-it", "gemma-4-31B"),
                     ("deepseek-v4-flash", "deepseek-v4-flash")):
        models.append((label, (lambda mm: lambda r, v: typo[r["id"]]["cond"][f"{v}|raw|{mm}"]["correct"])(m),
                       [r for r in R if r["id"] in typo]))
    rows = []
    for name, f, rs in models:
        b0 = np.mean([f(r, "base") for r in rs]) * 100
        b1 = np.mean([f(r, "S1_hi") for r in rs]) * 100
        lose = sum(f(r, "base") and not f(r, "S1_hi") for r in rs)
        gain = sum(f(r, "S1_hi") and not f(r, "base") for r in rs)
        p = binomtest(lose, lose + gain, 0.5).pvalue
        rows.append((name, b0, b1, p, len(rs)))
    rows.sort(key=lambda x: x[1] - x[2])                     # 하락이 큰 모델이 위로
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.subplots_adjust(left=0.25, right=0.74, top=0.80, bottom=0.13)
    for i, (name, b0, b1, p, n) in enumerate(rows):
        ax.plot([b1, b0], [i, i], color=BASE, linewidth=2.2, solid_capstyle="round", zorder=1)
        ax.scatter([b0], [i], s=70, color=BLUE_L, edgecolors=SURF, linewidths=2, zorder=3)
        ax.scatter([b1], [i], s=70, color=BLUE_D, edgecolors=SURF, linewidths=2, zorder=3)
        sig = "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.05 else f"p={p:.2f} (유의하지 않음)"
        ax.text(101.2, i, f"{b1 - b0:+.0f}%p  {sig}", va="center", fontsize=9.5,
                color=INK if p < 0.05 else MUTED, clip_on=False)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(60, 100)
    ax.set_xticks(range(60, 101, 10), [f"{x}%" for x in range(60, 101, 10)])
    ax.set_xlabel("정답률 (그대로 질문)")
    grid(ax, "x")
    ax.scatter([], [], s=70, color=BLUE_L, label="기준선 (깨끗한 질문)")
    ax.scatter([], [], s=70, color=BLUE_D, label="오타 20% (한글 글자 20%에 자모 오타)")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.0), ncol=2, frameon=False, fontsize=9.5,
              labelcolor=INK2, handletextpad=0.3, columnspacing=1.4)
    title(fig, "심한 오타는 6개 모델 중 5개의 정답률을 유의하게 떨어뜨린다",
          "GSM-Symbolic P1 99문항 · 같은 문항을 기준선과 오타 20%로 짝지어 비교(McNemar) · 4개 회사 모델")
    fig.savefig(OUT / "fig1_typo.png", dpi=300)
    plt.close(fig)
    return rows


# ───────────────────────────── 그림 2
def fig2():
    rng = np.random.default_rng(20261006)
    pts = []
    for v in P.VARIANTS:
        rs = [r for r in R if v in r["prompts"] and (v not in P.LLM_AXES or r["checks"].get(f"{v}_valid"))]
        raw = np.array([common.cell(r, v, "raw", "cheap") for r in rs], float)
        rw = np.array([common.cell(r, v, "rw", "cheap") for r in rs], float)
        dacc = (rw[:, 0] - raw[:, 0]) * 100
        dcost = rw[:, 2] - raw[:, 2]
        base_cost = raw[:, 2].mean()
        bs = rng.integers(0, len(rs), (2000, len(rs)))
        acc_ci = np.percentile(dacc[bs].mean(1), [2.5, 97.5])
        cost_ci = np.percentile(dcost[bs].mean(1) / base_cost * 100, [2.5, 97.5])
        pts.append((v, dcost.mean() / base_cost * 100, cost_ci, dacc.mean(), acc_ci))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.80, bottom=0.12)
    ax.axhline(0, color=BASE, linewidth=1.0, zorder=1)
    focus = {"S1_hi": "오타 20%", "N1": "그럴듯한 무관 정보", "D2": "지시 제거"}
    for v, x, xci, y, yci in sorted(pts, key=lambda t: t[0] in focus):
        c = ACCENT if v in focus else MUTED
        ax.plot([xci[0], xci[1]], [y, y], color=c, linewidth=1.1, alpha=0.55 if v not in focus else 0.9, zorder=2)
        ax.plot([x, x], [yci[0], yci[1]], color=c, linewidth=1.1, alpha=0.55 if v not in focus else 0.9, zorder=2)
        ax.scatter([x], [y], s=60 if v in focus else 38, color=c, edgecolors=SURF, linewidths=2, zorder=3)
        if v in focus:
            tx, ty = {"S1_hi": (8, -12), "N1": (8, 15), "D2": (108, 15)}[v]          # 점 구름의 왼쪽 빈 곳 (데이터 좌표)
            ax.annotate(f"{focus[v]}\n{y:+.1f}%p [{yci[0]:+.1f}, {yci[1]:+.1f}]", (x, y), xytext=(tx, ty),
                        textcoords="data", fontsize=9.5, color=INK, va="center",
                        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=2, shrinkB=5))
    ax.set_xlim(0, 150)
    ax.set_xticks(range(0, 151, 25), [f"+{x}%" if x else "0" for x in range(0, 151, 25)])
    ax.set_xlabel("비용 증가 (재작성 - 그대로, 그대로 질문 비용 대비)")
    ax.set_ylabel("정답률 변화 (%p)")
    ax.set_ylim(-20, 20)
    grid(ax, "both")
    ax.text(148, -18.5, "회색: 나머지 6개 조건 (기준선·오타 5/10%·잡음 약/강·맥락 누락)",
            ha="right", fontsize=8.8, color=MUTED)
    title(fig, "질문을 다듬으면 비용은 늘지만 정답률은 유의하게 오르지 않는다",
          "저가 모델(gemma-3-27b) · 재작성기 qwen3.8-flash · 점 = 평균, 선 = 95% 부트스트랩 구간 (99문항)")
    fig.savefig(OUT / "fig2_rewrite.png", dpi=300)
    plt.close(fig)
    return pts


# ───────────────────────────── 그림 3
def fig3():
    t = json.loads((HERE / "results" / "voi" / "voi_curve.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.76, bottom=0.14, wspace=0.08)
    sizes = [10, 20, 30, 49]
    rng = np.random.default_rng(3)
    for ax, s in zip(axes, (0.02, 0.10)):
        rows = {x["train_n"]: x for x in t if x["severe"] == s and x["budget"] == 0.2}
        up = rows["상한"]["voi_mean"]
        ax.axhline(0, color=BASE, linewidth=1.0, zorder=1)
        ax.axhline(up, color=INK2, linewidth=1.2, zorder=1)
        ax.text(55, up + 0.12, f"상한 {up:+.2f}\n(정답표로 결정)", ha="right", va="bottom", fontsize=9, color=INK2)
        means = []
        for i, n in enumerate(sizes):
            runs = rows[n]["voi_runs"]
            xs = n + rng.uniform(-0.9, 0.9, len(runs))
            if len(runs) > 1:
                ax.scatter(xs, runs, s=22, color=BLUE_PALE, edgecolors=SURF, linewidths=1, zorder=2)
            means.append(np.mean(runs))
        ax.plot(sizes, means, color=ACCENT, linewidth=2, solid_capstyle="round", zorder=3)
        ax.scatter(sizes, means, s=60, color=ACCENT, edgecolors=SURF, linewidths=2, zorder=4)
        ax.annotate(f"{means[-1]:+.2f}", (49, means[-1]), xytext=(8, -4), textcoords="offset points",
                    fontsize=9.5, color=INK)
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.set_xlim(5, 56)
        ax.set_xlabel("보정 문항 수")
        ax.set_title(f"하루 질문 중 심한 오타 {s:.0%}", fontsize=10.5, color=INK, loc="left")
        grid(ax, "y")
    axes[0].set_ylabel("결함 정보의 가치 (가중정답/일)")
    axes[0].set_ylim(-4.5, 4.5)
    axes[0].scatter([], [], s=22, color=BLUE_PALE, label="무작위 보정 표본 (5회)")
    axes[0].plot([], [], color=ACCENT, linewidth=2, marker="o", markersize=7, label="평균 (49 = 전체 학습 문항)")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
    title(fig, "결함 정보의 가치는 분명하지만, 보정 49문항으로는 대부분 실현되지 않는다",
          "예산 = 전부 고가 비용의 20% · 결함 인식 - 결함 무시 입찰가격 정책 · 모의 하루 200일 × 40문항")
    fig.savefig(OUT / "fig3_voi.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    rows = fig1()
    for r in rows:
        print(f"  그림1 {r[0]:22s} {r[1]:.0f}% → {r[2]:.0f}%  p={r[3]:.4f}  n={r[4]}")
    for v, x, xci, y, yci in fig2():
        print(f"  그림2 {P.LABEL[v]:10s} 비용 {x:+.0f}% [{xci[0]:+.0f}, {xci[1]:+.0f}]  정답률 {y:+.1f}%p [{yci[0]:+.1f}, {yci[1]:+.1f}]")
    fig3()
    print(f"저장 → {OUT}")
