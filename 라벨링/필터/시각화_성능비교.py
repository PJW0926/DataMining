import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# 0. 저장 경로 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parent

# ==============================
# 1. 한글 폰트 설정
# ==============================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ==============================
# 2. 성능 지표 입력
# ==============================
# 여기에 네 실제 4차, 5차 성능값을 넣으면 됨
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "4차 필터": [0.75, 0.72, 0.80, 0.76],
    "5차 필터": [0.81, 0.79, 0.83, 0.81]
})

print(metrics_df)

# ==============================
# 3. 그래프 생성
# ==============================
plot_df = metrics_df.set_index("Metric")

ax = plot_df.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.75
)

plt.title("그림 3-2. 4차 및 5차 진정성 필터의 주요 성능 지표 비교", fontsize=15, pad=15)
plt.xlabel("성능 지표")
plt.ylabel("점수")
plt.ylim(0, 1.0)
plt.xticks(rotation=0)
plt.legend(title="필터 버전")
plt.grid(axis="y", linestyle="--", alpha=0.4)

# 막대 위 숫자 표시
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=10)

plt.tight_layout()

# ==============================
# 4. 저장
# ==============================
output_path = BASE_DIR / "그림_3-2_4차_5차_성능지표비교.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("저장 완료:", output_path)

plt.show()