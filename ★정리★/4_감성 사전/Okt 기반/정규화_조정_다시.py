import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

input_file = BASE_DIR / "crawling_all_filter_with_sentiment_star_v9_token_only_정규화수정.csv"
df = pd.read_csv(input_file, encoding="utf-8-sig")

df["category_total_score"] = pd.to_numeric(df["category_total_score"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

kakao = df[df["platform"] == "kakao"].copy()
kakao_scores = kakao["category_total_score"].dropna()

# 비교할 정규화 기준
quantile_settings = [
    (0.01, 0.85),
    (0.01, 0.90),
    (0.01, 0.95),
    (0.02, 0.85),
    (0.02, 0.90),
    (0.03, 0.85),
    (0.03, 0.90),
    (0.05, 0.90),
    (0.05, 0.95),
]

rows = []

for q_min, q_max in quantile_settings:
    score_min = kakao_scores.quantile(q_min)
    score_max = kakao_scores.quantile(q_max)

    def sentiment_to_star(score):
        if pd.isna(score):
            return np.nan

        if score_max == score_min:
            return np.nan

        normalized = (score - score_min) / (score_max - score_min)
        normalized = max(0, min(1, normalized))
        return round(1 + normalized * 4, 2)

    temp = kakao.copy()
    temp["sentiment_star_test"] = temp["category_total_score"].apply(sentiment_to_star)

    valid = temp.dropna(subset=["sentiment_star_test", "rating"]).copy()

    mae = (valid["sentiment_star_test"] - valid["rating"]).abs().mean()
    rmse = np.sqrt(((valid["sentiment_star_test"] - valid["rating"]) ** 2).mean())

    star_5_ratio = (temp["sentiment_star_test"] == 5).mean() * 100
    star_1_ratio = (temp["sentiment_star_test"] == 1).mean() * 100

    rows.append({
        "q_min": q_min,
        "q_max": q_max,
        "score_min": score_min,
        "score_max": score_max,
        "MAE": mae,
        "RMSE": rmse,
        "avg_sentiment_star": temp["sentiment_star_test"].mean(),
        "median_sentiment_star": temp["sentiment_star_test"].median(),
        "star_5_ratio_percent": star_5_ratio,
        "star_1_ratio_percent": star_1_ratio,
    })

result = pd.DataFrame(rows)
result = result.round(4)

print(result)

result.to_csv(
    BASE_DIR / "normalization_sensitivity_check.csv",
    index=False,
    encoding="utf-8-sig"
)