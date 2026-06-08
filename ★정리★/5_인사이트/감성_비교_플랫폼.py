import pandas as pd

df = pd.read_csv("final_high_trust_reviews_with_sentiment_star_v9_token_only_star_정규화수정.csv", encoding="utf-8-sig")

# 감성별점 기준 긍정 여부
df["is_positive"] = df["sentiment_star"] >= 4

platform_sentiment = df.groupby("platform").agg(
    review_count=("review_text", "count"),
    avg_sentiment_score=("sentiment_star", "mean"),
    median_sentiment_score=("sentiment_star", "median"),
    avg_sentiment_star=("sentiment_star", "mean"),
    positive_ratio=("is_positive", "mean")
).reset_index()

platform_sentiment["positive_ratio"] = platform_sentiment["positive_ratio"] * 100

print(platform_sentiment)