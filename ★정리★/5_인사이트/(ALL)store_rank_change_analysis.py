import pandas as pd
from pathlib import Path

# =========================================================
# 1. 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

input_file = BASE_DIR / "crawling_all_filter_with_sentiment_star_v9_token_only_정규화수정.csv"

# =========================================================
# 2. 플랫폼 필터 설정
# =========================================================
# 전체 분석: PLATFORM_FILTER = None
# 네이버만 분석: PLATFORM_FILTER = "naver"
# 카카오만 분석: PLATFORM_FILTER = "kakao"

PLATFORM_FILTER = None
# PLATFORM_FILTER = "naver"
# PLATFORM_FILTER = "kakao"

suffix = PLATFORM_FILTER if PLATFORM_FILTER is not None else "all"

output_file = BASE_DIR / f"store_rank_change_all_vs_hightrust_{suffix}_trustlevel_high.csv"
rank_up_file = BASE_DIR / f"store_rank_up_top5_{suffix}_trustlevel_high.csv"
rank_down_file = BASE_DIR / f"store_rank_down_top5_{suffix}_trustlevel_high.csv"

# =========================================================
# 3. 데이터 불러오기
# =========================================================

df = pd.read_csv(input_file, encoding="utf-8-sig")

print("=== 전체 컬럼 확인 ===")
print(df.columns.tolist())
print()

# =========================================================
# 4. 사용할 컬럼명 설정
# =========================================================

STORE_COL = "store_name"
PLATFORM_COL = "platform"
TEXT_COL = "review_text"
SENTIMENT_STAR_COL = "sentiment_star"

# 중요: 고신뢰 기준은 pred_label이 아니라 trust_level == high
TRUST_COL = "trust_level"
HIGH_TRUST_VALUE = "high"

RATING_COL = "rating" if "rating" in df.columns else None

# =========================================================
# 5. 필수 컬럼 확인
# =========================================================

required_cols = [
    STORE_COL,
    PLATFORM_COL,
    TEXT_COL,
    SENTIMENT_STAR_COL,
    TRUST_COL,
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"필수 컬럼이 없습니다: {col}")

# =========================================================
# 6. 플랫폼 필터 적용
# =========================================================

df[PLATFORM_COL] = df[PLATFORM_COL].astype(str).str.lower().str.strip()
df[TRUST_COL] = df[TRUST_COL].astype(str).str.lower().str.strip()

if PLATFORM_FILTER is not None:
    df = df[df[PLATFORM_COL] == PLATFORM_FILTER].copy()

    print(f"=== {PLATFORM_FILTER} 데이터만 분석 ===")
    print("필터링 후 리뷰 수:", len(df))
    print()

    if len(df) == 0:
        raise ValueError(f"{PLATFORM_FILTER} 데이터가 없습니다. platform 컬럼 값을 확인하세요.")
else:
    print("=== 전체 플랫폼 통합 분석 ===")
    print("전체 리뷰 수:", len(df))
    print()

# =========================================================
# 7. 기본 전처리
# =========================================================

df[SENTIMENT_STAR_COL] = pd.to_numeric(df[SENTIMENT_STAR_COL], errors="coerce")

if RATING_COL is not None:
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")

df = df.dropna(subset=[STORE_COL])
df = df.dropna(subset=[SENTIMENT_STAR_COL])

# 리뷰 수 기준
MIN_TOTAL_REVIEWS = 5
MIN_HIGH_TRUST_REVIEWS = 3

# 플랫폼별로 너무 많이 빠지면 아래로 낮춰도 됨
# MIN_TOTAL_REVIEWS = 3
# MIN_HIGH_TRUST_REVIEWS = 2

print("=== 플랫폼별 리뷰 수 ===")
print(df[PLATFORM_COL].value_counts(dropna=False))
print()

print("=== trust_level 분포 ===")
print(df[TRUST_COL].value_counts(dropna=False))
print()

print(f"=== 고신뢰 기준: {TRUST_COL} == '{HIGH_TRUST_VALUE}' ===")
print("고신뢰 리뷰 수:", (df[TRUST_COL] == HIGH_TRUST_VALUE).sum())
print()

# =========================================================
# 8. 전체 리뷰 기준 음식점 평균
# =========================================================

store_all = df.groupby(STORE_COL).agg(
    total_review_count=(TEXT_COL, "count"),
    all_avg_sentiment_star=(SENTIMENT_STAR_COL, "mean"),
    all_median_sentiment_star=(SENTIMENT_STAR_COL, "median")
).reset_index()

# 플랫폼별 리뷰 수 붙이기
platform_count = (
    df.pivot_table(
        index=STORE_COL,
        columns=PLATFORM_COL,
        values=TEXT_COL,
        aggfunc="count",
        fill_value=0
    )
    .reset_index()
)

platform_count.columns = [
    STORE_COL if col == STORE_COL else f"{col}_review_count"
    for col in platform_count.columns
]

store_all = pd.merge(store_all, platform_count, on=STORE_COL, how="left")

# =========================================================
# 9. 고신뢰 리뷰 기준 음식점 평균
# =========================================================

high_df = df[df[TRUST_COL] == HIGH_TRUST_VALUE].copy()

store_high = high_df.groupby(STORE_COL).agg(
    high_trust_review_count=(TEXT_COL, "count"),
    high_avg_sentiment_star=(SENTIMENT_STAR_COL, "mean"),
    high_median_sentiment_star=(SENTIMENT_STAR_COL, "median")
).reset_index()

# =========================================================
# 10. 전체 기준 vs 고신뢰 기준 합치기
# =========================================================

store_compare = pd.merge(store_all, store_high, on=STORE_COL, how="left")

# 고신뢰 리뷰가 없는 음식점 제외
store_compare = store_compare.dropna(subset=["high_avg_sentiment_star"])

# 최소 리뷰 수 기준 적용
store_compare = store_compare[
    (store_compare["total_review_count"] >= MIN_TOTAL_REVIEWS) &
    (store_compare["high_trust_review_count"] >= MIN_HIGH_TRUST_REVIEWS)
].copy()

if len(store_compare) == 0:
    raise ValueError(
        "분석 대상 음식점이 0개입니다. "
        "MIN_TOTAL_REVIEWS 또는 MIN_HIGH_TRUST_REVIEWS 기준을 낮춰보세요."
    )

# =========================================================
# 11. 점수 변화 계산
# =========================================================

store_compare["score_change"] = (
    store_compare["high_avg_sentiment_star"] -
    store_compare["all_avg_sentiment_star"]
)

store_compare["score_change_abs"] = store_compare["score_change"].abs()

# =========================================================
# 12. 순위 계산
# =========================================================

store_compare["rank_all"] = store_compare["all_avg_sentiment_star"].rank(
    ascending=False,
    method="min"
).astype(int)

store_compare["rank_high_trust"] = store_compare["high_avg_sentiment_star"].rank(
    ascending=False,
    method="min"
).astype(int)

# 양수면 고신뢰 기준에서 순위 상승
# 음수면 고신뢰 기준에서 순위 하락
store_compare["rank_change"] = (
    store_compare["rank_all"] -
    store_compare["rank_high_trust"]
)

# =========================================================
# 13. 카카오 실제 별점 추가 비교
# =========================================================

if RATING_COL is not None:
    kakao_df = df[df[PLATFORM_COL] == "kakao"].copy()

    if len(kakao_df) > 0:
        kakao_high_df = kakao_df[kakao_df[TRUST_COL] == HIGH_TRUST_VALUE].copy()

        kakao_all_rating = kakao_df.groupby(STORE_COL).agg(
            kakao_total_rating_count=(RATING_COL, "count"),
            kakao_all_avg_rating=(RATING_COL, "mean")
        ).reset_index()

        kakao_high_rating = kakao_high_df.groupby(STORE_COL).agg(
            kakao_high_rating_count=(RATING_COL, "count"),
            kakao_high_avg_rating=(RATING_COL, "mean")
        ).reset_index()

        kakao_rating_compare = pd.merge(
            kakao_all_rating,
            kakao_high_rating,
            on=STORE_COL,
            how="left"
        )

        kakao_rating_compare["kakao_rating_change"] = (
            kakao_rating_compare["kakao_high_avg_rating"] -
            kakao_rating_compare["kakao_all_avg_rating"]
        )

        store_compare = pd.merge(
            store_compare,
            kakao_rating_compare,
            on=STORE_COL,
            how="left"
        )

# =========================================================
# 14. 컬럼 순서 정리
# =========================================================

base_cols = [
    STORE_COL,
    "total_review_count",
    "high_trust_review_count",
    "all_avg_sentiment_star",
    "high_avg_sentiment_star",
    "score_change",
    "rank_all",
    "rank_high_trust",
    "rank_change"
]

platform_review_cols = [
    col for col in store_compare.columns
    if col.endswith("_review_count")
]

kakao_rating_cols = [
    col for col in [
        "kakao_total_rating_count",
        "kakao_all_avg_rating",
        "kakao_high_rating_count",
        "kakao_high_avg_rating",
        "kakao_rating_change"
    ]
    if col in store_compare.columns
]

final_cols = base_cols + platform_review_cols + kakao_rating_cols
final_cols = list(dict.fromkeys(final_cols))

store_compare = store_compare[final_cols].copy()

# 소수점 정리
round_cols = [
    "all_avg_sentiment_star",
    "high_avg_sentiment_star",
    "score_change",
    "kakao_all_avg_rating",
    "kakao_high_avg_rating",
    "kakao_rating_change"
]

for col in round_cols:
    if col in store_compare.columns:
        store_compare[col] = store_compare[col].round(2)

# 고신뢰 기준 순위로 정렬
store_compare = store_compare.sort_values("rank_high_trust")

# =========================================================
# 15. 결과 저장
# =========================================================

store_compare.to_csv(output_file, index=False, encoding="utf-8-sig")

rank_up = store_compare.sort_values(
    "rank_change",
    ascending=False
).head(5)

rank_down = store_compare.sort_values(
    "rank_change",
    ascending=True
).head(5)

rank_up.to_csv(rank_up_file, index=False, encoding="utf-8-sig")
rank_down.to_csv(rank_down_file, index=False, encoding="utf-8-sig")

# =========================================================
# 16. 콘솔 출력
# =========================================================

print("=== 전체 기준 vs 고신뢰 기준 음식점 순위 비교 ===")
print(store_compare)
print()

print("=== 고신뢰 리뷰 기준에서 순위 상승 TOP 5 ===")
print(rank_up[[STORE_COL, "rank_all", "rank_high_trust", "rank_change", "score_change"]])
print()

print("=== 고신뢰 리뷰 기준에서 순위 하락 TOP 5 ===")
print(rank_down[[STORE_COL, "rank_all", "rank_high_trust", "rank_change", "score_change"]])
print()

print("=== 저장 완료 ===")
print(f"전체 비교 결과: {output_file}")
print(f"순위 상승 TOP 5: {rank_up_file}")
print(f"순위 하락 TOP 5: {rank_down_file}")