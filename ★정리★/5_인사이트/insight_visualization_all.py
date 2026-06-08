import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 0. 한글 폰트 설정
# =========================================================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 1. 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

input_files = {
    "all": BASE_DIR / "store_rank_change_all_vs_hightrust_all_trustlevel_high.csv",
    "naver": BASE_DIR / "store_rank_change_all_vs_hightrust_naver_trustlevel_high.csv",
    "kakao": BASE_DIR / "store_rank_change_all_vs_hightrust_kakao_trustlevel_high.csv",
}

OUTPUT_DIR = BASE_DIR / "insight_visualizations_trustlevel_high_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

# 변화 없음 판정 기준
# score_change가 ±0.005 이하면 변화 없음으로 처리
EPS = 0.005

# =========================================================
# 2. 데이터 불러오기
# =========================================================

dfs = {}

for scope, file_path in input_files.items():
    if not file_path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df["analysis_scope"] = scope

    numeric_cols = [
        "total_review_count",
        "high_trust_review_count",
        "all_avg_sentiment_star",
        "high_avg_sentiment_star",
        "score_change",
        "rank_all",
        "rank_high_trust",
        "rank_change",
        "kakao_total_rating_count",
        "kakao_all_avg_rating",
        "kakao_high_rating_count",
        "kakao_high_avg_rating",
        "kakao_rating_change",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["high_trust_ratio"] = (
        df["high_trust_review_count"] / df["total_review_count"] * 100
    )

    dfs[scope] = df

all_detail = pd.concat(dfs.values(), ignore_index=True)

print("=== 입력 파일 확인 ===")
for scope, df in dfs.items():
    print(scope, df.shape)
print()

# =========================================================
# 3. 공통 함수
# =========================================================

def save_bar(df, x_col, y_col, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col], df[y_col])
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("저장:", path)


def save_barh(df, value_col, label_col, title, xlabel, filename):
    plot_df = df.copy()

    plt.figure(figsize=(10, max(5, len(plot_df) * 0.45)))
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("저장:", path)


def save_scatter(df, x_col, y_col, title, xlabel, ylabel, filename):
    plot_df = df.dropna(subset=[x_col, y_col]).copy()

    plt.figure(figsize=(7, 7))
    plt.scatter(plot_df[x_col], plot_df[y_col])

    min_val = min(plot_df[x_col].min(), plot_df[y_col].min())
    max_val = max(plot_df[x_col].max(), plot_df[y_col].max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("저장:", path)


def save_hist(df, col, title, xlabel, filename, bins=10):
    plot_df = df.dropna(subset=[col]).copy()

    plt.figure(figsize=(8, 5))
    plt.hist(plot_df[col], bins=bins)
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("음식점 수")
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("저장:", path)


def save_grouped_bar(summary_df, metric_all_col, metric_high_col, title, ylabel, filename):
    plot_df = summary_df.copy()
    x = range(len(plot_df))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(
        [i - width / 2 for i in x],
        plot_df[metric_all_col],
        width=width,
        label="전체 리뷰 기준"
    )
    plt.bar(
        [i + width / 2 for i in x],
        plot_df[metric_high_col],
        width=width,
        label="high 리뷰 기준"
    )

    plt.xticks(list(x), plot_df["analysis_scope"])
    plt.title(title)
    plt.xlabel("분석 범위")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("저장:", path)


# =========================================================
# 4. 분석 범위별 요약표 생성
# =========================================================

summary_rows = []

for scope, df in dfs.items():
    score_up = (df["score_change"] > EPS).sum()
    score_down = (df["score_change"] < -EPS).sum()
    score_same = (df["score_change"].abs() <= EPS).sum()

    row = {
        "analysis_scope": scope,
        "store_count": len(df),
        "total_review_count_sum": df["total_review_count"].sum(),
        "high_trust_review_count_sum": df["high_trust_review_count"].sum(),
        "high_trust_ratio_total": df["high_trust_review_count"].sum() / df["total_review_count"].sum() * 100,

        "store_avg_all_sentiment_star": df["all_avg_sentiment_star"].mean(),
        "store_avg_high_sentiment_star": df["high_avg_sentiment_star"].mean(),
        "store_avg_score_change": df["score_change"].mean(),
        "store_avg_abs_score_change": df["score_change"].abs().mean(),

        "score_up_store_count": score_up,
        "score_down_store_count": score_down,
        "score_same_store_count": score_same,

        "rank_changed_store_count": (df["rank_change"] != 0).sum(),
        "rank_changed_store_ratio": (df["rank_change"] != 0).mean() * 100,
        "avg_rank_change": df["rank_change"].mean(),
        "avg_abs_rank_change": df["rank_change"].abs().mean(),
        "max_rank_up": df["rank_change"].max(),
        "max_rank_down": df["rank_change"].min(),
    }

    if "kakao_all_avg_rating" in df.columns and df["kakao_all_avg_rating"].notna().sum() > 0:
        rating_df = df.dropna(subset=["kakao_all_avg_rating", "kakao_high_avg_rating"]).copy()

        row["store_avg_kakao_all_rating"] = rating_df["kakao_all_avg_rating"].mean()
        row["store_avg_kakao_high_rating"] = rating_df["kakao_high_avg_rating"].mean()
        row["store_avg_kakao_rating_change"] = rating_df["kakao_rating_change"].mean()
        row["store_avg_abs_kakao_rating_change"] = rating_df["kakao_rating_change"].abs().mean()
        row["kakao_rating_up_store_count"] = (rating_df["kakao_rating_change"] > EPS).sum()
        row["kakao_rating_down_store_count"] = (rating_df["kakao_rating_change"] < -EPS).sum()
        row["kakao_rating_same_store_count"] = (rating_df["kakao_rating_change"].abs() <= EPS).sum()

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_output = OUTPUT_DIR / "summary_by_scope_trustlevel_high.csv"
summary_df.to_csv(summary_output, index=False, encoding="utf-8-sig")

detail_output = OUTPUT_DIR / "detail_all_scopes_trustlevel_high.csv"
all_detail.to_csv(detail_output, index=False, encoding="utf-8-sig")

print("요약표 저장:", summary_output)
print("통합 상세표 저장:", detail_output)
print()
print("=== 분석 범위별 요약 ===")
print(summary_df.round(3).to_string(index=False))
print()

# =========================================================
# 5. 전체/네이버/카카오 비교 시각화
# =========================================================

# 5-1. 전체 리뷰 기준 vs high 리뷰 기준 감성별점
save_grouped_bar(
    summary_df,
    "store_avg_all_sentiment_star",
    "store_avg_high_sentiment_star",
    "분석 범위별 전체 리뷰 기준 vs high 리뷰 기준 감성별점",
    "평균 감성별점",
    "01_scope_all_vs_high_sentiment_star.png"
)

# 5-2. 평균 감성별점 변화
save_bar(
    summary_df,
    "analysis_scope",
    "store_avg_score_change",
    "분석 범위별 high 리뷰 반영 후 감성별점 변화",
    "분석 범위",
    "감성별점 변화",
    "02_scope_score_change.png"
)

# 5-3. 평균 절대 감성별점 변화폭
save_bar(
    summary_df,
    "analysis_scope",
    "store_avg_abs_score_change",
    "분석 범위별 평균 감성별점 변화폭",
    "분석 범위",
    "절대 감성별점 변화",
    "03_scope_abs_score_change.png"
)

# 5-4. 순위 변동 음식점 비율
save_bar(
    summary_df,
    "analysis_scope",
    "rank_changed_store_ratio",
    "분석 범위별 순위 변동 음식점 비율",
    "분석 범위",
    "순위 변동 비율(%)",
    "04_scope_rank_changed_ratio.png"
)

# 5-5. 평균 순위 변화폭
save_bar(
    summary_df,
    "analysis_scope",
    "avg_abs_rank_change",
    "분석 범위별 평균 순위 변화폭",
    "분석 범위",
    "평균 순위 변화폭",
    "05_scope_abs_rank_change.png"
)

# 5-6. high 리뷰 비율
save_bar(
    summary_df,
    "analysis_scope",
    "high_trust_ratio_total",
    "분석 범위별 high 리뷰 비율",
    "분석 범위",
    "high 리뷰 비율(%)",
    "06_scope_high_trust_ratio.png"
)

# 5-7. 감성별점 상승/하락/변화 없음 음식점 수
score_count_df = summary_df[
    [
        "analysis_scope",
        "score_up_store_count",
        "score_down_store_count",
        "score_same_store_count"
    ]
].copy()

x = range(len(score_count_df))
width = 0.25

plt.figure(figsize=(8, 5))
plt.bar(
    [i - width for i in x],
    score_count_df["score_up_store_count"],
    width=width,
    label="상승"
)
plt.bar(
    list(x),
    score_count_df["score_down_store_count"],
    width=width,
    label="하락"
)
plt.bar(
    [i + width for i in x],
    score_count_df["score_same_store_count"],
    width=width,
    label="변화 없음"
)

plt.xticks(list(x), score_count_df["analysis_scope"])
plt.title("분석 범위별 감성별점 상승/하락/변화 없음 음식점 수")
plt.xlabel("분석 범위")
plt.ylabel("음식점 수")
plt.legend()
plt.tight_layout()

path = OUTPUT_DIR / "07_scope_score_up_down_same_count.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close()
print("저장:", path)

# =========================================================
# 6. 범위별 상세 시각화
# =========================================================

for scope, df in dfs.items():

    # 6-1. 식당별 all vs high 감성별점 비교
    compare_df = df.sort_values("high_avg_sentiment_star", ascending=True).copy()

    plt.figure(figsize=(10, max(6, len(compare_df) * 0.45)))
    y = range(len(compare_df))

    plt.scatter(compare_df["all_avg_sentiment_star"], y, label="전체 리뷰 기준")
    plt.scatter(compare_df["high_avg_sentiment_star"], y, label="high 리뷰 기준")

    for i, row in enumerate(compare_df.itertuples()):
        plt.plot(
            [row.all_avg_sentiment_star, row.high_avg_sentiment_star],
            [i, i],
            linewidth=1
        )

    plt.yticks(list(y), compare_df["store_name"])
    plt.title(f"[{scope}] 식당별 전체 리뷰 기준 vs high 리뷰 기준 감성별점")
    plt.xlabel("감성별점")
    plt.ylabel("")
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / f"08_{scope}_store_all_vs_high_sentiment_star.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("저장:", path)

    # 6-2. 감성별점 상승 TOP 10
    score_up = (
        df.sort_values("score_change", ascending=False)
        .head(10)
        .sort_values("score_change", ascending=True)
    )

    save_barh(
        score_up,
        "score_change",
        "store_name",
        f"[{scope}] high 리뷰 반영 후 감성별점 상승 TOP 10",
        "감성별점 변화",
        f"09_{scope}_score_up_top10.png"
    )

    # 6-3. 감성별점 하락 TOP 10
    score_down = (
        df.sort_values("score_change", ascending=True)
        .head(10)
        .sort_values("score_change", ascending=False)
    )

    save_barh(
        score_down,
        "score_change",
        "store_name",
        f"[{scope}] high 리뷰 반영 후 감성별점 하락 TOP 10",
        "감성별점 변화",
        f"10_{scope}_score_down_top10.png"
    )

    # 6-4. 순위 상승 TOP 10
    rank_up = (
        df.sort_values("rank_change", ascending=False)
        .head(10)
        .sort_values("rank_change", ascending=True)
    )

    save_barh(
        rank_up,
        "rank_change",
        "store_name",
        f"[{scope}] high 리뷰 반영 후 순위 상승 TOP 10",
        "순위 변화",
        f"11_{scope}_rank_up_top10.png"
    )

    # 6-5. 순위 하락 TOP 10
    rank_down = (
        df.sort_values("rank_change", ascending=True)
        .head(10)
        .sort_values("rank_change", ascending=False)
    )

    save_barh(
        rank_down,
        "rank_change",
        "store_name",
        f"[{scope}] high 리뷰 반영 후 순위 하락 TOP 10",
        "순위 변화",
        f"12_{scope}_rank_down_top10.png"
    )

    # 6-6. 전체 감성별점 vs high 감성별점 산점도
    save_scatter(
        df,
        "all_avg_sentiment_star",
        "high_avg_sentiment_star",
        f"[{scope}] 전체 리뷰 기준 vs high 리뷰 기준 감성별점",
        "전체 리뷰 기준 감성별점",
        "high 리뷰 기준 감성별점",
        f"13_{scope}_all_vs_high_sentiment_scatter.png"
    )

    # 6-7. 감성별점 변화 분포
    save_hist(
        df,
        "score_change",
        f"[{scope}] high 리뷰 반영 후 감성별점 변화 분포",
        "감성별점 변화",
        f"14_{scope}_score_change_distribution.png",
        bins=8
    )

    # 6-8. 순위 변화 분포
    save_hist(
        df,
        "rank_change",
        f"[{scope}] high 리뷰 반영 후 순위 변화 분포",
        "순위 변화",
        f"15_{scope}_rank_change_distribution.png",
        bins=10
    )

    # 6-9. high 리뷰 비율 상위 15개 식당
    high_ratio_top = (
        df.sort_values("high_trust_ratio", ascending=False)
        .head(15)
        .sort_values("high_trust_ratio", ascending=True)
    )

    save_barh(
        high_ratio_top,
        "high_trust_ratio",
        "store_name",
        f"[{scope}] high 리뷰 비율 상위 음식점",
        "high 리뷰 비율(%)",
        f"16_{scope}_high_trust_ratio_top15.png"
    )

    # 6-10. high 리뷰 비율과 감성별점 변화 관계
    save_scatter(
        df,
        "high_trust_ratio",
        "score_change",
        f"[{scope}] high 리뷰 비율과 감성별점 변화",
        "high 리뷰 비율(%)",
        "감성별점 변화",
        f"17_{scope}_high_ratio_vs_score_change.png"
    )

# =========================================================
# 7. 카카오 실제 별점 관련 시각화
# =========================================================

for scope in ["all", "kakao"]:
    df = dfs[scope].copy()

    required_rating_cols = [
        "kakao_all_avg_rating",
        "kakao_high_avg_rating",
        "kakao_rating_change",
    ]

    if not all(col in df.columns for col in required_rating_cols):
        continue

    rating_df = df.dropna(subset=required_rating_cols).copy()

    if len(rating_df) == 0:
        continue

    # 7-1. 카카오 실제 별점 전체 vs high 비교
    compare_df = rating_df.sort_values("kakao_high_avg_rating", ascending=True).copy()

    plt.figure(figsize=(10, max(6, len(compare_df) * 0.45)))
    y = range(len(compare_df))

    plt.scatter(compare_df["kakao_all_avg_rating"], y, label="전체 카카오 실제 별점")
    plt.scatter(compare_df["kakao_high_avg_rating"], y, label="high 카카오 실제 별점")

    for i, row in enumerate(compare_df.itertuples()):
        plt.plot(
            [row.kakao_all_avg_rating, row.kakao_high_avg_rating],
            [i, i],
            linewidth=1
        )

    plt.yticks(list(y), compare_df["store_name"])
    plt.title(f"[{scope}] 카카오 실제 별점 전체 기준 vs high 기준")
    plt.xlabel("카카오 실제 별점")
    plt.ylabel("")
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / f"18_{scope}_kakao_all_vs_high_rating_by_store.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("저장:", path)

    # 7-2. 카카오 실제 별점 상승 TOP 10
    rating_up = (
        rating_df.sort_values("kakao_rating_change", ascending=False)
        .head(10)
        .sort_values("kakao_rating_change", ascending=True)
    )

    save_barh(
        rating_up,
        "kakao_rating_change",
        "store_name",
        f"[{scope}] high 리뷰 반영 후 카카오 실제 별점 상승 TOP 10",
        "카카오 실제 별점 변화",
        f"19_{scope}_kakao_rating_up_top10.png"
    )

    # 7-3. 카카오 실제 별점 하락 TOP 10
    rating_down = (
        rating_df.sort_values("kakao_rating_change", ascending=True)
        .head(10)
        .sort_values("kakao_rating_change", ascending=False)
    )

    save_barh(
        rating_down,
        "kakao_rating_change",
        "store_name",
        f"[{scope}] high 리뷰 반영 후 카카오 실제 별점 하락 TOP 10",
        "카카오 실제 별점 변화",
        f"20_{scope}_kakao_rating_down_top10.png"
    )

    # 7-4. 카카오 전체 실제 별점 vs high 실제 별점 산점도
    save_scatter(
        rating_df,
        "kakao_all_avg_rating",
        "kakao_high_avg_rating",
        f"[{scope}] 카카오 전체 실제 별점 vs high 실제 별점",
        "전체 카카오 실제 별점",
        "high 카카오 실제 별점",
        f"21_{scope}_kakao_all_vs_high_rating_scatter.png"
    )

    # 7-5. 카카오 실제 별점 변화 분포
    save_hist(
        rating_df,
        "kakao_rating_change",
        f"[{scope}] high 리뷰 반영 후 카카오 실제 별점 변화 분포",
        "카카오 실제 별점 변화",
        f"22_{scope}_kakao_rating_change_distribution.png",
        bins=8
    )

    # 7-6. 카카오 실제 별점 상승/하락/변화 없음 음식점 수
    rating_up_count = (rating_df["kakao_rating_change"] > EPS).sum()
    rating_down_count = (rating_df["kakao_rating_change"] < -EPS).sum()
    rating_same_count = (rating_df["kakao_rating_change"].abs() <= EPS).sum()

    rating_count_df = pd.DataFrame({
        "change_type": ["상승", "하락", "변화 없음"],
        "store_count": [rating_up_count, rating_down_count, rating_same_count]
    })

    plt.figure(figsize=(7, 5))
    plt.bar(rating_count_df["change_type"], rating_count_df["store_count"])
    plt.title(f"[{scope}] 카카오 실제 별점 상승/하락/변화 없음 음식점 수")
    plt.xlabel("변화 유형")
    plt.ylabel("음식점 수")
    plt.tight_layout()

    path = OUTPUT_DIR / f"23_{scope}_kakao_rating_up_down_same_count.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("저장:", path)

# =========================================================
# 8. 보고서용 표 저장
# =========================================================

key_summary_cols = [
    "analysis_scope",
    "store_count",
    "total_review_count_sum",
    "high_trust_review_count_sum",
    "high_trust_ratio_total",
    "store_avg_all_sentiment_star",
    "store_avg_high_sentiment_star",
    "store_avg_score_change",
    "score_up_store_count",
    "score_down_store_count",
    "score_same_store_count",
    "rank_changed_store_count",
    "rank_changed_store_ratio",
    "avg_abs_rank_change",
]

extra_rating_cols = [
    "store_avg_kakao_all_rating",
    "store_avg_kakao_high_rating",
    "store_avg_kakao_rating_change",
    "kakao_rating_up_store_count",
    "kakao_rating_down_store_count",
    "kakao_rating_same_store_count",
]

key_summary_cols += [col for col in extra_rating_cols if col in summary_df.columns]

key_summary = summary_df[key_summary_cols].copy()
key_summary = key_summary.round(3)

key_summary_output = OUTPUT_DIR / "key_summary_for_report.csv"
key_summary.to_csv(key_summary_output, index=False, encoding="utf-8-sig")

print("보고서용 핵심 요약표 저장:", key_summary_output)

# =========================================================
# 9. 핵심 사례표 저장
# =========================================================

case_rows = []

for scope, df in dfs.items():
    top_score_up = df.sort_values("score_change", ascending=False).head(5).copy()
    top_score_up["case_type"] = "감성별점 상승 TOP5"
    top_score_up["analysis_scope"] = scope

    top_score_down = df.sort_values("score_change", ascending=True).head(5).copy()
    top_score_down["case_type"] = "감성별점 하락 TOP5"
    top_score_down["analysis_scope"] = scope

    top_rank_up = df.sort_values("rank_change", ascending=False).head(5).copy()
    top_rank_up["case_type"] = "순위 상승 TOP5"
    top_rank_up["analysis_scope"] = scope

    top_rank_down = df.sort_values("rank_change", ascending=True).head(5).copy()
    top_rank_down["case_type"] = "순위 하락 TOP5"
    top_rank_down["analysis_scope"] = scope

    case_rows.extend([top_score_up, top_score_down, top_rank_up, top_rank_down])

case_table = pd.concat(case_rows, ignore_index=True)

case_cols = [
    "analysis_scope",
    "case_type",
    "store_name",
    "total_review_count",
    "high_trust_review_count",
    "all_avg_sentiment_star",
    "high_avg_sentiment_star",
    "score_change",
    "rank_all",
    "rank_high_trust",
    "rank_change",
]

case_cols = [col for col in case_cols if col in case_table.columns]

case_table = case_table[case_cols].copy()
case_table = case_table.round(3)

case_output = OUTPUT_DIR / "key_cases_for_report.csv"
case_table.to_csv(case_output, index=False, encoding="utf-8-sig")

print("보고서용 핵심 사례표 저장:", case_output)

# =========================================================
# 10. 콘솔 핵심 인사이트 출력
# =========================================================

print("\n================ 핵심 인사이트 후보 ================")

for scope, df in dfs.items():
    print(f"\n[{scope}]")

    store_count = len(df)
    total_reviews = df["total_review_count"].sum()
    high_reviews = df["high_trust_review_count"].sum()
    high_ratio = high_reviews / total_reviews * 100

    avg_all = df["all_avg_sentiment_star"].mean()
    avg_high = df["high_avg_sentiment_star"].mean()
    avg_change = df["score_change"].mean()

    score_up_count = (df["score_change"] > EPS).sum()
    score_down_count = (df["score_change"] < -EPS).sum()
    score_same_count = (df["score_change"].abs() <= EPS).sum()

    rank_changed = (df["rank_change"] != 0).sum()
    rank_changed_ratio = rank_changed / store_count * 100
    avg_abs_rank_change = df["rank_change"].abs().mean()

    print(f"- 분석 음식점 수: {store_count}개")
    print(f"- 전체 리뷰 수: {int(total_reviews):,}개")
    print(f"- high 리뷰 수: {int(high_reviews):,}개 ({high_ratio:.1f}%)")
    print(f"- 전체 리뷰 기준 평균 감성별점: {avg_all:.2f}")
    print(f"- high 리뷰 기준 평균 감성별점: {avg_high:.2f}")
    print(f"- 평균 감성별점 변화: {avg_change:+.2f}")
    print(f"- 감성별점 상승/하락/변화 없음 음식점 수: {score_up_count}개 상승, {score_down_count}개 하락, {score_same_count}개 변화 없음")
    print(f"- 순위 변동 음식점 수: {rank_changed}/{store_count}개 ({rank_changed_ratio:.1f}%)")
    print(f"- 평균 순위 변화폭: {avg_abs_rank_change:.2f}계단")

    max_score_up = df.sort_values("score_change", ascending=False).iloc[0]
    max_score_down = df.sort_values("score_change", ascending=True).iloc[0]
    max_rank_up = df.sort_values("rank_change", ascending=False).iloc[0]
    max_rank_down = df.sort_values("rank_change", ascending=True).iloc[0]

    print(
        f"- 감성별점 상승 최대: {max_score_up['store_name']} "
        f"({max_score_up['all_avg_sentiment_star']:.2f} → "
        f"{max_score_up['high_avg_sentiment_star']:.2f}, "
        f"{max_score_up['score_change']:+.2f})"
    )

    print(
        f"- 감성별점 하락 최대: {max_score_down['store_name']} "
        f"({max_score_down['all_avg_sentiment_star']:.2f} → "
        f"{max_score_down['high_avg_sentiment_star']:.2f}, "
        f"{max_score_down['score_change']:+.2f})"
    )

    print(
        f"- 순위 상승 최대: {max_rank_up['store_name']} "
        f"({int(max_rank_up['rank_all'])}위 → "
        f"{int(max_rank_up['rank_high_trust'])}위, "
        f"{int(max_rank_up['rank_change']):+d}계단)"
    )

    print(
        f"- 순위 하락 최대: {max_rank_down['store_name']} "
        f"({int(max_rank_down['rank_all'])}위 → "
        f"{int(max_rank_down['rank_high_trust'])}위, "
        f"{int(max_rank_down['rank_change']):+d}계단)"
    )

    if "kakao_rating_change" in df.columns and df["kakao_rating_change"].notna().sum() > 0:
        rating_df = df.dropna(subset=["kakao_rating_change"]).copy()

        avg_rating_all = rating_df["kakao_all_avg_rating"].mean()
        avg_rating_high = rating_df["kakao_high_avg_rating"].mean()
        avg_rating_change = rating_df["kakao_rating_change"].mean()

        rating_up_count = (rating_df["kakao_rating_change"] > EPS).sum()
        rating_down_count = (rating_df["kakao_rating_change"] < -EPS).sum()
        rating_same_count = (rating_df["kakao_rating_change"].abs() <= EPS).sum()

        max_rating_up = rating_df.sort_values("kakao_rating_change", ascending=False).iloc[0]
        max_rating_down = rating_df.sort_values("kakao_rating_change", ascending=True).iloc[0]

        print(f"- 카카오 전체 실제 별점 평균: {avg_rating_all:.2f}")
        print(f"- 카카오 high 실제 별점 평균: {avg_rating_high:.2f}")
        print(f"- 카카오 실제 별점 변화: {avg_rating_change:+.2f}")
        print(f"- 카카오 실제 별점 상승/하락/변화 없음 음식점 수: {rating_up_count}개 상승, {rating_down_count}개 하락, {rating_same_count}개 변화 없음")

        print(
            f"- 카카오 실제 별점 상승 최대: {max_rating_up['store_name']} "
            f"({max_rating_up['kakao_all_avg_rating']:.2f} → "
            f"{max_rating_up['kakao_high_avg_rating']:.2f}, "
            f"{max_rating_up['kakao_rating_change']:+.2f})"
        )

        print(
            f"- 카카오 실제 별점 하락 최대: {max_rating_down['store_name']} "
            f"({max_rating_down['kakao_all_avg_rating']:.2f} → "
            f"{max_rating_down['kakao_high_avg_rating']:.2f}, "
            f"{max_rating_down['kakao_rating_change']:+.2f})"
        )

print("\n================ 저장 완료 ================")
print("시각화 폴더:", OUTPUT_DIR)