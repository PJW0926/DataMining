import pandas as pd
from pathlib import Path
from collections import defaultdict

# =========================================================
# 1. 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

review_file = BASE_DIR / "final_high_trust_reviews_pos.csv"
dict_file = BASE_DIR / "통합_감성사전_v8_다같이수정_토큰보완.csv"

output_file = BASE_DIR / "final_high_trust_reviews_category_sentiment_v8.csv"
platform_output = BASE_DIR / "platform_category_sentiment_summary.csv"
store_output = BASE_DIR / "store_category_sentiment_summary.csv"
review_score_output = BASE_DIR / "review_sentiment_scores.csv"

# =========================================================
# 2. 파일 불러오기
# =========================================================

df = pd.read_csv(review_file, encoding="utf-8-sig")
sent_dict = pd.read_csv(dict_file, encoding="utf-8-sig")

# 필요한 열만 사용
sent_dict = sent_dict[["word", "category", "polarity", "score"]].copy()

# 결측 제거
sent_dict = sent_dict.dropna(subset=["word", "category", "polarity", "score"])

# 자료형 정리
sent_dict["word"] = sent_dict["word"].astype(str).str.strip()
sent_dict["category"] = sent_dict["category"].astype(str).str.strip()
sent_dict["polarity"] = sent_dict["polarity"].astype(str).str.strip()

# score 숫자 변환
sent_dict["score"] = pd.to_numeric(sent_dict["score"], errors="coerce")
sent_dict = sent_dict.dropna(subset=["score"])
sent_dict["score"] = sent_dict["score"].astype(int)

# 리뷰 tokens 열 정리
df["tokens"] = df["tokens"].fillna("").astype(str)

# =========================================================
# 3. 카테고리 설정
# =========================================================
# 기존 4개 카테고리 + 애매한 단어용 general 추가

categories = ["food", "price", "service", "atmosphere", "general"]

# 혹시 사전에 오타/예상 밖 카테고리가 들어간 경우 확인용
dict_categories = sorted(sent_dict["category"].unique())
unknown_categories = [cat for cat in dict_categories if cat not in categories]

if unknown_categories:
    print("[주의] categories 목록에 없는 카테고리가 사전에 있습니다:")
    print(unknown_categories)
    print("이 카테고리들은 점수 계산에서 제외됩니다.")

# =========================================================
# 4. 사전 구조 만들기
# =========================================================
# 같은 단어가 여러 카테고리에 들어갈 수 있으므로 list 형태로 저장

sentiment_map = defaultdict(list)

for _, row in sent_dict.iterrows():
    word = row["word"]
    category = row["category"]
    score = row["score"]

    sentiment_map[word].append((category, score))

# =========================================================
# 5. 리뷰별 카테고리 감성점수 계산
# =========================================================
# 채점 방식:
# 1) 2어절 표현을 먼저 매칭
# 2) 2어절로 이미 매칭된 단어 위치는 1어절 계산에서 제외
# 3) 2어절로 안 잡힌 단어만 1어절로 계산
#
# 예:
# "고기 맛있다"가 사전에 있으면
# "고기", "맛있다"는 따로 점수 반영하지 않음
#
# general:
# 맛있다, 좋다, 괜찮다, 별로다처럼 특정 카테고리로 넣기 애매한 단어를
# general로 분류하면 general_score에 반영됨.

def analyze_review(tokens_text):
    tokens = str(tokens_text).split()

    scores = {cat: 0 for cat in categories}
    matched = {cat: [] for cat in categories}

    used_token_idx = set()   # 2어절에 사용된 1어절 위치
    matched_terms = set()    # 같은 표현 반복 점수 방지

    # =====================================================
    # 1단계: 2어절 표현 먼저 매칭
    # =====================================================
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i + 1]

        if bigram in sentiment_map and bigram not in matched_terms:
            for category, score in sentiment_map[bigram]:
                if category in scores:
                    scores[category] += score
                    matched[category].append(f"{bigram}:{score}")

            matched_terms.add(bigram)

            # 2어절로 이미 점수 반영된 단어들은 1어절 계산에서 제외
            used_token_idx.add(i)
            used_token_idx.add(i + 1)

    # =====================================================
    # 2단계: 2어절에 포함되지 않은 1어절만 매칭
    # =====================================================
    for i, token in enumerate(tokens):
        if i in used_token_idx:
            continue

        if token in sentiment_map and token not in matched_terms:
            for category, score in sentiment_map[token]:
                if category in scores:
                    scores[category] += score
                    matched[category].append(f"{token}:{score}")

            matched_terms.add(token)

    result = {}

    # 카테고리별 점수, 매칭 단어, 라벨 생성
    for cat in categories:
        result[f"{cat}_score"] = scores[cat]
        result[f"{cat}_matched_words"] = " ".join(matched[cat])

        if scores[cat] > 0:
            result[f"{cat}_label"] = "positive"
        elif scores[cat] < 0:
            result[f"{cat}_label"] = "negative"
        else:
            result[f"{cat}_label"] = "neutral"

    # 전체 감성점수: food + price + service + atmosphere + general
    result["category_total_score"] = sum(scores.values())

    if result["category_total_score"] > 0:
        result["category_total_label"] = "positive"
    elif result["category_total_score"] < 0:
        result["category_total_label"] = "negative"
    else:
        result["category_total_label"] = "neutral"

    return pd.Series(result)

result_df = df["tokens"].apply(analyze_review)
df = pd.concat([df, result_df], axis=1)

# =========================================================
# 6. 플랫폼별 요약
# =========================================================

if "platform" in df.columns:
    agg_dict = {
        "review_count": ("review_text", "count"),
        "avg_total_score": ("category_total_score", "mean"),
        "positive_ratio": ("category_total_label", lambda x: (x == "positive").mean()),
        "neutral_ratio": ("category_total_label", lambda x: (x == "neutral").mean()),
        "negative_ratio": ("category_total_label", lambda x: (x == "negative").mean())
    }

    for cat in categories:
        agg_dict[f"avg_{cat}_score"] = (f"{cat}_score", "mean")

    platform_summary = df.groupby("platform").agg(**agg_dict).reset_index()

    # 보기 좋게 컬럼 순서 정리
    platform_cols = (
        ["platform", "review_count"]
        + [f"avg_{cat}_score" for cat in categories]
        + ["avg_total_score", "positive_ratio", "neutral_ratio", "negative_ratio"]
    )

    platform_summary = platform_summary[platform_cols]
    platform_summary.to_csv(platform_output, index=False, encoding="utf-8-sig")

# =========================================================
# 7. 식당별 요약
# =========================================================

if "store_name" in df.columns and "platform" in df.columns:
    agg_dict = {
        "review_count": ("review_text", "count"),
        "avg_total_score": ("category_total_score", "mean"),
        "positive_ratio": ("category_total_label", lambda x: (x == "positive").mean()),
        "neutral_ratio": ("category_total_label", lambda x: (x == "neutral").mean()),
        "negative_ratio": ("category_total_label", lambda x: (x == "negative").mean())
    }

    for cat in categories:
        agg_dict[f"avg_{cat}_score"] = (f"{cat}_score", "mean")

    store_summary = df.groupby(["store_name", "platform"]).agg(**agg_dict).reset_index()

    # 보기 좋게 컬럼 순서 정리
    store_cols = (
        ["store_name", "platform", "review_count"]
        + [f"avg_{cat}_score" for cat in categories]
        + ["avg_total_score", "positive_ratio", "neutral_ratio", "negative_ratio"]
    )

    store_summary = store_summary[store_cols]
    store_summary.to_csv(store_output, index=False, encoding="utf-8-sig")

# =========================================================
# 8. 리뷰별 감성점수 확인용 파일 저장
# =========================================================

review_score_cols = [
    "platform",
    "store_name",
    "review_text",
    "tokens"
]

# 카테고리별 score
review_score_cols += [f"{cat}_score" for cat in categories]

# 전체 score
review_score_cols += [
    "category_total_score",
    "category_total_label"
]

# 카테고리별 label
review_score_cols += [f"{cat}_label" for cat in categories]

# 카테고리별 matched words
review_score_cols += [f"{cat}_matched_words" for cat in categories]

# 실제 존재하는 열만 저장
review_score_cols = [col for col in review_score_cols if col in df.columns]

df[review_score_cols].to_csv(
    review_score_output,
    index=False,
    encoding="utf-8-sig"
)

# 전체 결과 저장
df.to_csv(output_file, index=False, encoding="utf-8-sig")

# =========================================================
# 9. 실행 결과 출력
# =========================================================

print("완료:", output_file)
print("리뷰별 감성점수 확인 파일:", review_score_output)

if "platform" in df.columns:
    print("플랫폼 요약:", platform_output)

if "store_name" in df.columns and "platform" in df.columns:
    print("식당 요약:", store_output)

print("\n[사용 카테고리]")
print(categories)

print("\n[사전에 들어 있는 카테고리]")
print(dict_categories)

print("\n[감성분석 결과 미리보기]")

preview_cols = [
    "review_text",
    "tokens"
]

preview_cols += [f"{cat}_score" for cat in categories]
preview_cols += ["category_total_score", "category_total_label"]
preview_cols += [f"{cat}_matched_words" for cat in categories]

preview_cols = [col for col in preview_cols if col in df.columns]

print(df[preview_cols].head())