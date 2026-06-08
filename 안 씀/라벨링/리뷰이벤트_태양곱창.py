import pandas as pd
import re

# =========================
# 1. CSV 불러오기
# =========================
event_df = pd.read_csv("라벨링/태양곱창 크롤링.csv", encoding="cp949")
normal_df = pd.read_csv("라벨링/쭌곱창 크롤링.csv", encoding="cp949")

event_df["group"] = "event"
normal_df["group"] = "normal"

df = pd.concat([event_df, normal_df], ignore_index=True)

df["리뷰 내용"] = df["리뷰 내용"].fillna("").astype(str)
df["계정의 리뷰 수"] = pd.to_numeric(df["계정의 리뷰 수"], errors="coerce").fillna(0)


# =========================
# 2. Feature 단어 사전
# =========================
food_detail_words = [
    "식감", "매콤", "양", "국물", "재료", "조리",
    "잡내", "얼큰", "푸짐", "듬뿍", "부드러"
]

menu_words = [
    "곱창전골", "돼지곱창전골", "마라곱창", "볶음밥",
    "감자전", "콘치즈", "치즈감자전", "야곱", "곱창"
]

experience_words = [
    "웨이팅", "대기", "직원", "서비스", "자리",
    "분위기", "가격", "포장", "가성비"
]

context_words = [
    "점심", "저녁", "주말", "평일", "겨울",
    "회식", "데이트", "친구", "가족", "2차",
    "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
    "월욜", "화욜", "수욜", "목욜", "금욜", "토욜", "일욜"
]

generic_praise_words = [
    "맛있", "최고", "넘"
]

intensifier_words = [
    "진짜", "너무", "완전", "엄청", "존맛", "대박", "짱"
]

sns_event_words = [
    "인스타", "릴스", "SNS", "이벤트", "서비스"
]

positive_words = [
    "맛있", "좋", "최고", "강추", "추천", "든든", "괜찮"
]

negative_words = [
    "별로", "아쉽", "비싸", "길", "불편",
    "실망", "밍밍", "맹맹", "최악", "짜"
]

contrast_words = [
    "다만", "근데", "하지만", "아쉽", "그래도", "긴 해", "긴해"
]

trendy_revisit_words = [
    "재방문", "또 올게요", "또올게요", "또 갈게요", "또갈게요"
]


# =========================
# 3. 보조 함수
# =========================
def count_keywords(text, words):
    return sum(text.count(word) for word in words)

def has_mixed_sentiment(text):
    has_pos = any(word in text for word in positive_words)
    has_neg = any(word in text for word in negative_words)
    has_contrast = any(word in text for word in contrast_words)
    return int((has_pos and has_neg) or (has_pos and has_contrast))

def count_number_patterns(text):
    patterns = [
        r"\d+\s*분",
        r"\d+\s*인분",
        r"\d+\s*원",
        r"\d+\s*명",
        r"\d+\s*시",
        r"\d+\s*차",
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


# =========================
# 4. 기본 Feature 생성
# =========================
df["review_length"] = df["리뷰 내용"].str.len()
df["length_over_30"] = (df["review_length"] >= 30).astype(int)
df["length_over_60"] = (df["review_length"] >= 60).astype(int)

df["food_detail_count"] = df["리뷰 내용"].apply(lambda x: count_keywords(x, food_detail_words))
df["has_food_detail"] = (df["food_detail_count"] > 0).astype(int)
df["strong_food_detail_review"] = (df["food_detail_count"] >= 2).astype(int)

df["menu_count"] = df["리뷰 내용"].apply(lambda x: count_keywords(x, menu_words))
df["has_menu"] = (df["menu_count"] > 0).astype(int)
df["rich_menu_review"] = (df["menu_count"] >= 2).astype(int)

df["experience_count"] = df["리뷰 내용"].apply(lambda x: count_keywords(x, experience_words))
df["has_experience"] = (df["experience_count"] > 0).astype(int)
df["strong_experience_review"] = (df["experience_count"] >= 2).astype(int)

df["context_count"] = df["리뷰 내용"].apply(lambda x: count_keywords(x, context_words))
df["has_context"] = (df["context_count"] > 0).astype(int)

df["has_any_number"] = df["리뷰 내용"].str.contains(r"\d", regex=True).astype(int)
df["number_detail_count"] = df["리뷰 내용"].apply(count_number_patterns)
df["has_number_detail"] = (df["number_detail_count"] > 0).astype(int)

df["generic_praise_count"] = df["리뷰 내용"].apply(lambda x: count_keywords(x, generic_praise_words))
df["has_generic_praise"] = (df["generic_praise_count"] > 0).astype(int)

df["intensifier_count"] = df["리뷰 내용"].apply(lambda x: count_keywords(x, intensifier_words))
df["intensifier_over_3"] = (df["intensifier_count"] >= 3).astype(int)

df["exclamation_count"] = df["리뷰 내용"].str.count("!")
df["exclamation_over_3"] = (df["exclamation_count"] >= 3).astype(int)

df["sns_event_count"] = df["리뷰 내용"].apply(lambda x: count_keywords(x, sns_event_words))
df["has_sns_event"] = (df["sns_event_count"] > 0).astype(int)

df["mixed_sentiment"] = df["리뷰 내용"].apply(has_mixed_sentiment)

df["trendy_revisit_count"] = df["리뷰 내용"].apply(lambda x: count_keywords(x, trendy_revisit_words))
df["has_trendy_revisit_phrase"] = (df["trendy_revisit_count"] > 0).astype(int)

df["account_review_count"] = df["계정의 리뷰 수"]
df["account_review_count_is_1"] = (df["account_review_count"] == 1).astype(int)
df["account_review_count_over_10"] = (df["account_review_count"] >= 10).astype(int)
df["account_review_count_over_50"] = (df["account_review_count"] >= 50).astype(int)


# =========================
# 5. 조합 Feature 생성
# =========================

df["menu_and_food_detail"] = (
    (df["has_menu"] == 1) & (df["has_food_detail"] == 1)
).astype(int)

df["short_generic_review"] = (
    (df["review_length"] < 30) & (df["has_generic_praise"] == 1)
).astype(int)

df["ultra_short_praise"] = (
    (df["review_length"] < 10) & (df["has_generic_praise"] == 1)
).astype(int)

df["generic_without_detail"] = (
    (df["has_generic_praise"] == 1)
    & (df["has_food_detail"] == 0)
    & (df["has_experience"] == 0)
    & (df["has_context"] == 0)
    & (df["has_number_detail"] == 0)
).astype(int)

df["menu_only_praise"] = (
    (df["has_menu"] == 1)
    & (df["has_generic_praise"] == 1)
    & (df["has_food_detail"] == 0)
    & (df["has_experience"] == 0)
    & (df["has_context"] == 0)
    & (df["has_number_detail"] == 0)
).astype(int)

df["experience_rich_review"] = (
    (df["has_experience"] == 1)
    | (df["has_context"] == 1)
    | (df["has_number_detail"] == 1)
).astype(int)

df["high_specificity_review"] = (
    (df["has_menu"] == 1)
    & (df["has_food_detail"] == 1)
    & (df["length_over_30"] == 1)
).astype(int)

df["rich_context_review"] = (
    (df["length_over_60"] == 1)
    & ((df["has_context"] == 1) | (df["has_experience"] == 1))
    & (df["food_detail_count"] >= 2)
).astype(int)

df["low_information_review"] = (
    (df["review_length"] < 30)
    & (df["has_food_detail"] == 0)
    & (df["has_experience"] == 0)
    & (df["has_context"] == 0)
    & (df["has_number_detail"] == 0)
).astype(int)

df["weak_detail_review"] = (
    (df["food_detail_count"] == 1)
    & (df["review_length"] < 40)
).astype(int)

df["weak_experience_review"] = (
    (df["experience_count"] == 1)
    & (df["review_length"] < 40)
    & (df["strong_experience_review"] == 0)
).astype(int)

df["weak_context_review"] = (
    (df["has_context"] == 1)
    & (df["review_length"] < 50)
    & (df["has_food_detail"] == 0)
    & (df["strong_experience_review"] == 0)
).astype(int)


# =========================
# 6. v6 Trust Score
# =========================
df["high_trust_score_v6"] = (
    df["length_over_30"] * 1
    + df["has_food_detail"] * 0.2
    + df["has_menu"] * 0.1
    + df["menu_and_food_detail"] * 1
    + df["strong_food_detail_review"] * 2
    + df["rich_menu_review"] * 2
    + df["has_experience"] * 0.5
    + df["strong_experience_review"] * 3
    + df["has_context"] * 0.5
    + df["has_number_detail"] * 2
    + df["experience_rich_review"] * 2
    + df["rich_context_review"] * 4
    + df["high_specificity_review"] * 1
    + df["mixed_sentiment"] * 4
    + df["account_review_count_over_10"] * 1
    + df["account_review_count_over_50"] * 1
)

df["low_trust_score_v6"] = (
    df["short_generic_review"] * 3
    + df["ultra_short_praise"] * 3
    + df["generic_without_detail"] * 2
    + df["menu_only_praise"] * 3
    + df["low_information_review"] * 2
    + df["weak_detail_review"] * 2
    + df["weak_experience_review"] * 2
    + df["weak_context_review"] * 2
    + df["has_trendy_revisit_phrase"] * 1.5
    + df["intensifier_over_3"] * 1
    + df["exclamation_over_3"] * 1
    + df["has_sns_event"] * 2
    + df["account_review_count_is_1"] * 3
)

df["trust_score_v6"] = df["high_trust_score_v6"] - df["low_trust_score_v6"]


# =========================
# 7. 이벤트 식당 vs 일반 식당 비교
# =========================
features_to_compare = [
    "review_length",
    "length_over_30",
    "length_over_60",

    "food_detail_count",
    "has_food_detail",
    "strong_food_detail_review",
    "weak_detail_review",

    "menu_count",
    "has_menu",
    "rich_menu_review",
    "menu_and_food_detail",
    "high_specificity_review",

    "experience_count",
    "has_experience",
    "strong_experience_review",
    "weak_experience_review",

    "context_count",
    "has_context",
    "weak_context_review",

    "has_any_number",
    "number_detail_count",
    "has_number_detail",

    "experience_rich_review",
    "rich_context_review",

    "generic_praise_count",
    "has_generic_praise",
    "short_generic_review",
    "ultra_short_praise",
    "generic_without_detail",
    "menu_only_praise",

    "trendy_revisit_count",
    "has_trendy_revisit_phrase",

    "intensifier_count",
    "intensifier_over_3",
    "exclamation_count",
    "exclamation_over_3",
    "sns_event_count",
    "has_sns_event",

    "mixed_sentiment",

    "account_review_count",
    "account_review_count_is_1",
    "account_review_count_over_10",
    "account_review_count_over_50",

    "low_information_review",

    "high_trust_score_v6",
    "low_trust_score_v6",
    "trust_score_v6"
]

comparison = df.groupby("group")[features_to_compare].mean().round(3)

print("\n[이벤트 식당 vs 일반 식당 Feature 평균 비교 - v6]")
print(comparison)


# =========================
# 8. 저장
# =========================
df.to_csv("reviews_with_features_v6.csv", index=False, encoding="utf-8-sig")
comparison.to_csv("feature_comparison_v6.csv", encoding="utf-8-sig")

print("\n저장 완료:")
print("- reviews_with_features_v6.csv")
print("- feature_comparison_v6.csv")