import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ==============================
# 0. 한글 폰트 설정
# ==============================
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ==============================
# 1. 경로 설정
# ==============================
BASE_DIR = Path(__file__).resolve().parent

file4 = BASE_DIR / "all_wrong_cases_4차.csv"
file5 = BASE_DIR / "all_wrong_cases_5차.csv"

df4 = pd.read_csv(file4, encoding="utf-8-sig")
df5 = pd.read_csv(file5, encoding="utf-8-sig")

print("4차 wrong case:", df4.shape)
print("5차 wrong case:", df5.shape)

# ==============================
# 2. 주요 feature 목록
# ==============================
# 그래프에 넣고 싶은 feature만 여기에 남기면 됨
feature_cols = [
    "has_generic_praise",
    "generic_without_detail",
    "has_number_detail",
    "has_menu",
    "has_reason",
    "has_quality_detail",
    "has_sensory_detail",
    "has_experience_context",
    "has_temporal_context",
    "has_mixed_opinion",
    "has_detailed_negative",
    "short_review",
    "low_information_review",
    "specific_review",
    "positive_narrative",
    "mixed_opinion_review",
    "temporal_experience_review",
    "short_but_specific_food_review",
    "detailed_negative_narrative",
    "first_visit",
    "revisit",
]

# ==============================
# 3. 4차에 없는 feature는 review_text에서 재계산
# ==============================
generic_words = [
    "맛있", "마싯", "맛잇", "좋아", "좋았", "좋습", "좋네", "좋음",
    "친절", "최고", "추천", "강추", "굿", "굳", "존맛", "맛집", "짱", "대박"
]

reason_words = [
    "때문", "해서", "이라", "는데", "지만", "식감", "소스", "양", "가격",
    "친절", "응대", "서비스", "분위기", "웨이팅", "대기", "고소", "담백",
    "쫄깃", "바삭", "신선", "깔끔", "진한", "잡내", "국물", "재료"
]

menu_words = [
    "곱창", "막창", "대창", "전골", "볶음밥", "감자전", "콘치즈",
    "국밥", "찌개", "제육", "불고기", "갈비", "삼겹살", "고기",
    "냉면", "칼국수", "보쌈", "족발", "김밥", "떡볶이", "라면",
    "만두", "반찬", "육회", "해장국", "파스타", "피자", "스테이크",
    "감자탕", "수제비", "사리", "국수", "막걸리", "치즈", "면",
    "야채", "밥", "국물", "간장", "김치", "두부"
]

quality_detail_words = [
    "누룽지", "참기름", "찍어", "비벼", "곁들", "간이 딱",
    "잡내 안", "부드러", "신선", "싱싱", "고소", "시원", "새콤",
    "직접", "덜 자극", "불향", "리필", "육즙", "감칠맛", "식감"
]

sensory_detail_words = [
    "매콤", "얼큰", "진하", "진한", "크리미", "고소", "담백",
    "자극적", "부드럽", "질기", "쫄깃", "바삭", "촉촉",
    "잡내", "비리", "육즙", "감칠맛", "칼칼", "따뜻", "뜨끈",
    "푸짐", "살살녹", "새콤", "식감", "밸런스"
]

experience_context_words = [
    "방문", "주문", "나왔", "기다리", "주차", "직원", "아주머니",
    "먹고 싶", "점심", "저녁", "친구", "가족", "바로", "서비스",
    "포장", "예약", "자리", "앉", "먹었", "먹고", "다녀왔",
    "시간", "주말", "평일", "같이", "매장", "입장", "가봤", "와봤"
]

temporal_context_words = [
    "처음", "예전", "전에", "지난번", "오랜만", "몇 년",
    "다시", "또", "재방문", "종종", "다음에도", "다음에", "또 오", "또 방문"
]

mixed_opinion_words = [
    "다만", "아쉽", "아쉬", "근데", "그런데", "하지만", "그러나",
    "별관", "사람 많", "그닥", "비싸", "가격", "시끄", "기다려서"
]

detailed_negative_words = [
    "짜", "싱겁", "맵", "달", "느끼", "비리", "차갑", "식었",
    "불친절", "늦", "오래 걸", "웨이팅", "기다렸", "대기", "비싸",
    "양이 적", "불편", "실수", "이물질", "응대", "위생", "최악", "아쉬"
]

def has_any(text, words):
    text = str(text)
    return int(any(word in text for word in words))

def count_number_patterns(text):
    text = str(text)
    patterns = [
        r"\d+\s*분", r"\d+\s*인분", r"\d+\s*원", r"\d+\s*명",
        r"\d+\s*시", r"\d+\s*차", r"\d+\s*번", r"\d+\s*개",
        r"\d+\s*점", r"\d+\s*년", r"\d+\s*-\s*\d+\s*년"
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)

def add_missing_features(df):
    df = df.copy()

    if "review_text" not in df.columns:
        raise ValueError("review_text 컬럼이 필요합니다.")

    df["review_text"] = df["review_text"].fillna("").astype(str)

    if "review_length" not in df.columns:
        df["review_length"] = df["review_text"].str.len()

    # 없는 feature만 보충 생성
    if "has_generic_praise" not in df.columns:
        df["has_generic_praise"] = df["review_text"].apply(lambda x: has_any(x, generic_words))

    if "has_menu" not in df.columns:
        df["has_menu"] = df["review_text"].apply(lambda x: has_any(x, menu_words))

    if "has_reason" not in df.columns:
        df["has_reason"] = df["review_text"].apply(lambda x: has_any(x, reason_words))

    if "has_number_detail" not in df.columns:
        df["has_number_detail"] = df["review_text"].apply(lambda x: int(count_number_patterns(x) >= 1))

    if "has_quality_detail" not in df.columns:
        df["has_quality_detail"] = df["review_text"].apply(lambda x: has_any(x, quality_detail_words))

    if "has_sensory_detail" not in df.columns:
        df["has_sensory_detail"] = df["review_text"].apply(lambda x: has_any(x, sensory_detail_words))

    if "has_experience_context" not in df.columns:
        df["has_experience_context"] = df["review_text"].apply(lambda x: has_any(x, experience_context_words))

    if "has_temporal_context" not in df.columns:
        df["has_temporal_context"] = df["review_text"].apply(lambda x: has_any(x, temporal_context_words))

    if "has_mixed_opinion" not in df.columns:
        df["has_mixed_opinion"] = df["review_text"].apply(lambda x: has_any(x, mixed_opinion_words))

    if "has_detailed_negative" not in df.columns:
        df["has_detailed_negative"] = df["review_text"].apply(lambda x: has_any(x, detailed_negative_words))

    if "short_review" not in df.columns:
        df["short_review"] = (df["review_length"] < 30).astype(int)

    if "low_information_review" not in df.columns:
        df["low_information_review"] = (
            (df["short_review"] == 1)
            & (df["has_menu"] == 0)
            & (df["has_number_detail"] == 0)
            & (df["has_detailed_negative"] == 0)
        ).astype(int)

    if "specific_review" not in df.columns:
        df["specific_review"] = (
            (df["has_menu"] == 1)
            | (df["has_number_detail"] == 1)
            | (df["has_quality_detail"] == 1)
            | (df["has_sensory_detail"] == 1)
            | (df["has_experience_context"] == 1)
            | (df["has_temporal_context"] == 1)
            | (df["has_mixed_opinion"] == 1)
            | (df["has_detailed_negative"] == 1)
        ).astype(int)

    if "generic_without_detail" not in df.columns:
        df["generic_without_detail"] = (
            (df["has_generic_praise"] == 1)
            & (df["specific_review"] == 0)
            & (df["has_menu"] == 0)
        ).astype(int)

    # 아래 세 개는 5차 전용 복합 feature라 4차에서는 재계산
    if "positive_narrative" not in df.columns:
        df["positive_narrative"] = (
            (df["has_generic_praise"] == 1)
            & (
                (df["has_sensory_detail"] == 1)
                | (df["has_quality_detail"] == 1)
                | (df["has_menu"] == 1)
            )
            & (
                (df["has_experience_context"] == 1)
                | (df["has_temporal_context"] == 1)
                | (df["review_length"] >= 70)
            )
        ).astype(int)

    if "mixed_opinion_review" not in df.columns:
        df["mixed_opinion_review"] = (
            (
                (df["has_mixed_opinion"] == 1)
                | (df["has_detailed_negative"] == 1)
            )
            & (
                (df["has_generic_praise"] == 1)
                | (df["has_sensory_detail"] == 1)
                | (df["has_quality_detail"] == 1)
                | (df["has_menu"] == 1)
            )
            & (df["review_length"] >= 40)
        ).astype(int)

    if "temporal_experience_review" not in df.columns:
        df["temporal_experience_review"] = (
            (df["has_temporal_context"] == 1)
            & (
                (df["review_length"] >= 40)
                | (df["has_menu"] == 1)
                | (df["has_experience_context"] == 1)
            )
        ).astype(int)

    if "short_but_specific_food_review" not in df.columns:
        df["short_but_specific_food_review"] = (
            (df["review_length"] < 60)
            & (
                (df["has_menu"] == 1)
                | (df["has_sensory_detail"] == 1)
                | (df["has_quality_detail"] == 1)
            )
            & (df["generic_without_detail"] == 0)
        ).astype(int)

    if "detailed_negative_narrative" not in df.columns:
        df["detailed_negative_narrative"] = (
            (df["has_detailed_negative"] == 1)
            & (
                (df["review_length"] >= 50)
                | (df["has_experience_context"] == 1)
                | (df["has_temporal_context"] == 1)
            )
        ).astype(int)

    return df

df4 = add_missing_features(df4)
df5 = add_missing_features(df5)

# ==============================
# 4. feature 출현 개수 계산
# ==============================
available_features = [col for col in feature_cols if col in df4.columns and col in df5.columns]

summary_rows = []

for feature in available_features:
    count4 = int((pd.to_numeric(df4[feature], errors="coerce").fillna(0) > 0).sum())
    count5 = int((pd.to_numeric(df5[feature], errors="coerce").fillna(0) > 0).sum())

    rate4 = round(count4 / len(df4) * 100, 1)
    rate5 = round(count5 / len(df5) * 100, 1)

    summary_rows.append({
        "feature": feature,
        "4차_출현건수": count4,
        "5차_출현건수": count5,
        "4차_출현비율": rate4,
        "5차_출현비율": rate5,
        "차이_5차-4차": count5 - count4
    })

feature_summary = pd.DataFrame(summary_rows)

# 보기 좋게 정렬: 4차+5차 합계가 큰 순서
feature_summary["합계"] = feature_summary["4차_출현건수"] + feature_summary["5차_출현건수"]
feature_summary = feature_summary.sort_values("합계", ascending=False).drop(columns="합계")

print("\n===== 4차 vs 5차 wrong case 내 주요 feature 출현 개수 =====")
print(feature_summary.to_string(index=False))

# 저장
feature_summary.to_csv(
    BASE_DIR / "오분류케이스_주요feature_출현개수_4차_5차.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==============================
# 5. 그래프 1: 출현 건수 비교
# ==============================
plot_df = feature_summary.set_index("feature")[["4차_출현건수", "5차_출현건수"]]

# feature가 너무 많으면 상위 15개만
top_n = 15
plot_df = plot_df.head(top_n)

ax = plot_df.plot(kind="bar", figsize=(15, 7), width=0.8)

plt.title("4차와 5차 오분류 케이스 내 주요 Feature 출현 건수 비교")
plt.xlabel("주요 Feature")
plt.ylabel("출현 건수")
plt.xticks(rotation=35, ha="right")
plt.legend(["4차", "5차"])
plt.tight_layout()

for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)

plt.savefig(
    BASE_DIR / "오분류케이스_주요feature_출현건수_비교.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# ==============================
# 6. 그래프 2: 출현 비율 비교
# ==============================
# 4차와 5차 wrong case 총수가 다르므로 비율 그래프도 같이 만드는 게 더 정확함
rate_df = feature_summary.set_index("feature")[["4차_출현비율", "5차_출현비율"]]
rate_df = rate_df.head(top_n)

ax = rate_df.plot(kind="bar", figsize=(15, 7), width=0.8)

plt.title("4차와 5차 오분류 케이스 내 주요 Feature 출현 비율 비교")
plt.xlabel("주요 Feature")
plt.ylabel("출현 비율(%)")
plt.xticks(rotation=35, ha="right")
plt.legend(["4차", "5차"])
plt.tight_layout()

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", padding=3)

plt.savefig(
    BASE_DIR / "오분류케이스_주요feature_출현비율_비교.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\n저장 완료:")
print("- 오분류케이스_주요feature_출현개수_4차_5차.csv")
print("- 오분류케이스_주요feature_출현건수_비교.png")
print("- 오분류케이스_주요feature_출현비율_비교.png")