import pandas as pd
import re
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# ==============================
# 1. 파일 불러오기
# ==============================

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "all_라벨링_2차.csv"

df = pd.read_csv(csv_path, encoding="utf-8-sig")

# 컬럼명 공백 제거
df.columns = df.columns.str.strip()

# 출력 생략 방지
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

# ==============================
# 2. 데이터 정리
# ==============================

# 리뷰 내용 정리
df["review_text"] = df["review_text"].fillna("").astype(str)

# 리뷰 글자 수 다시 계산
df["review_length"] = df["review_text"].str.len()

# visit_count 원본 확인
print("\n===== visit_count 원본 값 확인 =====")
print(df["visit_count"].value_counts(dropna=False).head(20).to_string())

# visit_count에서 숫자만 추출
if "visit_count" in df.columns:
    df["visit_count"] = (
        df["visit_count"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )

# 숫자형 컬럼 정리
numeric_cols = [
    "account_review_count",
    "review_length",
    "has_photo",
    "rating",
    "account_avg_rating",
    "manual_label"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 라벨 정리
df = df.dropna(subset=["manual_label"])
df["manual_label"] = df["manual_label"].astype(int)

# 사진 유무 정리
df["has_photo"] = pd.to_numeric(df["has_photo"], errors="coerce").fillna(0).astype(int)

# 비교할 컬럼
compare_cols = [
    "account_review_count",
    "visit_count",
    "review_length",
    "has_photo",
    "rating",
    "account_avg_rating"
]

existing_cols = [col for col in compare_cols if col in df.columns]

# ==============================
# 3. 전체 데이터 패턴 확인
# ==============================

print("\n===== 라벨 개수 =====")
print(df["manual_label"].value_counts().to_string())

print("\n===== 전체 평균 비교 =====")
summary = df.groupby("manual_label")[existing_cols].mean()
print(summary.to_string())

print("\n===== 전체 중앙값 비교 =====")
median_summary = df.groupby("manual_label")[existing_cols].median()
print(median_summary.to_string())

print("\n===== 플랫폼별 라벨 개수 =====")
print(pd.crosstab(df["platform"], df["manual_label"]).to_string())

print("\n===== 플랫폼별 평균 비교 =====")
platform_summary = df.groupby(["platform", "manual_label"])[existing_cols].mean()
print(platform_summary.to_string())

print("\n===== 플랫폼별 중앙값 비교 =====")
platform_median = df.groupby(["platform", "manual_label"])[existing_cols].median()
print(platform_median.to_string())

# ==============================
# 4. 규칙 기반 진정성 필터 함수
#    반드시 apply 하기 전에 먼저 정의되어야 함
# ==============================

def predict_authenticity(row):
    score = 0
    text = str(row["review_text"]).strip()

    # =========================
    # 단어 사전
    # =========================

    # 일반적 칭찬 (근거 없으면 suspicious)
    generic_words = [
        "맛있", "마싯", "맛잇",
        "좋아", "좋았", "좋습", "좋네",
        "친절",
        "최고",
        "추천", "강추",
        "굿", "굳",
        "존맛",
        "맛집",
        "짱",
        "대박",
        "훌륭",
        "완전 추천"
    ]

    # 일반적 비난
    generic_negative = [
        "별로였", "별로임", "별로네요", "별로예요", "개별로",
        "맛있지", "맛없", "맛이 없",
        "최악",
        "비추",
        "실망",
        "노맛",
        "그냥",
        "아쉬워", "아쉽",
        "다신 안",
        "재방문 안"
    ]

    # 구체적 부정 (오히려 real 신호)
    detailed_negative = [
        "짜", "싱겁", "맵", "달",
        "느끼", "비리", "차갑", "식었",
        "불친절", "늦", "오래 걸", "오래걸",
        "웨이팅", "기다렸", "대기",
        "비싸", "가격",
        "양이 적", "적음",
        "불편",
        "실수",
        "별로였"
    ]

    # 근거 표현
    reason_words = [
        "때문", "해서", "이라", "는데", "지만",
        "식감", "소스", "양", "가격",
        "친절", "응대", "서비스",
        "분위기", "인테리어",
        "웨이팅", "대기",
        "고소", "담백", "쫄깃", "바삭",
        "신선", "깔끔", "진한"
    ]

    # 한식 메뉴
    menu_words = [
        "비빔밥", "돌솥", "국밥", "순대국", "돼지국밥",
        "김치찌개", "된장찌개", "청국장", "찌개",
        "제육", "제육볶음", "불고기", "갈비",
        "삼겹살", "목살", "고기",
        "냉면", "막국수", "칼국수", "수제비",
        "보쌈", "족발", "수육",
        "닭갈비", "찜닭", "백숙", "삼계탕",
        "파전", "전", "계란말이", "잡채",
        "쌈밥", "쌈",
        "김밥", "떡볶이", "라면", "만두",
        "김치", "반찬", "밑반찬",
        "찌개", "전골", "육회", "해장국"
    ]

    # 카테고리별 구체 표현
    categories = {
        "atmosphere": [
            "분위기", "인테리어", "조용", "깔끔",
            "쾌적", "넓", "아늑"
        ],
        "service": [
            "친절", "응대", "직원", "서빙",
            "사장님", "알바"
        ],
        "waiting": [
            "웨이팅", "대기", "기다렸",
            "줄서", "예약"
        ],
        "taste": [
            "식감", "쫄깃", "바삭",
            "고소", "담백", "진한",
            "싱겁", "짜", "맵", "달",
            "부드럽", "촉촉"
        ]
    }

    # =========================
    # 1. 짧은 리뷰
    # =========================
    if row["review_length"] < 30:
        score += 1

    # =========================
    # 2. 계정 리뷰 수 적음
    # =========================
    if pd.notna(row["account_review_count"]) and row["account_review_count"] < 10:
        score += 1

    # =========================
    # 3. 메뉴 언급 -> real
    # =========================
    if any(word in text for word in menu_words):
        score -= 2

    # =========================
    # 4. 카테고리 구체성 -> real
    # =========================
    category_count = 0
    for word_list in categories.values():
        if any(word in text for word in word_list):
            category_count += 1

    score -= min(category_count, 3)

    # =========================
    # 5. 구체적 부정 -> real
    # =========================
    if any(word in text for word in detailed_negative):
        score -= 2

    # =========================
    # 6. generic praise
    # =========================
    generic_count = sum(word in text for word in generic_words)
    reason_count = sum(word in text for word in reason_words)

    has_generic = generic_count >= 2
    has_reason = reason_count >= 2

    if has_generic:
        score += 1

# 이유 없는 칭찬
    if has_generic and not has_reason:
        score += 2

    # =========================
    # 7. 이유 없는 비난
    # =========================
    has_negative = any(word in text for word in generic_negative)
    has_detailed_negative = any(word in text for word in detailed_negative)

    if has_negative and not has_detailed_negative:
        score += 2

    # =========================
    # 8. 플랫폼 보조 규칙
    # =========================
    if row["platform"] == "kakao":
        if pd.notna(row["rating"]) and row["rating"] == 5 and row["review_length"] < 80:
            score += 1

        if pd.notna(row["account_avg_rating"]) and row["account_avg_rating"] >= 4.3 and row["review_length"] < 80:
            score += 1

    if row["platform"] == "naver":
        if pd.notna(row["visit_count"]) and row["visit_count"] == 1 and row["review_length"] < 80:
            score += 1

    # 최종 판정
    pred = 0 if score > 0 else 1

    return pd.Series({
    "score": score,
    "pred_label": pred,
    "generic_count": generic_count,
    "reason_count": reason_count,
    "category_count": category_count,
    "has_negative": int(has_negative),
    "has_detailed_negative": int(has_detailed_negative),
    "menu_count": sum(word in text for word in menu_words)

})


# ==============================
# 5. train/test 분할
# ==============================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["manual_label"]
)

print("\n===== train/test 크기 확인 =====")
print("train:", train_df.shape)
print("test:", test_df.shape)

print("\n===== train 라벨 비율 =====")
print(train_df["manual_label"].value_counts(normalize=True).to_string())

print("\n===== test 라벨 비율 =====")
print(test_df["manual_label"].value_counts(normalize=True).to_string())

print("\n===== train 데이터 기준 평균 비교 =====")
print(train_df.groupby("manual_label")[existing_cols].mean().to_string())

print("\n===== train 데이터 기준 중앙값 비교 =====")
print(train_df.groupby("manual_label")[existing_cols].median().to_string())

# ==============================
# 6. train/test 예측
# ==============================

train_df = train_df.copy()
test_df = test_df.copy()

train_result = train_df.apply(predict_authenticity, axis=1)
test_result = test_df.apply(predict_authenticity, axis=1)

train_df = pd.concat([train_df, train_result], axis=1)
test_df = pd.concat([test_df, test_result], axis=1)

print("\n===== TRAIN score 분포 =====")
print(train_df["score"].value_counts().sort_index().to_string())

print("\n===== TEST score 분포 =====")
print(test_df["score"].value_counts().sort_index().to_string())

# ==============================
# 7. 성능 평가
# ==============================

print("\n==============================")
print("TRAIN 성능")
print("==============================")
print(confusion_matrix(train_df["manual_label"], train_df["pred_label"]))
print(classification_report(train_df["manual_label"], train_df["pred_label"], target_names=["fake", "real"]))
print("Accuracy:", accuracy_score(train_df["manual_label"], train_df["pred_label"]))
print("F1 Score:", f1_score(train_df["manual_label"], train_df["pred_label"]))

print("\n==============================")
print("TEST 성능")
print("==============================")
print(confusion_matrix(test_df["manual_label"], test_df["pred_label"]))
print(classification_report(test_df["manual_label"], test_df["pred_label"], target_names=["fake", "real"]))
print("Accuracy:", accuracy_score(test_df["manual_label"], test_df["pred_label"]))
print("F1 Score:", f1_score(test_df["manual_label"], test_df["pred_label"]))

# ==============================
# 8. test에서 틀린 것 저장
# ==============================

test_wrong = test_df[test_df["manual_label"] != test_df["pred_label"]]

wrong_cols = [
    "platform",
    "store_name",
    "account_id",
    "account_review_count",
    "visit_date",
    "visit_count",
    "review_text",
    "review_length",
    "has_photo",
    "rating",
    "account_avg_rating",
    "score",
    "manual_label",
    "pred_label"
]

existing_wrong_cols = [col for col in wrong_cols if col in test_wrong.columns]

test_wrong[existing_wrong_cols].to_csv(
    BASE_DIR / "test_wrong_cases.csv",
    index=False,
    encoding="utf-8-sig"
)


from sklearn.ensemble import RandomForestClassifier

feature_cols = [
    "account_review_count",
    "visit_count",
    "review_length",
    "has_photo",
    "rating",
    "account_avg_rating",
    "generic_count",
    "reason_count",
    "category_count",
    "menu_count",
    "has_negative",
    "has_detailed_negative"
]

X_train = train_df[feature_cols].fillna(0)
y_train = train_df["manual_label"]

X_test = test_df[feature_cols].fillna(0)
y_test = test_df["manual_label"]

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n===== Random Forest =====")
print(classification_report(y_test, rf_pred))

importance = pd.Series(
    rf.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\n===== Feature Importance =====")
print(importance.to_string())

print("\nTEST 오분류 사례 저장 완료: test_wrong_cases.csv")
print("TEST 오분류 개수:", len(test_wrong))
