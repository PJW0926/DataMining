import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# ==============================
# 1. 파일 불러오기
# ==============================

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "all_라벨링_1차.csv"

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
    fake_score = 0
    text = str(row["review_text"])

    # 1. 리뷰 길이 기준
    if row["review_length"] < 30:
        fake_score += 2
    elif row["review_length"] < 100:
        fake_score += 1

    # 2. 계정 리뷰 수가 너무 적은 경우
    if pd.notna(row["account_review_count"]) and row["account_review_count"] < 10:
        fake_score += 1

    # 3. 짧고 일반적인 칭찬어만 있는 경우
    generic_words = [
        "맛있어요", "맛있습니다", "좋아요", "좋습니다", "친절해요",
        "친절합니다", "최고", "추천", "굿", "존맛", "맛집"
    ]

    if row["review_length"] < 60 and any(word in text for word in generic_words):
        fake_score += 1

    # 4. 카카오 전용: 별점 5점 + 짧은 리뷰 / 별점 5점 + 계정 리뷰 수 1개 += 2
    if row["platform"] == "kakao":
        if pd.notna(row["rating"]) and row["rating"] == 5 and row["review_length"] < 80:
            fake_score += 1

        if pd.notna(row["account_avg_rating"]) and row["account_avg_rating"] >= 4.3 and row["review_length"] < 80:
            fake_score += 1

    # 5. 네이버 전용: 1번째 방문 + 짧은 리뷰 / 1번째 방문 + 계정 리뷰 수 1개 + 칭찬 3개 이상 += 2
    if row["platform"] == "naver":
        if pd.notna(row["visit_count"]) and row["visit_count"] == 1 and row["review_length"] < 80:
            fake_score += 1

    # fake_score가 2점 이상이면 가짜/의심, 아니면 진짜
    return 0 if fake_score >= 2 else 1

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

train_df["pred_label"] = train_df.apply(predict_authenticity, axis=1)
test_df["pred_label"] = test_df.apply(predict_authenticity, axis=1)

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
    "manual_label",
    "pred_label"
]

existing_wrong_cols = [col for col in wrong_cols if col in test_wrong.columns]

test_wrong[existing_wrong_cols].to_csv(
    BASE_DIR / "test_wrong_cases.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTEST 오분류 사례 저장 완료: test_wrong_cases.csv")
print("TEST 오분류 개수:", len(test_wrong))