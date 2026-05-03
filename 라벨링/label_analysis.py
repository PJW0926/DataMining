import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

print("현재 py 파일 폴더:", BASE_DIR)
print("현재 py 파일 폴더 안 파일들:")
for file in BASE_DIR.iterdir():
    print(file.name)

csv_path = BASE_DIR / "all_마포_라벨링.csv"

print("\nCSV 찾는 위치:", csv_path)
print("파일 존재 여부:", csv_path.exists())

df = pd.read_csv(csv_path, encoding="utf-8-sig")

print("===== 컬럼명 확인 =====")
print(df.columns.tolist())

print("\n===== 데이터 크기 확인 =====")
print(df.shape)

print("\n===== 앞부분 확인 =====")
print(df.head())

print("\n===== 라벨 개수 확인 =====")
print(df["manual_label"].value_counts(dropna=False))

print("\n===== 사진 유무 값 확인 =====")
print(df["has_photo"].value_counts(dropna=False))

# ==============================
# 1. 데이터 정리
# ==============================

# 컬럼명 앞뒤 공백 제거
df.columns = df.columns.str.strip()

# 리뷰 내용 빈칸 처리
df["review_text"] = df["review_text"].fillna("").astype(str)

# 리뷰 글자 수 다시 계산
df["review_length"] = df["review_text"].str.len()

# 숫자로 바꿔야 하는 컬럼들 정리
numeric_cols = [
    "account_review_count",
    "visit_count",
    "review_length",
    "has_photo",
    "rating",
    "account_avg_rating",
    "manual_label"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 라벨 결측치 제거
df = df.dropna(subset=["manual_label"])
df["manual_label"] = df["manual_label"].astype(int)

# 사진 유무 결측치 처리
df["has_photo"] = df["has_photo"].fillna(0).astype(int)

# 정리된 파일 저장
df.to_csv(BASE_DIR / "labeled_971_clean.csv", index=False, encoding="utf-8-sig")

print("\n===== 정리 완료 =====")
print(df.info())
print("\n===== 라벨 개수 =====")
print(df["manual_label"].value_counts())

# ==============================
# 2. 진짜/가짜 평균 비교
# ==============================

compare_cols = [
    "account_review_count",
    "visit_count",
    "review_length",
    "has_photo",
    "rating",
    "account_avg_rating"
]

existing_cols = [col for col in compare_cols if col in df.columns]

summary = df.groupby("manual_label")[existing_cols].mean()

print("\n===== 진짜/가짜 평균 비교 =====")
print(summary)