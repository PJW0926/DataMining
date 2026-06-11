import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline


# ==============================
# 1. 설정
# ==============================

INPUT_PATH = r"C:\Users\reply\Desktop\2026-1\수업_데이터마이닝\Project\★정리★\4_감성 사전\final_high_trust_reviews.csv"

OUTPUT_PATH = r"C:\Users\reply\Desktop\2026-1\수업_데이터마이닝\Project\★정리★\4_감성 사전\final_high_trust_reviews_문맥감성분석.xlsx"
TEXT_COL = "review_text"

# 한국어 감성분석 모델
# 실행 시 자동 다운로드됨
MODEL_NAME = "WhitePeak/bert-base-cased-Korean-sentiment"


# ==============================
# 2. 파일 불러오기
# ==============================

def load_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError("csv 또는 xlsx 파일만 지원합니다.")


df = load_file(INPUT_PATH)

if TEXT_COL not in df.columns:
    raise ValueError(f"'{TEXT_COL}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)


# ==============================
# 3. 모델 불러오기
# ==============================

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=-1  # CPU 사용. GPU 있으면 0으로 변경 가능
)


# ==============================
# 4. 리뷰 감성분석
# ==============================

sentiment_labels = []
sentiment_scores = []

for text in tqdm(df[TEXT_COL], desc="감성분석 진행 중"):
    text = text.strip()

    if text == "":
        sentiment_labels.append("EMPTY")
        sentiment_scores.append(0.0)
        continue

    # BERT 계열 모델은 너무 긴 텍스트를 자르면 안정적임
    text = text[:512]

    try:
        result = classifier(text)[0]
        sentiment_labels.append(result["label"])
        sentiment_scores.append(result["score"])

    except Exception as e:
        sentiment_labels.append("ERROR")
        sentiment_scores.append(0.0)


df["kobert_sentiment_label"] = sentiment_labels
df["kobert_sentiment_score"] = sentiment_scores


# ==============================
# 5. label 정리
# ==============================

def convert_sentiment(label):
    label = str(label).lower()

    if "positive" in label or "pos" in label or label == "1":
        return "positive"
    elif "negative" in label or "neg" in label or label == "0":
        return "negative"
    else:
        return "unknown"


df["kobert_sentiment_clean"] = df["kobert_sentiment_label"].apply(convert_sentiment)


# ==============================
# 6. 결과 저장
# ==============================

df.to_excel(OUTPUT_PATH, index=False)

print("완료!")
print(f"저장 위치: {OUTPUT_PATH}")
print()
print(df["kobert_sentiment_clean"].value_counts())