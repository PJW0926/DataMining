import re
import pandas as pd
from kiwipiepy import Kiwi

# =========================
# 1) 파일 설정
# =========================
INPUT_CSV = "naver_reviews_페르시안궁전.csv"           # 원본 CSV
OUTPUT_CSV = "reviews_scored.csv"   # 결과 CSV

# =========================
# 2) 데이터 불러오기
# =========================
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

if "리뷰 내용" not in df.columns:
    raise ValueError("'리뷰 내용' 컬럼이 없습니다. 컬럼명을 확인하세요.")

df["리뷰 내용"] = df["리뷰 내용"].fillna("").astype(str)

# =========================
# 3) 텍스트 정제
# =========================
def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)              # HTML 태그 제거
    text = re.sub(r"[^가-힣0-9\s]", " ", text)       # 한글/숫자/공백만 남김
    text = re.sub(r"\s+", " ", text)                # 연속 공백 정리
    return text.strip()

# =========================
# 4) 형태소 분석기
# =========================
kiwi = Kiwi()

# =========================
# 5) 불용어
# =========================
stopwords = {
    "하다", "되다", "있다", "없다", "이다", "아니다",
    "가다", "오다", "먹다", "보다", "주다",
    "좀", "진짜", "너무", "정말", "그냥", "약간",
    "그리고", "근데", "하지만", "또", "잘", "더"
}

# =========================
# 6) 감성사전
# =========================
sentiment_dict = {
    "맛있": 2,
    "좋": 2,
    "친절": 2,
    "깔끔": 2,
    "신선": 2,
    "추천": 2,
    "최고": 3,
    "훌륭": 3,
    "만족": 2,
    "재방문": 3,
    "가성비": 2,
    "푸짐": 2,
    "정갈": 2,
    "쾌적": 2,
    "빠르": 1,

    "맛없": -2,
    "별로": -2,
    "최악": -3,
    "불친절": -3,
    "짜": -1,
    "싱겁": -1,
    "느끼": -1,
    "비리": -2,
    "비싸": -2,
    "더럽": -3,
    "지저분": -3,
    "시끄럽": -1,
    "늦": -1,
    "아쉽": -1,
    "실망": -2,
    "불편": -2,
    "오래걸리": -1
}

# =========================
# 7) 토큰화
# Kiwi 품사 예: NNG, NNP, VA, VV 등
# 명사/형용사/동사 어간 위주 추출
# =========================
def tokenize(text):
    cleaned = clean_text(text)
    tokens = kiwi.tokenize(cleaned)

    words = []
    for token in tokens:
        word = token.form
        tag = token.tag

        if tag.startswith("N") or tag.startswith("V"):
            if len(word) >= 2 and word not in stopwords:
                words.append(word)

    return words

# =========================
# 8) 감성점수 계산
# =========================
def calculate_sentiment_score(tokens, sentiment_dict):
    score = 0
    matched_words = []

    for token in tokens:
        for key, val in sentiment_dict.items():
            if key in token:
                score += val
                matched_words.append(key)

    return score, matched_words

# =========================
# 9) 리뷰별 처리
# =========================
all_cleaned = []
all_tokens = []
all_scores = []
all_matched = []
all_labels = []

for review in df["리뷰 내용"]:
    cleaned = clean_text(review)
    tokens = tokenize(review)
    score, matched_words = calculate_sentiment_score(tokens, sentiment_dict)

    if score > 0:
        label = "긍정"
    elif score < 0:
        label = "부정"
    else:
        label = "중립"

    all_cleaned.append(cleaned)
    all_tokens.append(", ".join(tokens))
    all_scores.append(score)
    all_matched.append(", ".join(matched_words))
    all_labels.append(label)

df["정제 리뷰"] = all_cleaned
df["토큰"] = all_tokens
df["감성단어"] = all_matched
df["감성점수"] = all_scores
df["감성라벨"] = all_labels

# =========================
# 10) 저장
# =========================
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(df[["리뷰 내용", "토큰", "감성단어", "감성점수", "감성라벨"]].head(20))
print(f"\n저장 완료: {OUTPUT_CSV}")