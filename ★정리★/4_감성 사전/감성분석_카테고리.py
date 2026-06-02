import pandas as pd
import re
from konlpy.tag import Okt
from pathlib import Path

# =========================
# 현재 .py 파일이 있는 폴더를 기준 경로로 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# 1. 파일 불러오기
# =========================================================

input_file = BASE_DIR / "final_high_trust_reviews_pos.csv"
output_file = BASE_DIR / "final_high_trust_reviews_category_sentiment.csv"

print("현재 코드 위치:", BASE_DIR)
print("읽을 파일:", input_file)

if not input_file.exists():
    raise FileNotFoundError(
        f"입력 파일이 없습니다: {input_file}\n"
        "먼저 감성분석_토큰화.py를 실행해서 final_high_trust_reviews_pos.csv를 생성하세요."
    )

df = pd.read_csv(input_file, encoding="utf-8-sig")

# tokens 열 확인
if "tokens" not in df.columns:
    raise ValueError("tokens 열이 없습니다. 먼저 Okt 토큰화 파일을 생성해야 합니다.")

df["tokens"] = df["tokens"].fillna("").astype(str)

# =========================================================
# 2. 카테고리별 감성사전
# =========================================================
# 점수 기준:
# 긍정 단어: +1 또는 +2
# 부정 단어: -1 또는 -2
#
# 필요하면 프로젝트 데이터 보면서 단어를 계속 추가하면 됨.

CATEGORY_SENTIMENT_DICT = {
    # -------------------------
    # 음식 - 긍정
    # -------------------------
    "food_pos": {
        "맛있다": 2,
        "맛나다": 2,
        "맛좋다": 2,
        "괜찮다": 1,
        "좋다": 1,
        "훌륭하다": 2,
        "고소하다": 1,
        "담백하다": 1,
        "깔끔하다": 1,
        "신선하다": 2,
        "부드럽다": 1,
        "쫄깃하다": 1,
        "바삭하다": 1,
        "촉촉하다": 1,
        "진하다": 1,
        "풍부하다": 1,
        "알차다": 1,
        "푸짐하다": 1,
        "든든하다": 1,
        "개운하다": 1,
        "시원하다": 1,
        "달달하다": 1,
        "달콤하다": 1,
        "매콤하다": 1,
        "칼칼하다": 1,
        "구수하다": 1,
        "육즙": 1,
        "감칠맛": 2,
        "존맛": 2,
        "JMT": 2,
        "최고": 2,
        "추천": 1,
    },

    # -------------------------
    # 음식 - 부정
    # -------------------------
    "food_neg": {
        "맛없다": -2,
        "별로": -2,
        "최악": -2,
        "실망": -2,
        "아쉽다": -1,
        "아쉬움": -1,
        "싱겁다": -1,
        "짜다": -1,
        "달다": -1,
        "맵다": -1,
        "느끼하다": -1,
        "퍽퍽하다": -1,
        "질기다": -1,
        "딱딱하다": -1,
        "차갑다": -1,
        "식다": -1,
        "비리다": -2,
        "잡내": -2,
        "냄새": -1,
        "눅눅하다": -1,
        "기름지다": -1,
        "물리다": -1,
        "부실하다": -1,
        "애매하다": -1,
        "평범하다": -1,
    },

    # -------------------------
    # 가격 - 긍정
    # -------------------------
    "price_pos": {
        "저렴하다": 2,
        "싸다": 2,
        "합리적": 2,
        "합리": 1,
        "가성비": 2,
        "혜자": 2,
        "괜찮다": 1,
        "만족": 1,
        "푸짐하다": 1,
        "양많다": 2,
        "양": 1,
        "넉넉하다": 1,
        "적당하다": 1,
        "무난하다": 1,
    },

    # -------------------------
    # 가격 - 부정
    # -------------------------
    "price_neg": {
        "비싸다": -2,
        "비쌈": -2,
        "가격": -1,
        "부담": -1,
        "창렬": -2,
        "아깝다": -2,
        "아쉬움": -1,
        "아쉽다": -1,
        "양적다": -2,
        "적다": -1,
        "부족하다": -1,
        "비추": -2,
    },

    # -------------------------
    # 서비스 - 긍정
    # -------------------------
    "service_pos": {
        "친절하다": 2,
        "친절": 2,
        "상냥하다": 2,
        "세심하다": 2,
        "빠르다": 1,
        "신속하다": 1,
        "응대": 1,
        "서비스": 1,
        "직원": 1,
        "사장": 1,
        "설명": 1,
        "배려": 2,
        "감사하다": 1,
        "좋다": 1,
        "만족": 1,
    },

    # -------------------------
    # 서비스 - 부정
    # -------------------------
    "service_neg": {
        "불친절하다": -2,
        "불친절": -2,
        "느리다": -1,
        "늦다": -1,
        "기다리다": -1,
        "대기": -1,
        "웨이팅": -1,
        "응대": -1,
        "서비스": -1,
        "직원": -1,
        "짜증": -2,
        "불쾌하다": -2,
        "무시": -2,
        "실수": -1,
        "누락": -1,
        "엉망": -2,
    },

    # -------------------------
    # 분위기 - 긍정
    # -------------------------
    "atmosphere_pos": {
        "깔끔하다": 2,
        "깨끗하다": 2,
        "청결하다": 2,
        "예쁘다": 1,
        "이쁘다": 1,
        "아늑하다": 1,
        "조용하다": 1,
        "편하다": 1,
        "편안하다": 1,
        "분위기": 1,
        "인테리어": 1,
        "감성": 1,
        "쾌적하다": 2,
        "넓다": 1,
        "뷰": 1,
        "사진": 1,
        "데이트": 1,
        "모임": 1,
        "좋다": 1,
    },

    # -------------------------
    # 분위기 - 부정
    # -------------------------
    "atmosphere_neg": {
        "시끄럽다": -2,
        "좁다": -1,
        "답답하다": -1,
        "불편하다": -1,
        "더럽다": -2,
        "지저분하다": -2,
        "냄새": -1,
        "복잡하다": -1,
        "어수선하다": -1,
        "덥다": -1,
        "춥다": -1,
        "불쾌하다": -2,
        "별로": -2,
    }
}

# =========================================================
# 3. 감성분석 함수
# =========================================================

def analyze_category_sentiment(tokens_text):
    tokens = str(tokens_text).split()

    result = {}

    for category, word_score_dict in CATEGORY_SENTIMENT_DICT.items():
        matched = []
        score = 0

        for token in tokens:
            if token in word_score_dict:
                token_score = word_score_dict[token]
                score += token_score
                matched.append(f"{token}:{token_score}")

        result[f"{category}_score"] = score
        result[f"{category}_count"] = len(matched)
        result[f"{category}_words"] = " ".join(matched)

    return pd.Series(result)

# =========================================================
# 4. 카테고리별 감성분석 적용
# =========================================================

category_result = df["tokens"].apply(analyze_category_sentiment)
df = pd.concat([df, category_result], axis=1)

# =========================================================
# 5. 보기 쉬운 통합 점수 생성
# =========================================================

df["food_score"] = df["food_pos_score"] + df["food_neg_score"]
df["price_score"] = df["price_pos_score"] + df["price_neg_score"]
df["service_score"] = df["service_pos_score"] + df["service_neg_score"]
df["atmosphere_score"] = df["atmosphere_pos_score"] + df["atmosphere_neg_score"]

# =========================================================
# 6. 카테고리별 라벨 생성
# =========================================================

def score_to_label(score):
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

df["food_label"] = df["food_score"].apply(score_to_label)
df["price_label"] = df["price_score"].apply(score_to_label)
df["service_label"] = df["service_score"].apply(score_to_label)
df["atmosphere_label"] = df["atmosphere_score"].apply(score_to_label)

# 전체 카테고리 감성점수
df["category_total_score"] = (
    df["food_score"]
    + df["price_score"]
    + df["service_score"]
    + df["atmosphere_score"]
)

df["category_total_label"] = df["category_total_score"].apply(score_to_label)

# =========================================================
# 7. 저장
# =========================================================

df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("완료:", output_file)
print("데이터 크기:", df.shape)

print(
    df[
        [
            "review_text",
            "tokens",
            "food_score",
            "price_score",
            "service_score",
            "atmosphere_score",
            "category_total_score",
            "category_total_label"
        ]
    ].head()
)