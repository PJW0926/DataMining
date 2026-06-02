import pandas as pd
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# v18 trust-level system
# 핵심 방향:
# - v16/v17의 trust-level 구조 유지
# - 식당 단위 점수 계산은 제외
# - 리뷰별 trust_weight만 산출
# - positive experiential narrative 보정 추가
# - temporal/revisit context 보정 추가
# - mixed opinion review 보정 추가
# - 구체적인 긍정 리뷰가 fake(0)로 떨어지는 문제 완화
# =========================================================


# ==============================
# 0. 설정
# ==============================

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "all_라벨링_4차.csv"

OUTPUT_DIR = BASE_DIR / "filter_result_v19_trust_level_eval"
OUTPUT_DIR.mkdir(exist_ok=True)

# binary 평가용 threshold
BINARY_TRUST_THRESHOLD = 5.7

# 3단계 trust level 기준
LOW_TRUST_THRESHOLD = 0.5
HIGH_TRUST_THRESHOLD = 3.5

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ==============================
# 1. 파일 불러오기
# ==============================

if csv_path.exists():
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
else:
    xlsx_path = BASE_DIR / "all_라벨링_4차.xlsx"
    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError("all_라벨링_4차.csv 또는 all_라벨링_4차.xlsx 파일이 같은 폴더에 없습니다.")

df.columns = df.columns.str.strip()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)


# ==============================
# 2. 데이터 정리
# ==============================

if "review_text" not in df.columns:
    raise ValueError("review_text 컬럼이 필요합니다.")

if "manual_label" not in df.columns:
    raise ValueError("manual_label 컬럼이 필요합니다.")

df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
df["review_length"] = df["review_text"].str.len()

if "platform" not in df.columns:
    df["platform"] = "unknown"

if "store_name" not in df.columns:
    df["store_name"] = "unknown"

if "visit_count" not in df.columns:
    df["visit_count"] = pd.NA

df["visit_count"] = (
    df["visit_count"]
    .astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

numeric_cols = [
    "account_review_count",
    "review_length",
    "has_photo",
    "rating",
    "account_avg_rating",
    "manual_label",
    "visit_count",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = pd.NA

df = df.dropna(subset=["manual_label"]).copy()
df["manual_label"] = df["manual_label"].astype(int)

df["has_photo"] = pd.to_numeric(df["has_photo"], errors="coerce").fillna(0).astype(int)

before_empty = len(df)
df = df[df["review_text"] != ""].copy()
after_empty = len(df)

print("\n===== 빈 리뷰 제거 =====")
print(f"{before_empty}개 -> {after_empty}개")


# ==============================
# 3. 단어 사전
# ==============================

generic_words = [
    "맛있", "마싯", "맛잇",
    "좋아", "좋았", "좋습", "좋네", "좋음",
    "친절", "최고", "추천", "강추",
    "굿", "굳", "존맛", "맛집", "짱", "대박",
]

generic_negative = [
    "별로", "맛없", "최악", "비추", "실망", "노맛",
    "아쉬워", "아쉽", "다신 안", "재방문 안",
]

detailed_negative = [
    "짜", "싱겁", "맵", "달", "느끼", "비리",
    "차갑", "식었", "불친절", "늦", "오래 걸", "오래걸",
    "웨이팅", "기다렸", "대기", "비싸", "가격",
    "양이 적", "적음", "불편", "실수", "이물질", "응대",
    "맛 없어", "맛없", "낭비", "대체재", "현금영수증",
    "위생", "불청결", "돼지털", "설거지", "재사용",
    "엉망", "안옵니다", "쫓겨", "최악",
    "아쉬", "그닥", "연하고", "적은", "조금", "좁", "시끄",
    "귀 아픔", "시장통", "기대", "평범", "특별", "기다려서",
]

reason_words = [
    "때문", "해서", "이라", "는데", "지만", "그런가",
    "식감", "소스", "양", "가격",
    "친절", "응대", "서비스",
    "분위기", "인테리어",
    "웨이팅", "대기",
    "고소", "담백", "쫄깃", "바삭",
    "신선", "깔끔", "진한",
    "잡내", "국물", "재료", "조리",
    "밸런스", "감칠맛", "신맛", "간장", "독특", "색다른",
]

menu_words = [
    "곱창", "막창", "대창", "전골", "볶음밥", "감자전", "콘치즈",
    "비빔밥", "국밥", "찌개", "제육", "불고기", "갈비",
    "삼겹살", "고기", "냉면", "칼국수", "보쌈", "족발",
    "김밥", "떡볶이", "라면", "만두", "반찬",
    "육회", "해장국", "파스타", "피자", "스테이크",
    "감자탕", "수제비", "사리", "국수", "쌈밥", "막걸리",
    "오뎅탕", "보리된장국수", "묵은지", "닭볶음탕",
    "닭갈비", "치즈", "면", "문어", "마제소바", "양념",
    "야채", "밥", "국물", "간장", "김치", "두부",
]

categories = {
    "atmosphere": ["분위기", "인테리어", "조용", "깔끔", "쾌적", "넓", "아늑", "시끄럽", "소음", "캠핑", "시장통"],
    "service": ["친절", "응대", "직원", "서빙", "사장님", "알바", "서비스", "아주머니"],
    "waiting": ["웨이팅", "대기", "기다렸", "줄서", "예약", "줄없이", "바로 입장"],
    "taste": ["식감", "쫄깃", "바삭", "고소", "담백", "진한", "싱겁", "짜", "맵", "달", "부드럽", "촉촉", "잡내", "국물", "자극적", "감칠맛", "신맛", "밸런스"],
    "context": ["점심", "저녁", "주말", "평일", "회식", "데이트", "친구", "가족", "혼밥", "퇴근", "2차", "방문", "주문", "처음", "예전"],
}

event_words = ["이벤트", "리뷰이벤트", "인스타", "릴스", "sns", "SNS", "협찬", "체험단"]

intensifier_words = ["진짜", "너무", "완전", "엄청", "존맛", "대박", "짱", "개맛", "핵맛", "미쳤"]

quality_detail_words = [
    "누룽지", "참기름", "찍어", "비벼", "곁들",
    "간이 딱", "잡내 안", "잡내가 안",
    "부드러", "신선", "싱싱", "고소", "시원", "새콤",
    "살코기", "반죽", "직접", "덜 자극", "자극적이지",
    "불향", "리필", "수육", "육즙", "크리미", "간장베이스",
    "밸런스", "감칠맛", "신맛", "간장", "독특", "색다른",
    "식감", "살아있", "마제소바", "부드러워", "편하게",
]

sensory_detail_words = [
    "매콤", "얼큰", "진하", "진한", "크리미", "고소", "담백",
    "자극적이지", "자극적", "부드럽", "질기", "쫄깃", "바삭", "촉촉",
    "잡내", "비리", "육즙", "감칠맛", "칼칼", "따뜻", "뜨끈",
    "배부르게", "양도많", "양이 많", "계속 숟가락", "먹기 좋아",
    "가볍게", "무겁지", "푸짐", "살코기", "특별함", "불향",
    "상성", "살살녹", "셔요", "새콤", "진하고",
    "신맛", "연하고", "부드러워", "식감", "밸런스", "간장",
    "독특", "색다른", "양념", "맛나요",
]

popularity_waiting_words = ["웨이팅", "대기", "줄", "기다렸", "손님 많", "사람 많", "회전율", "별관", "유명"]

experience_context_words = [
    "방문", "주문", "나왔", "기다리", "주차", "본관", "별관",
    "직원", "아주머니", "현금영수증", "반죽", "넣어주",
    "먹고 싶", "점심", "저녁", "친구", "가족", "사리",
    "바로", "편해", "불편", "낭비", "대체재", "서비스",
    "포장", "예약", "자리", "앉", "먹었", "먹고", "다녀왔",
    "시간", "주말", "평일", "일요일", "같이", "옆에", "타워주차장",
    "처음", "지난번", "전에", "오랜만", "찾아온", "멀리",
    "알바", "설명", "매장", "입장", "다녀", "가봤",
    "와봤", "왔는데", "들어가", "식사", "강아지", "애견", "홍대",
    "8년", "9년", "8-9년", "그때나 지금이나", "처음 갔",
]

temporal_context_words = [
    "처음", "처음 와", "처음 갔", "처음 먹",
    "예전", "전에", "지난번", "오랜만", "몇 년", "8년", "9년", "8-9년",
    "그때나 지금이나", "지금이나", "다시", "또", "재방문", "종종",
    "다음에도", "다음에", "또 오", "또 방문", "또 먹",
]

mixed_opinion_words = [
    "다만", "아쉽", "아쉬", "근데", "그런데", "하지만", "그러나",
    "별관", "사람이 많", "사람 많", "그닥", "연하고", "적은",
    "비싸", "가격", "시장통", "시끄", "귀 아픔", "기다려서",
]


# ==============================
# 4. Feature 생성 함수
# ==============================

def count_keyword_occurrences(text, words):
    text = str(text)
    return sum(text.count(word) for word in words)


def has_any_keyword(text, words):
    text = str(text)
    return int(any(word in text for word in words))


def count_number_patterns(text):
    text = str(text)
    patterns = [
        r"\d+\s*분",
        r"\d+\s*인분",
        r"\d+\s*원",
        r"\d+\s*명",
        r"\d+\s*시",
        r"\d+\s*차",
        r"\d+\s*번",
        r"\d+\s*개",
        r"\d+\s*점",
        r"\d+\s*년",
        r"\d+\s*-\s*\d+\s*년",
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def sentence_count(text):
    parts = re.split(r"[.!?。~]+|\n", str(text))
    return len([p for p in parts if p.strip()])


def build_rule_features(row):
    text = str(row["review_text"]).strip()
    review_length = row["review_length"]

    generic_count = count_keyword_occurrences(text, generic_words)
    reason_count = count_keyword_occurrences(text, reason_words)
    menu_count = count_keyword_occurrences(text, menu_words)
    detailed_negative_count = count_keyword_occurrences(text, detailed_negative)
    generic_negative_count = count_keyword_occurrences(text, generic_negative)
    event_word_count = count_keyword_occurrences(text, event_words)
    intensifier_count = count_keyword_occurrences(text, intensifier_words)

    quality_detail_count = count_keyword_occurrences(text, quality_detail_words)
    sensory_detail_count = count_keyword_occurrences(text, sensory_detail_words)
    waiting_popularity_count = count_keyword_occurrences(text, popularity_waiting_words)
    experience_context_count = count_keyword_occurrences(text, experience_context_words)
    temporal_context_count = count_keyword_occurrences(text, temporal_context_words)
    mixed_opinion_count = count_keyword_occurrences(text, mixed_opinion_words)

    category_count = 0
    for word_list in categories.values():
        if has_any_keyword(text, word_list):
            category_count += 1

    number_detail_count = count_number_patterns(text)
    sent_count = sentence_count(text)

    has_generic_praise = int(generic_count >= 1)
    strong_generic_praise = int(generic_count >= 2)
    has_reason = int(reason_count >= 1)
    strong_reason = int(reason_count >= 2)
    has_menu = int(menu_count >= 1)
    has_detailed_negative = int(detailed_negative_count >= 1)
    has_generic_negative = int(generic_negative_count >= 1)
    has_number_detail = int(number_detail_count >= 1)

    has_quality_detail = int(quality_detail_count >= 1)
    strong_quality_detail = int(quality_detail_count >= 2)
    has_sensory_detail = int(sensory_detail_count >= 1)
    strong_sensory_detail = int(sensory_detail_count >= 2)
    has_waiting_popularity = int(waiting_popularity_count >= 1)
    has_experience_context = int(experience_context_count >= 1)
    has_temporal_context = int(temporal_context_count >= 1)
    has_mixed_opinion = int(mixed_opinion_count >= 1)

    short_review = int(review_length < 30)
    ultra_short_review = int(review_length < 10)
    multi_sentence_review = int(sent_count >= 2)

    first_visit = int(pd.notna(row["visit_count"]) and row["visit_count"] == 1)
    revisit = int(pd.notna(row["visit_count"]) and row["visit_count"] >= 2)

    account_review_count = row["account_review_count"]
    account_review_count_is_1 = int(pd.notna(account_review_count) and account_review_count == 1)
    account_review_count_under_5 = int(pd.notna(account_review_count) and account_review_count <= 5)
    account_review_count_under_10 = int(pd.notna(account_review_count) and account_review_count < 10)
    account_review_count_over_50 = int(pd.notna(account_review_count) and account_review_count >= 50)
    account_review_count_over_100 = int(pd.notna(account_review_count) and account_review_count >= 100)

    specific_review = int(
        has_detailed_negative == 1
        or has_number_detail == 1
        or has_quality_detail == 1
        or has_sensory_detail == 1
        or has_experience_context == 1
        or has_temporal_context == 1
        or has_mixed_opinion == 1
        or (has_reason == 1 and review_length >= 45)
        or (category_count >= 2 and review_length >= 50)
        or (multi_sentence_review == 1 and review_length >= 60)
    )

    generic_without_detail = int(
        has_generic_praise == 1
        and specific_review == 0
        and has_menu == 0
    )

    short_generic_review = int(short_review == 1 and has_generic_praise == 1)

    low_information_review = int(
        short_review == 1
        and specific_review == 0
        and has_menu == 0
        and has_detailed_negative == 0
    )

    popularity_only_review = int(
        has_waiting_popularity == 1
        and category_count <= 1
        and has_number_detail == 0
        and has_quality_detail == 0
        and has_sensory_detail == 0
        and has_experience_context == 0
        and has_temporal_context == 0
        and review_length < 80
    )

    enthusiastic_but_detailed = int(
        has_generic_praise == 1
        and (
            specific_review == 1
            or has_quality_detail == 1
            or has_sensory_detail == 1
            or has_detailed_negative == 1
            or has_experience_context == 1
            or has_temporal_context == 1
            or has_mixed_opinion == 1
        )
    )

    positive_narrative = int(
        has_generic_praise == 1
        and (
            has_sensory_detail == 1
            or has_quality_detail == 1
            or has_menu == 1
        )
        and (
            has_experience_context == 1
            or has_temporal_context == 1
            or category_count >= 2
            or review_length >= 70
            or multi_sentence_review == 1
        )
    )

    mixed_opinion_review = int(
        (
            has_mixed_opinion == 1
            or has_detailed_negative == 1
            or has_generic_negative == 1
        )
        and (
            has_generic_praise == 1
            or has_sensory_detail == 1
            or has_quality_detail == 1
            or has_menu == 1
        )
        and review_length >= 40
    )

    temporal_experience_review = int(
        has_temporal_context == 1
        and (
            review_length >= 40
            or has_menu == 1
            or has_experience_context == 1
        )
    )

    menu_price_listing_only = int(
        menu_count >= 2
        and number_detail_count >= 2
        and has_sensory_detail == 0
        and has_quality_detail == 0
        and has_experience_context == 0
        and has_temporal_context == 0
        and category_count == 0
        and sent_count <= 1
    )

    short_but_specific_food_review = int(
        review_length < 60
        and (
            has_menu == 1
            or has_sensory_detail == 1
            or has_quality_detail == 1
        )
        and generic_without_detail == 0
    )

    detailed_negative_narrative = int(
        has_detailed_negative == 1
        and (
            review_length >= 50
            or sent_count >= 2
            or category_count >= 1
            or has_experience_context == 1
            or has_temporal_context == 1
        )
    )

    return {
        "generic_count": generic_count,
        "reason_count": reason_count,
        "menu_count": menu_count,
        "category_count": category_count,
        "number_detail_count": number_detail_count,
        "generic_negative_count": generic_negative_count,
        "detailed_negative_count": detailed_negative_count,
        "event_word_count": event_word_count,
        "intensifier_count": intensifier_count,
        "quality_detail_count": quality_detail_count,
        "sensory_detail_count": sensory_detail_count,
        "waiting_popularity_count": waiting_popularity_count,
        "experience_context_count": experience_context_count,
        "temporal_context_count": temporal_context_count,
        "mixed_opinion_count": mixed_opinion_count,
        "sentence_count": sent_count,

        "has_generic_praise": has_generic_praise,
        "strong_generic_praise": strong_generic_praise,
        "has_reason": has_reason,
        "strong_reason": strong_reason,
        "has_menu": has_menu,
        "has_number_detail": has_number_detail,
        "has_generic_negative": has_generic_negative,
        "has_detailed_negative": has_detailed_negative,
        "has_quality_detail": has_quality_detail,
        "strong_quality_detail": strong_quality_detail,
        "has_sensory_detail": has_sensory_detail,
        "strong_sensory_detail": strong_sensory_detail,
        "has_waiting_popularity": has_waiting_popularity,
        "has_experience_context": has_experience_context,
        "has_temporal_context": has_temporal_context,
        "has_mixed_opinion": has_mixed_opinion,
        "multi_sentence_review": multi_sentence_review,

        "short_review": short_review,
        "ultra_short_review": ultra_short_review,
        "short_generic_review": short_generic_review,
        "generic_without_detail": generic_without_detail,
        "low_information_review": low_information_review,
        "specific_review": specific_review,
        "popularity_only_review": popularity_only_review,
        "enthusiastic_but_detailed": enthusiastic_but_detailed,
        "positive_narrative": positive_narrative,
        "mixed_opinion_review": mixed_opinion_review,
        "temporal_experience_review": temporal_experience_review,
        "menu_price_listing_only": menu_price_listing_only,
        "short_but_specific_food_review": short_but_specific_food_review,
        "detailed_negative_narrative": detailed_negative_narrative,

        "first_visit": first_visit,
        "revisit": revisit,
        "account_review_count_is_1": account_review_count_is_1,
        "account_review_count_under_5": account_review_count_under_5,
        "account_review_count_under_10": account_review_count_under_10,
        "account_review_count_over_50": account_review_count_over_50,
        "account_review_count_over_100": account_review_count_over_100,
    }


# ==============================
# 5. trust level 변환 함수
# ==============================

def convert_score_to_trust_level(score):
    if score < LOW_TRUST_THRESHOLD:
        return "low"
    elif score < HIGH_TRUST_THRESHOLD:
        return "ambiguous"
    else:
        return "high"


def convert_level_to_weight(level):
    if level == "low":
        return 0.2
    elif level == "ambiguous":
        return 0.5
    else:
        return 1.0


# ==============================
# 6. 규칙 기반 진정성 점수 함수
# ==============================

def predict_authenticity(row):
    features = build_rule_features(row)
    trust_score = 0
    trust_reasons = []

    # ------------------------------
    # 낮은 진정성 신호
    # ------------------------------

    if features["ultra_short_review"] == 1:
        trust_score -= 2
        trust_reasons.append("ultra_short_review:-2")

    if features["short_review"] == 1 and features["specific_review"] == 0:
        trust_score -= 1.5
        trust_reasons.append("short_low_specificity:-1.5")

    if features["strong_generic_praise"] == 1 and features["specific_review"] == 0:
        trust_score -= 1.5
        trust_reasons.append("strong_generic_without_specificity:-1.5")

    if features["generic_without_detail"] == 1:
        trust_score -= 3
        trust_reasons.append("generic_without_detail:-3")

    if features["short_generic_review"] == 1:
        if features["specific_review"] == 1:
            trust_score -= 0.2
            trust_reasons.append("short_generic_but_specific:-0.2")
        else:
            trust_score -= 2
            trust_reasons.append("short_generic_review:-2")

    if features["low_information_review"] == 1:
        trust_score -= 2
        trust_reasons.append("low_information_review:-2")

    if (
        features["has_number_detail"] == 0
        and features["has_generic_praise"] == 1
        and features["specific_review"] == 0
    ):
        trust_score -= 1
        trust_reasons.append("no_number_detail_with_praise:-1")

    if (
        features["has_menu"] == 1
        and features["has_quality_detail"] == 0
        and features["has_sensory_detail"] == 0
        and features["has_experience_context"] == 0
        and features["has_temporal_context"] == 0
        and features["category_count"] <= 1
        and row["review_length"] < 80
    ):
        trust_score -= 1.0
        trust_reasons.append("menu_without_experience_detail:-1.0")

    if features["account_review_count_is_1"] == 1:
        trust_score -= 1.0
        trust_reasons.append("account_review_count_is_1:-1.0")
    elif features["account_review_count_under_5"] == 1:
        trust_score -= 0.5
        trust_reasons.append("account_review_count_under_5:-0.5")
    elif features["account_review_count_under_10"] == 1:
        trust_score -= 0.2
        trust_reasons.append("account_review_count_under_10:-0.2")

    if features["first_visit"] == 1 and row["review_length"] < 80 and features["specific_review"] == 0:
        trust_score -= 1.0
        trust_reasons.append("first_visit_short_low_specificity:-1")
    elif features["first_visit"] == 1:
        trust_score -= 0.2
        trust_reasons.append("first_visit:-0.2")

    if features["event_word_count"] > 0:
        trust_score -= 2
        trust_reasons.append("event_word:-2")

    if (
        features["intensifier_count"] >= 2
        and features["category_count"] <= 1
        and features["has_sensory_detail"] == 0
        and features["has_quality_detail"] == 0
        and features["has_experience_context"] == 0
        and features["has_temporal_context"] == 0
        and row["review_length"] < 80
    ):
        trust_score -= 1.0
        trust_reasons.append("many_intensifier_weak_context:-1")

    if features["has_generic_negative"] == 1 and features["has_detailed_negative"] == 0:
        if features["has_experience_context"] == 1 or features["has_temporal_context"] == 1 or row["review_length"] >= 40:
            trust_score -= 0.5
            trust_reasons.append("generic_negative_weak_detail:-0.5")
        else:
            trust_score -= 1.0
            trust_reasons.append("generic_negative_without_detail:-1.0")

    if features["popularity_only_review"] == 1:
        trust_score -= 1
        trust_reasons.append("popularity_only_without_detail:-1")

    if features["menu_price_listing_only"] == 1:
        trust_score -= 3
        trust_reasons.append("menu_price_listing_only:-3")

    # ------------------------------
    # 높은 진정성 신호
    # ------------------------------

    if features["has_experience_context"] == 1 and row["review_length"] >= 35:
        trust_score += 1.0
        trust_reasons.append("experience_context:+1.0")

    if features["has_temporal_context"] == 1 and row["review_length"] >= 35:
        trust_score += 1.0
        trust_reasons.append("temporal_context:+1.0")

    if features["temporal_experience_review"] == 1:
        trust_score += 1.0
        trust_reasons.append("temporal_experience_review:+1.0")

    if features["has_menu"] == 1:
        trust_score += 0.3
        trust_reasons.append("has_menu:+0.3")

    if features["has_quality_detail"] == 1:
        trust_score += 0.6
        trust_reasons.append("has_quality_detail:+0.6")

    if features["strong_quality_detail"] == 1:
        trust_score += 0.5
        trust_reasons.append("strong_quality_detail:+0.5")

    if features["has_sensory_detail"] == 1:
        trust_score += 0.6
        trust_reasons.append("has_sensory_detail:+0.6")

    if features["strong_sensory_detail"] == 1:
        trust_score += 0.5
        trust_reasons.append("strong_sensory_detail:+0.5")

    if (
        features["has_sensory_detail"] == 1
        and features["has_menu"] == 1
        and row["review_length"] >= 50
    ):
        trust_score += 1.0
        trust_reasons.append("menu_sensory_review:+1.0")

    if features["positive_narrative"] == 1:
        trust_score += 2.0
        trust_reasons.append("positive_narrative:+2.0")

    if features["mixed_opinion_review"] == 1:
        trust_score += 1.5
        trust_reasons.append("mixed_opinion_review:+1.5")

    if features["has_mixed_opinion"] == 1 and row["review_length"] >= 40:
        trust_score += 0.8
        trust_reasons.append("mixed_opinion_context:+0.8")

    if features["has_reason"] == 1 and row["review_length"] >= 45:
        trust_score += 0.8
        trust_reasons.append("has_reason_with_length:+0.8")

    if features["strong_reason"] == 1 and row["review_length"] >= 70:
        trust_score += 0.8
        trust_reasons.append("strong_reason_with_length:+0.8")

    if features["category_count"] >= 1:
        add = min(features["category_count"], 2) * 0.7
        trust_score += add
        trust_reasons.append(f"category_count:+{add}")

    if features["category_count"] >= 2 and row["review_length"] >= 50:
        trust_score += 0.8
        trust_reasons.append("rich_category_context:+0.8")

    if features["multi_sentence_review"] == 1 and row["review_length"] >= 60:
        trust_score += 0.8
        trust_reasons.append("multi_sentence_with_length:+0.8")

    if features["has_number_detail"] == 1:
        trust_score += 1
        trust_reasons.append("has_number_detail:+1")

    if features["has_detailed_negative"] == 1:
        trust_score += 1.5
        trust_reasons.append("has_detailed_negative:+1.5")

    if features["has_detailed_negative"] == 1 and row["review_length"] >= 40:
        trust_score += 1.2
        trust_reasons.append("detailed_negative_with_context:+1.2")

    if features["detailed_negative_narrative"] == 1:
        trust_score += 1.5
        trust_reasons.append("detailed_negative_narrative:+1.5")

    if features["has_waiting_popularity"] == 1:
        trust_score += 0.2
        trust_reasons.append("waiting_or_popularity_context:+0.2")

    if features["enthusiastic_but_detailed"] == 1:
        if (
            row["review_length"] >= 70
            and (
                features["category_count"] >= 2
                or features["has_sensory_detail"] == 1
                or features["has_experience_context"] == 1
                or features["has_temporal_context"] == 1
            )
        ):
            trust_score += 0.5
            trust_reasons.append("enthusiastic_but_detailed_rich:+0.5")
        else:
            trust_score += 0.0
            trust_reasons.append("enthusiastic_but_not_rich:+0")

    if features["short_but_specific_food_review"] == 1:
        if features["has_sensory_detail"] == 1 or features["has_quality_detail"] == 1:
            trust_score += 0.5
            trust_reasons.append("short_but_specific_food_review:+0.5")
        else:
            trust_score += 0.2
            trust_reasons.append("short_but_menu_only_food_review:+0.2")

    if features["revisit"] == 1:
        trust_score += 0.8
        trust_reasons.append("revisit:+0.8")

    if features["account_review_count_over_100"] == 1:
        trust_score += 1.0
        trust_reasons.append("account_review_count_over_100:+1.0")
    elif features["account_review_count_over_50"] == 1:
        trust_score += 0.7
        trust_reasons.append("account_review_count_over_50:+0.7")

    if pd.notna(row["has_photo"]) and int(row["has_photo"]) == 1:
        if row["review_length"] >= 80 and features["specific_review"] == 1:
            trust_score += 0.2
            trust_reasons.append("has_photo_with_detail:+0.2")
        else:
            trust_score -= 0.2
            trust_reasons.append("photo_without_enough_detail:-0.2")

    # 플랫폼 보조 규칙
    if str(row["platform"]).lower() == "kakao":
        if pd.notna(row["rating"]) and row["rating"] == 5 and row["review_length"] < 80 and features["specific_review"] == 0:
            trust_score -= 1
            trust_reasons.append("kakao_5star_short_low_specificity:-1")

    if str(row["platform"]).lower() == "naver":
        if features["first_visit"] == 1 and row["review_length"] < 80 and features["has_generic_praise"] == 1 and features["specific_review"] == 0:
            trust_score -= 0.7
            trust_reasons.append("naver_first_visit_short_praise_low_specificity:-0.7")

    # binary label
    pred_label = 1 if trust_score >= BINARY_TRUST_THRESHOLD else 0

    # 3단계 trust system
    trust_level = convert_score_to_trust_level(trust_score)
    trust_weight = convert_level_to_weight(trust_level)

    # 식당 점수 계산은 하지 않음.
    # 대신 리뷰 단위에서만 신뢰 가중치 반영값을 저장.
    if pd.notna(row.get("rating", pd.NA)):
        rating_x_trust_weight = row["rating"] * trust_weight
    else:
        rating_x_trust_weight = pd.NA

    result = {
        "trust_score": trust_score,
        "pred_label": pred_label,
        "trust_level": trust_level,
        "trust_weight": trust_weight,
        "rating_x_trust_weight": rating_x_trust_weight,
        "trust_reasons": " | ".join(trust_reasons),
    }
    result.update(features)

    return pd.Series(result)


# ==============================
# 7. train/test 분할 및 예측
# ==============================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["manual_label"]
)

train_df = train_df.copy()
test_df = test_df.copy()

train_result = train_df.apply(predict_authenticity, axis=1)
test_result = test_df.apply(predict_authenticity, axis=1)

train_df = pd.concat([train_df, train_result], axis=1)
test_df = pd.concat([test_df, test_result], axis=1)

all_result_df = pd.concat(
    [train_df.assign(split="train"), test_df.assign(split="test")],
    ignore_index=True
)


# ==============================
# 8. 성능 평가
# ==============================

def evaluate_result(name, result_df):
    y_true = result_df["manual_label"]
    y_pred = result_df["pred_label"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    fake_precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    fake_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    fake_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    print("\n==============================")
    print(f"{name} Binary 성능")
    print("==============================")
    print(cm)
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["fake", "real"], zero_division=0))
    print("Accuracy:", acc)
    print("F1 Score(real=1):", f1)
    print("F1 Score(fake=0):", fake_f1)

    return {
        "dataset": name,
        "accuracy": acc,
        "precision_real_1": precision,
        "recall_real_1": recall,
        "f1_real_1": f1,
        "precision_fake_0": fake_precision,
        "recall_fake_0": fake_recall,
        "f1_fake_0": fake_f1,
        "wrong_cases": int((y_true != y_pred).sum()),
        "total": int(len(result_df)),
        "tn_fake_correct": int(cm[0, 0]),
        "fp_fake_to_real": int(cm[0, 1]),
        "fn_real_to_fake": int(cm[1, 0]),
        "tp_real_correct": int(cm[1, 1]),
    }


all_metrics = evaluate_result("ALL", all_result_df)
train_metrics = evaluate_result("TRAIN", train_df)
test_metrics = evaluate_result("TEST", test_df)

metrics_df = pd.DataFrame([all_metrics, train_metrics, test_metrics])
metrics_df.to_csv(OUTPUT_DIR / "filter_metrics_binary.csv", index=False, encoding="utf-8-sig")


# ==============================
# 9. Threshold 자동 탐색
# ==============================

threshold_results = []

for threshold in [x / 10 for x in range(-10, 81)]:
    temp_pred = (all_result_df["trust_score"] >= threshold).astype(int)

    y_true = all_result_df["manual_label"]
    y_pred = temp_pred

    f1_real = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_real": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_real": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_real": f1_real,
        "precision_fake": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_fake": f1_fake,
        "macro_f1": (f1_real + f1_fake) / 2,
    })

threshold_df = pd.DataFrame(threshold_results).sort_values("macro_f1", ascending=False)
threshold_df.to_csv(OUTPUT_DIR / "threshold_tuning_result.csv", index=False, encoding="utf-8-sig")

print("\n===== Threshold Tuning TOP 10 =====")
print(threshold_df.head(10).to_string(index=False))


# ==============================
# 10. Trust level 요약
# ==============================

trust_level_summary = (
    all_result_df
    .groupby(["trust_level", "manual_label"])
    .size()
    .reset_index(name="count")
)

trust_level_summary.to_csv(OUTPUT_DIR / "trust_level_summary.csv", index=False, encoding="utf-8-sig")

trust_level_pivot = pd.crosstab(
    all_result_df["trust_level"],
    all_result_df["manual_label"],
    margins=True
)

trust_level_pivot.to_csv(OUTPUT_DIR / "trust_level_pivot.csv", encoding="utf-8-sig")

print("\n===== Trust Level 분포 =====")
print(trust_level_pivot)


# ==============================
# 11. 결과 저장
# ==============================

all_wrong = all_result_df[all_result_df["manual_label"] != all_result_df["pred_label"]].copy()
test_wrong = test_df[test_df["manual_label"] != test_df["pred_label"]].copy()

all_result_df.to_csv(OUTPUT_DIR / "all_filter_result.csv", index=False, encoding="utf-8-sig")
train_df.to_csv(OUTPUT_DIR / "train_filter_result.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(OUTPUT_DIR / "test_filter_result.csv", index=False, encoding="utf-8-sig")

all_wrong.to_csv(OUTPUT_DIR / "all_wrong_cases.csv", index=False, encoding="utf-8-sig")
test_wrong.to_csv(OUTPUT_DIR / "test_wrong_cases.csv", index=False, encoding="utf-8-sig")


# ==============================
# 12. Feature Summary
# ==============================

key_feature_cols = [
    "review_length",
    "account_review_count",
    "visit_count",
    "has_photo",
    "rating",
    "account_avg_rating",

    "has_generic_praise",
    "generic_count",
    "strong_generic_praise",
    "generic_without_detail",
    "short_generic_review",
    "low_information_review",

    "has_number_detail",
    "number_detail_count",

    "quality_detail_count",
    "has_quality_detail",
    "strong_quality_detail",

    "sensory_detail_count",
    "has_sensory_detail",
    "strong_sensory_detail",

    "waiting_popularity_count",
    "has_waiting_popularity",
    "experience_context_count",
    "has_experience_context",
    "temporal_context_count",
    "has_temporal_context",
    "mixed_opinion_count",
    "has_mixed_opinion",

    "positive_narrative",
    "mixed_opinion_review",
    "temporal_experience_review",

    "popularity_only_review",
    "enthusiastic_but_detailed",
    "menu_price_listing_only",
    "short_but_specific_food_review",
    "detailed_negative_narrative",

    "sentence_count",
    "multi_sentence_review",

    "first_visit",
    "revisit",
    "account_review_count_is_1",
    "account_review_count_under_5",
    "account_review_count_under_10",
    "account_review_count_over_50",
    "account_review_count_over_100",

    "has_menu",
    "menu_count",
    "category_count",
    "has_detailed_negative",
    "specific_review",
    "trust_score",
    "trust_weight",
    "rating_x_trust_weight",
]

key_feature_cols = [col for col in key_feature_cols if col in all_result_df.columns]

label_feature_summary = all_result_df.groupby("manual_label")[key_feature_cols].mean(numeric_only=True).round(3)
label_feature_summary.to_csv(OUTPUT_DIR / "manual_label_feature_summary.csv", encoding="utf-8-sig")

pred_feature_summary = all_result_df.groupby("pred_label")[key_feature_cols].mean(numeric_only=True).round(3)
pred_feature_summary.to_csv(OUTPUT_DIR / "pred_label_feature_summary.csv", encoding="utf-8-sig")

trust_level_feature_summary = all_result_df.groupby("trust_level")[key_feature_cols].mean(numeric_only=True).round(3)
trust_level_feature_summary.to_csv(OUTPUT_DIR / "trust_level_feature_summary.csv", encoding="utf-8-sig")

print("\n===== manual_label별 핵심 feature 평균 =====")
print(label_feature_summary.to_string())

print("\n===== trust_level별 핵심 feature 평균 =====")
print(trust_level_feature_summary.to_string())


# ==============================
# 13. Random Forest 보조 검증
# ==============================

rf_feature_cols = [
    col for col in key_feature_cols
    if col not in ["trust_score", "trust_weight", "rating_x_trust_weight"]
]

X_train = train_df[rf_feature_cols].fillna(0)
y_train = train_df["manual_label"]

X_test = test_df[rf_feature_cols].fillna(0)
y_test = test_df["manual_label"]

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n===== Random Forest 보조 검증 =====")
print(classification_report(y_test, rf_pred, labels=[0, 1], target_names=["fake", "real"], zero_division=0))

importance = pd.Series(
    rf.feature_importances_,
    index=rf_feature_cols
).sort_values(ascending=False)

importance_df = importance.reset_index()
importance_df.columns = ["feature", "importance"]
importance_df.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False, encoding="utf-8-sig")

print("\n===== Random Forest Feature Importance TOP 20 =====")
print(importance.head(20).to_string())

# ==============================
# 13-1. Random Forest Feature Importance 시각화
# ==============================

# 상위 n개 feature만 표시
top_n = 20
rf_top = importance.head(top_n).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(rf_top.index, rf_top.values)

plt.title("Random Forest Feature Importance TOP 20", fontsize=14)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "random_forest_feature_importance_top20.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("- random_forest_feature_importance_top20.png 저장 완료")


# ==============================
# 13-2. Random Forest Confusion Matrix 시각화
# ==============================

rf_cm = confusion_matrix(y_test, rf_pred, labels=[0, 1])

plt.figure(figsize=(6, 5))
plt.imshow(rf_cm)

plt.title("Random Forest Confusion Matrix", fontsize=14)
plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.xticks([0, 1], ["fake(0)", "real(1)"])
plt.yticks([0, 1], ["fake(0)", "real(1)"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(rf_cm[i, j]), ha="center", va="center", fontsize=13)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("- random_forest_confusion_matrix.png 저장 완료")


# ==============================
# 13-3. Random Forest 주요 성능 지표 시각화
# ==============================

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision_real = precision_score(y_test, rf_pred, pos_label=1, zero_division=0)
rf_recall_real = recall_score(y_test, rf_pred, pos_label=1, zero_division=0)
rf_f1_real = f1_score(y_test, rf_pred, pos_label=1, zero_division=0)

rf_metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision(real=1)", "Recall(real=1)", "F1-score(real=1)"],
    "Score": [rf_accuracy, rf_precision_real, rf_recall_real, rf_f1_real]
})

rf_metrics_df.to_csv(
    OUTPUT_DIR / "random_forest_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(8, 5))
bars = plt.bar(rf_metrics_df["Metric"], rf_metrics_df["Score"])

plt.title("Random Forest 주요 성능 지표", fontsize=14)
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "random_forest_metrics.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("- random_forest_metrics.csv 저장 완료")
print("- random_forest_metrics.png 저장 완료")

# ==============================
# 14. 시각화
# ==============================

def save_confusion_matrix_png(cm, title, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["fake(0)", "real(1)"])
    plt.yticks([0, 1], ["fake(0)", "real(1)"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


all_cm = confusion_matrix(all_result_df["manual_label"], all_result_df["pred_label"], labels=[0, 1])
train_cm = confusion_matrix(train_df["manual_label"], train_df["pred_label"], labels=[0, 1])
test_cm = confusion_matrix(test_df["manual_label"], test_df["pred_label"], labels=[0, 1])

save_confusion_matrix_png(all_cm, "ALL Binary Confusion Matrix", "confusion_matrix_all.png")
save_confusion_matrix_png(train_cm, "TRAIN Binary Confusion Matrix", "confusion_matrix_train.png")
save_confusion_matrix_png(test_cm, "TEST Binary Confusion Matrix", "confusion_matrix_test.png")


# ==============================
# 15. 요약 저장
# ==============================

summary_path = OUTPUT_DIR / "분석_요약.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("v19 trust-level 기반 진정성 필터 결과\n")
    f.write("=" * 70 + "\n\n")

    f.write("[수정 방향]\n")
    f.write("- v16/v17의 3단계 trust system을 유지함\n")
    f.write("- 식당 단위 점수 계산은 제외하고 리뷰별 trust_weight만 산출함\n")
    f.write("- low=0.2, ambiguous=0.5, high=1.0의 리뷰별 신뢰 가중치를 부여함\n")
    f.write("- 구체적인 긍정 경험 리뷰가 fake로 떨어지는 문제를 완화하기 위해 positive_narrative feature를 추가함\n")
    f.write("- 예전 방문, 재방문, 처음 방문 등 temporal context를 genuine signal로 반영함\n")
    f.write("- 장점과 단점이 함께 나타나는 mixed opinion review를 genuine signal로 반영함\n")
    f.write("- negative detail 중심 판단에서 experience richness 중심 판단으로 보정함\n")
    f.write("- rating_x_trust_weight는 식당 점수가 아니라 리뷰 단위 가중 평점 요소로만 저장함\n\n")

    f.write("[Binary 성능]\n")
    f.write(metrics_df.to_string(index=False))
    f.write("\n\n")

    f.write("[Threshold Tuning TOP 10]\n")
    f.write(threshold_df.head(10).to_string(index=False))
    f.write("\n\n")

    f.write("[Trust Level 분포]\n")
    f.write(trust_level_pivot.to_string())
    f.write("\n\n")

    f.write("[manual_label별 핵심 feature 평균]\n")
    f.write(label_feature_summary.to_string())
    f.write("\n\n")

    f.write("[trust_level별 핵심 feature 평균]\n")
    f.write(trust_level_feature_summary.to_string())
    f.write("\n\n")

    f.write("[Random Forest Feature Importance TOP 20]\n")
    f.write(importance.head(20).to_string())
    f.write("\n")

print("\n[저장 완료]")
print(f"결과 폴더: {OUTPUT_DIR}")
print("- filter_metrics_binary.csv")
print("- threshold_tuning_result.csv")
print("- trust_level_summary.csv")
print("- trust_level_pivot.csv")
print("- all_wrong_cases.csv")
print("- test_wrong_cases.csv")
print("- train_filter_result.csv")
print("- test_filter_result.csv")
print("- all_filter_result.csv")
print("- manual_label_feature_summary.csv")
print("- pred_label_feature_summary.csv")
print("- trust_level_feature_summary.csv")
print("- confusion_matrix_all.png")
print("- confusion_matrix_train.png")
print("- confusion_matrix_test.png")
print("- random_forest_feature_importance.csv")
print("- 분석_요약.txt")
print("\nALL 전체 binary 오분류 개수:", len(all_wrong))
print("TEST binary 오분류 개수:", len(test_wrong))