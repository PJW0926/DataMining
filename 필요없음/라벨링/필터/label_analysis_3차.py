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
# 목적
# =========================================================
# - 수동 라벨링 데이터(all_라벨링_3차.csv)에 규칙 기반 진정성 필터 적용
# - 리뷰이벤트 식당 vs 일반 식당 비교에서 도출된 feature를 가중치에 반영
# - 특히 아래 feature를 최종 필터에 반영:
#   1) generic praise
#   2) generic without detail
#   3) has number detail
#   4) 방문횟수
#   5) 계정리뷰수
# - 성능은 manual_label 기준으로 Accuracy / Precision / Recall / F1 / Confusion Matrix로 평가
#
# 라벨 기준:
# - manual_label = 0 : 낮은 진정성 / 이벤트성·광고성 의심
# - manual_label = 1 : 높은 진정성 / 실제성 높음
#
# trust_score 기준:
# - trust_score가 높을수록 높은 진정성 / 신뢰도 높음
# - trust_score가 TRUST_THRESHOLD 이상이면 pred_label = 1
# - trust_score가 TRUST_THRESHOLD 미만이면 pred_label = 0
# =========================================================


# ==============================
# 0. 설정
# ==============================

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "all_라벨링_3차.csv"

OUTPUT_DIR = BASE_DIR / "filter_result_trust_score_event_pattern"
OUTPUT_DIR.mkdir(exist_ok=True)

# 이 값은 train/test 결과를 보고 조정 가능
# 현재는 trust_score >= 0 이면 높은 진정성으로 판정
TRUST_THRESHOLD = 0

# Windows 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ==============================
# 1. 파일 불러오기
# ==============================

df = pd.read_csv(csv_path, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)


# ==============================
# 2. 데이터 정리
# ==============================

# 필수 컬럼 보정
if "review_text" not in df.columns:
    raise ValueError("review_text 컬럼이 필요합니다.")

if "manual_label" not in df.columns:
    raise ValueError("manual_label 컬럼이 필요합니다.")

df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
df["review_length"] = df["review_text"].str.len()

# platform / store_name이 없을 경우 대비
if "platform" not in df.columns:
    df["platform"] = "unknown"

if "store_name" not in df.columns:
    df["store_name"] = "unknown"

# visit_count 원본 확인
print("\n===== visit_count 원본 값 확인 =====")
if "visit_count" in df.columns:
    print(df["visit_count"].value_counts(dropna=False).head(20).to_string())
else:
    print("visit_count 컬럼 없음")
    df["visit_count"] = pd.NA

# visit_count 숫자화: '1번째 방문' -> 1
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
    "manual_label",
    "visit_count",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df[col] = pd.NA

# 라벨 정리
df = df.dropna(subset=["manual_label"]).copy()
df["manual_label"] = df["manual_label"].astype(int)

# 사진 유무 정리
df["has_photo"] = pd.to_numeric(df["has_photo"], errors="coerce").fillna(0).astype(int)

# 빈 리뷰 제거
before_empty = len(df)
df = df[df["review_text"] != ""].copy()
after_empty = len(df)
print(f"\n===== 빈 리뷰 제거 =====")
print(f"{before_empty}개 -> {after_empty}개")


# ==============================
# 3. 전체 데이터 패턴 확인
# ==============================

compare_cols = [
    "account_review_count",
    "visit_count",
    "review_length",
    "has_photo",
    "rating",
    "account_avg_rating",
]
existing_cols = [col for col in compare_cols if col in df.columns]

print("\n===== 라벨 개수 =====")
print(df["manual_label"].value_counts().to_string())

print("\n===== 전체 평균 비교 =====")
summary = df.groupby("manual_label")[existing_cols].mean(numeric_only=True)
print(summary.to_string())

print("\n===== 전체 중앙값 비교 =====")
median_summary = df.groupby("manual_label")[existing_cols].median(numeric_only=True)
print(median_summary.to_string())

print("\n===== 플랫폼별 라벨 개수 =====")
print(pd.crosstab(df["platform"], df["manual_label"]).to_string())

print("\n===== 플랫폼별 평균 비교 =====")
platform_summary = df.groupby(["platform", "manual_label"])[existing_cols].mean(numeric_only=True)
print(platform_summary.to_string())


# ==============================
# 4. 단어 사전
# ==============================

# 일반적 칭찬
generic_words = [
    "맛있", "마싯", "맛잇",
    "좋아", "좋았", "좋습", "좋네", "좋음",
    "친절",
    "최고",
    "추천", "강추",
    "굿", "굳",
    "존맛",
    "맛집",
    "짱",
    "대박",
    "훌륭",
    "완전 추천",
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
    "재방문 안",
]

# 구체적 부정: 오히려 실제 경험 신호로 봄
detailed_negative = [
    "짜", "싱겁", "맵", "달",
    "느끼", "비리", "차갑", "식었",
    "불친절", "늦", "오래 걸", "오래걸",
    "웨이팅", "기다렸", "대기",
    "비싸", "가격",
    "양이 적", "적음",
    "불편",
    "실수",
    "별로였",
]

# 이유/근거 표현
reason_words = [
    "때문", "해서", "이라", "는데", "지만",
    "식감", "소스", "양", "가격",
    "친절", "응대", "서비스",
    "분위기", "인테리어",
    "웨이팅", "대기",
    "고소", "담백", "쫄깃", "바삭",
    "신선", "깔끔", "진한",
    "잡내", "국물", "재료", "조리",
]

# 메뉴 단어: 특정 식당에 너무 종속되지 않게 범용 한식/외식 메뉴 중심
menu_words = [
    "곱창", "막창", "대창", "전골", "볶음밥", "감자전", "콘치즈",
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
    "육회", "해장국",
    "파스타", "피자", "뇨끼", "리조또", "스테이크",
]

# 카테고리별 구체 표현
categories = {
    "atmosphere": [
        "분위기", "인테리어", "조용", "깔끔",
        "쾌적", "넓", "아늑", "시끄럽",
    ],
    "service": [
        "친절", "응대", "직원", "서빙",
        "사장님", "알바", "서비스",
    ],
    "waiting": [
        "웨이팅", "대기", "기다렸",
        "줄서", "예약",
    ],
    "taste": [
        "식감", "쫄깃", "바삭",
        "고소", "담백", "진한",
        "싱겁", "짜", "맵", "달",
        "부드럽", "촉촉", "잡내", "국물",
    ],
    "context": [
        "점심", "저녁", "주말", "평일",
        "회식", "데이트", "친구", "가족",
        "혼밥", "퇴근", "2차",
    ],
}

# 리뷰이벤트/과장 표현
event_words = [
    "이벤트", "리뷰이벤트", "인스타", "릴스", "sns", "SNS",
]

intensifier_words = [
    "진짜", "너무", "완전", "엄청", "존맛", "대박", "짱", "개맛", "핵맛",
]


# ==============================
# 5. Feature 생성 함수
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
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


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

    category_count = 0
    for word_list in categories.values():
        if has_any_keyword(text, word_list):
            category_count += 1

    number_detail_count = count_number_patterns(text)

    has_generic_praise = int(generic_count >= 1)
    strong_generic_praise = int(generic_count >= 2)
    has_reason = int(reason_count >= 1)
    strong_reason = int(reason_count >= 2)
    has_menu = int(menu_count >= 1)
    has_detailed_negative = int(detailed_negative_count >= 1)
    has_generic_negative = int(generic_negative_count >= 1)
    has_number_detail = int(number_detail_count >= 1)

    short_review = int(review_length < 30)
    ultra_short_review = int(review_length < 10)

    # 핵심: 리뷰이벤트 식당 비교에서 확인한 feature
    # generic praise는 있는데 근거/구체성/숫자/메뉴가 부족한 경우
    generic_without_detail = int(
        has_generic_praise == 1
        and has_reason == 0
        and category_count == 0
        and has_number_detail == 0
        and has_menu == 0
    )

    short_generic_review = int(short_review == 1 and has_generic_praise == 1)

    # 방문횟수
    first_visit = int(pd.notna(row["visit_count"]) and row["visit_count"] == 1)
    revisit = int(pd.notna(row["visit_count"]) and row["visit_count"] >= 2)

    # 계정 리뷰 수
    account_review_count = row["account_review_count"]
    account_review_count_is_1 = int(pd.notna(account_review_count) and account_review_count == 1)
    account_review_count_under_5 = int(pd.notna(account_review_count) and account_review_count <= 5)
    account_review_count_under_10 = int(pd.notna(account_review_count) and account_review_count < 10)
    account_review_count_over_50 = int(pd.notna(account_review_count) and account_review_count >= 50)
    account_review_count_over_100 = int(pd.notna(account_review_count) and account_review_count >= 100)

    # 정보량 낮은 리뷰
    low_information_review = int(
        short_review == 1
        and has_reason == 0
        and category_count == 0
        and has_number_detail == 0
        and has_menu == 0
        and has_detailed_negative == 0
    )

    # 구체성 높은 리뷰
    specific_review = int(
        has_menu == 1
        or has_number_detail == 1
        or category_count >= 2
        or has_detailed_negative == 1
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

        "has_generic_praise": has_generic_praise,
        "strong_generic_praise": strong_generic_praise,
        "has_reason": has_reason,
        "strong_reason": strong_reason,
        "has_menu": has_menu,
        "has_number_detail": has_number_detail,
        "has_generic_negative": has_generic_negative,
        "has_detailed_negative": has_detailed_negative,

        "short_review": short_review,
        "ultra_short_review": ultra_short_review,
        "short_generic_review": short_generic_review,
        "generic_without_detail": generic_without_detail,
        "low_information_review": low_information_review,
        "specific_review": specific_review,

        "first_visit": first_visit,
        "revisit": revisit,

        "account_review_count_is_1": account_review_count_is_1,
        "account_review_count_under_5": account_review_count_under_5,
        "account_review_count_under_10": account_review_count_under_10,
        "account_review_count_over_50": account_review_count_over_50,
        "account_review_count_over_100": account_review_count_over_100,
    }


# ==============================
# 6. 규칙 기반 진정성 필터 함수
# ==============================

def predict_authenticity(row):
    features = build_rule_features(row)
    trust_score = 0
    trust_reasons = []

    # -------------------------------------------------
    # A. 낮은 진정성 신호: trust_score 감소
    # -------------------------------------------------

    # 1. 짧은 리뷰
    if features["short_review"] == 1:
        trust_score -= 1
        trust_reasons.append("short_review:-1")

    if features["ultra_short_review"] == 1:
        trust_score -= 1
        trust_reasons.append("ultra_short_review:-1")

    # 2. generic praise
    # 리뷰이벤트 식당에서 일반적 칭찬이 높게 나왔으므로 trust 감소 요인으로 반영
    if features["strong_generic_praise"] == 1:
        trust_score -= 1
        trust_reasons.append("strong_generic_praise:-1")

    # 3. generic without detail
    # 이번 수정의 핵심: 일반 칭찬은 있는데 구체 근거가 없으면 강하게 trust 감소
    if features["generic_without_detail"] == 1:
        trust_score -= 3
        trust_reasons.append("generic_without_detail:-3")

    # 4. 짧고 일반 칭찬만 있는 리뷰
    if features["short_generic_review"] == 1:
        trust_score -= 2
        trust_reasons.append("short_generic_review:-2")

    # 5. low information
    if features["low_information_review"] == 1:
        trust_score -= 2
        trust_reasons.append("low_information_review:-2")

    # 6. 숫자 디테일 없음 + 일반 칭찬
    if features["has_number_detail"] == 0 and features["has_generic_praise"] == 1:
        trust_score -= 1
        trust_reasons.append("no_number_detail_with_praise:-1")

    # 7. 계정 리뷰 수 적음
    if features["account_review_count_is_1"] == 1:
        trust_score -= 3
        trust_reasons.append("account_review_count_is_1:-3")
    elif features["account_review_count_under_5"] == 1:
        trust_score -= 2
        trust_reasons.append("account_review_count_under_5:-2")
    elif features["account_review_count_under_10"] == 1:
        trust_score -= 1
        trust_reasons.append("account_review_count_under_10:-1")

    # 8. 방문횟수 1회
    if features["first_visit"] == 1 and row["review_length"] < 80:
        trust_score -= 1.5
        trust_reasons.append("first_visit_short:-1.5")
    elif features["first_visit"] == 1:
        trust_score -= 0.5
        trust_reasons.append("first_visit:-0.5")

    # 9. 이벤트 직접 언급
    if features["event_word_count"] > 0:
        trust_score -= 2
        trust_reasons.append("event_word:-2")

    # 10. 과장 표현이 많은데 구체성이 부족한 경우
    if features["intensifier_count"] >= 2 and features["specific_review"] == 0:
        trust_score -= 1
        trust_reasons.append("many_intensifier_without_specificity:-1")

    # 11. 이유 없는 비난도 낮은 정보량으로 봄
    if features["has_generic_negative"] == 1 and features["has_detailed_negative"] == 0:
        trust_score -= 2
        trust_reasons.append("generic_negative_without_detail:-2")

    # -------------------------------------------------
    # B. 높은 진정성 신호: trust_score 증가
    # -------------------------------------------------

    # 1. 메뉴 언급
    if features["has_menu"] == 1:
        trust_score += 1.5
        trust_reasons.append("has_menu:+1.5")

    # 2. 카테고리 구체성
    if features["category_count"] >= 1:
        add = min(features["category_count"], 3)
        trust_score += add
        trust_reasons.append(f"category_count:+{add}")

    # 3. 숫자 디테일
    # 몇 분, 몇 명, 몇 원, 몇 인분 등 구체 수치가 있으면 실제 경험 신호로 반영
    if features["has_number_detail"] == 1:
        trust_score += 2
        trust_reasons.append("has_number_detail:+2")

    # 4. 구체적 부정
    if features["has_detailed_negative"] == 1:
        trust_score += 2
        trust_reasons.append("has_detailed_negative:+2")

    # 5. 재방문
    if features["revisit"] == 1:
        trust_score += 1
        trust_reasons.append("revisit:+1")

    # 6. 계정 리뷰 수가 충분히 많음
    if features["account_review_count_over_100"] == 1:
        trust_score += 1.5
        trust_reasons.append("account_review_count_over_100:+1.5")
    elif features["account_review_count_over_50"] == 1:
        trust_score += 1
        trust_reasons.append("account_review_count_over_50:+1")

    # 7. 사진 있음: 약한 real 신호
    if pd.notna(row["has_photo"]) and int(row["has_photo"]) == 1:
        trust_score += 0.5
        trust_reasons.append("has_photo:+0.5")

    # -------------------------------------------------
    # C. 플랫폼 보조 규칙
    # -------------------------------------------------

    # Kakao: 별점 5 + 짧은 리뷰 + 일반 칭찬이면 trust 감소
    if str(row["platform"]).lower() == "kakao":
        if pd.notna(row["rating"]) and row["rating"] == 5 and row["review_length"] < 80 and features["has_generic_praise"] == 1:
            trust_score -= 1
            trust_reasons.append("kakao_5star_short_praise:-1")

        if pd.notna(row["account_avg_rating"]) and row["account_avg_rating"] >= 4.3 and row["review_length"] < 80 and features["specific_review"] == 0:
            trust_score -= 1
            trust_reasons.append("kakao_high_avg_short_low_specificity:-1")

    # Naver: 첫 방문 + 짧고 일반적인 리뷰이면 추가 trust 감소
    if str(row["platform"]).lower() == "naver":
        if features["first_visit"] == 1 and row["review_length"] < 80 and features["has_generic_praise"] == 1:
            trust_score -= 1
            trust_reasons.append("naver_first_visit_short_praise:-1")

    # 최종 판정
    pred = 1 if trust_score >= TRUST_THRESHOLD else 0

    result = {
        "trust_score": trust_score,
        "pred_label": pred,
        "trust_reasons": " | ".join(trust_reasons),
    }
    result.update(features)

    return pd.Series(result)

# ==============================
# 7. train/test 분할
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
print(train_df.groupby("manual_label")[existing_cols].mean(numeric_only=True).to_string())

print("\n===== train 데이터 기준 중앙값 비교 =====")
print(train_df.groupby("manual_label")[existing_cols].median(numeric_only=True).to_string())


# ==============================
# 8. train/test 예측
# ==============================

train_df = train_df.copy()
test_df = test_df.copy()

train_result = train_df.apply(predict_authenticity, axis=1)
test_result = test_df.apply(predict_authenticity, axis=1)

train_df = pd.concat([train_df, train_result], axis=1)
test_df = pd.concat([test_df, test_result], axis=1)

print("\n===== TRAIN trust_score 분포 =====")
print(train_df["trust_score"].value_counts().sort_index().to_string())

print("\n===== TEST trust_score 분포 =====")
print(test_df["trust_score"].value_counts().sort_index().to_string())


# ==============================
# 9. 성능 평가 함수
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
    print(f"{name} 성능")
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


train_metrics = evaluate_result("TRAIN", train_df)
test_metrics = evaluate_result("TEST", test_df)

metrics_df = pd.DataFrame([train_metrics, test_metrics])
metrics_df.to_csv(OUTPUT_DIR / "filter_metrics.csv", index=False, encoding="utf-8-sig")


# ==============================
# 10. 오분류 저장
# ==============================

test_wrong = test_df[test_df["manual_label"] != test_df["pred_label"]].copy()

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
    "trust_score",
    "manual_label",
    "pred_label",
    "trust_reasons",

    # 이번에 반영한 핵심 feature
    "has_generic_praise",
    "generic_count",
    "generic_without_detail",
    "has_number_detail",
    "number_detail_count",
    "first_visit",
    "revisit",
    "account_review_count_is_1",
    "account_review_count_under_5",
    "account_review_count_under_10",
    "account_review_count_over_50",
    "account_review_count_over_100",
]

existing_wrong_cols = [col for col in wrong_cols if col in test_wrong.columns]

test_wrong[existing_wrong_cols].to_csv(
    OUTPUT_DIR / "test_wrong_cases.csv",
    index=False,
    encoding="utf-8-sig"
)

train_df.to_csv(OUTPUT_DIR / "train_filter_result.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(OUTPUT_DIR / "test_filter_result.csv", index=False, encoding="utf-8-sig")

all_result_df = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")], ignore_index=True)
all_result_df.to_csv(OUTPUT_DIR / "all_filter_result.csv", index=False, encoding="utf-8-sig")


# ==============================
# 11. 핵심 feature 라벨별 평균 비교 저장
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
]

key_feature_cols = [col for col in key_feature_cols if col in all_result_df.columns]

label_feature_summary = all_result_df.groupby("manual_label")[key_feature_cols].mean(numeric_only=True).round(3)
label_feature_summary.to_csv(OUTPUT_DIR / "manual_label_feature_summary.csv", encoding="utf-8-sig")

pred_feature_summary = all_result_df.groupby("pred_label")[key_feature_cols].mean(numeric_only=True).round(3)
pred_feature_summary.to_csv(OUTPUT_DIR / "pred_label_feature_summary.csv", encoding="utf-8-sig")

print("\n===== manual_label별 핵심 feature 평균 =====")
print(label_feature_summary.to_string())


# ==============================
# 12. 시각화
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


train_cm = confusion_matrix(train_df["manual_label"], train_df["pred_label"], labels=[0, 1])
test_cm = confusion_matrix(test_df["manual_label"], test_df["pred_label"], labels=[0, 1])

save_confusion_matrix_png(train_cm, "TRAIN Confusion Matrix", "confusion_matrix_train.png")
save_confusion_matrix_png(test_cm, "TEST Confusion Matrix", "confusion_matrix_test.png")


# 성능 막대그래프
plot_metrics = metrics_df.set_index("dataset")[
    ["accuracy", "precision_real_1", "recall_real_1", "f1_real_1", "f1_fake_0"]
]

ax = plot_metrics.T.plot(kind="bar", figsize=(10, 6))
plt.title("Filter Performance Metrics")
plt.xlabel("metric")
plt.ylabel("score")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.legend(title="dataset")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "filter_performance_metrics.png", dpi=200)
plt.close()


# 핵심 feature 라벨별 비교 그래프
plot_feature_cols = [
    "has_generic_praise",
    "generic_without_detail",
    "has_number_detail",
    "first_visit",
    "revisit",
    "account_review_count_under_10",
    "account_review_count_over_50",
    "low_information_review",
    "specific_review",
]
plot_feature_cols = [c for c in plot_feature_cols if c in all_result_df.columns]

plot_feature_df = all_result_df.groupby("manual_label")[plot_feature_cols].mean(numeric_only=True).T
plot_feature_df.columns = ["fake_manual_0" if c == 0 else "real_manual_1" for c in plot_feature_df.columns]

plot_feature_df.plot(kind="bar", figsize=(11, 6))
plt.title("Manual Label별 핵심 Feature 평균 비교")
plt.xlabel("feature")
plt.ylabel("mean / ratio")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "manual_label_feature_comparison.png", dpi=200)
plt.close()


# ==============================
# 13. Random Forest 보조 검증
# ==============================
# 규칙 기반 필터가 메인이고,
# Random Forest는 "이 feature들이 manual_label 분류에 의미가 있는지" 확인하는 보조 분석

rf_feature_cols = [
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
    "number_detail_count",
    "event_word_count",
    "intensifier_count",

    "has_generic_praise",
    "strong_generic_praise",
    "generic_without_detail",
    "has_number_detail",
    "short_generic_review",
    "low_information_review",
    "first_visit",
    "revisit",
    "account_review_count_is_1",
    "account_review_count_under_5",
    "account_review_count_under_10",
    "account_review_count_over_50",
    "account_review_count_over_100",
    "has_detailed_negative",
    "specific_review",
]

rf_feature_cols = [c for c in rf_feature_cols if c in train_df.columns]

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

rf_metrics = {
    "accuracy": accuracy_score(y_test, rf_pred),
    "f1_real_1": f1_score(y_test, rf_pred, pos_label=1, zero_division=0),
    "f1_fake_0": f1_score(y_test, rf_pred, pos_label=0, zero_division=0),
}
print("Random Forest metrics:", rf_metrics)

importance = pd.Series(
    rf.feature_importances_,
    index=rf_feature_cols
).sort_values(ascending=False)

importance_df = importance.reset_index()
importance_df.columns = ["feature", "importance"]
importance_df.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False, encoding="utf-8-sig")

print("\n===== Random Forest Feature Importance =====")
print(importance.head(20).to_string())

# Feature Importance 그래프
top_importance = importance.head(20).sort_values(ascending=True)
plt.figure(figsize=(10, 7))
plt.barh(top_importance.index, top_importance.values)
plt.title("Random Forest Feature Importance TOP 20")
plt.xlabel("importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "random_forest_feature_importance.png", dpi=200)
plt.close()


# ==============================
# 14. 요약 저장
# ==============================

summary_path = OUTPUT_DIR / "분석_요약.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("리뷰이벤트 패턴 반영 trust_score 기반 진정성 필터 결과\n")
    f.write("=" * 70 + "\n\n")

    f.write("[반영한 핵심 feature]\n")
    f.write("- generic praise: 일반적 칭찬이 많으면 trust_score 감소\n")
    f.write("- generic without detail: 일반 칭찬은 있으나 구체 근거가 없으면 trust_score 크게 감소\n")
    f.write("- has number detail: 숫자 기반 구체 정보가 있으면 trust_score 증가\n")
    f.write("- visit_count: 첫 방문은 trust_score 감소, 재방문은 trust_score 증가\n")
    f.write("- account_review_count: 리뷰 수가 매우 적으면 trust_score 감소, 충분히 많으면 trust_score 증가\n\n")

    f.write("[성능]\n")
    f.write(metrics_df.to_string(index=False))
    f.write("\n\n")

    f.write("[manual_label별 핵심 feature 평균]\n")
    f.write(label_feature_summary.to_string())
    f.write("\n\n")

    f.write("[Random Forest Feature Importance TOP 20]\n")
    f.write(importance.head(20).to_string())
    f.write("\n")

print("\n[저장 완료]")
print(f"결과 폴더: {OUTPUT_DIR}")
print("- filter_metrics.csv")
print("- test_wrong_cases.csv")
print("- train_filter_result.csv")
print("- test_filter_result.csv")
print("- all_filter_result.csv")
print("- manual_label_feature_summary.csv")
print("- pred_label_feature_summary.csv")
print("- confusion_matrix_train.png")
print("- confusion_matrix_test.png")
print("- filter_performance_metrics.png")
print("- manual_label_feature_comparison.png")
print("- random_forest_feature_importance.csv")
print("- random_forest_feature_importance.png")
print("- 분석_요약.txt")
print("\nTEST 오분류 개수:", len(test_wrong))
