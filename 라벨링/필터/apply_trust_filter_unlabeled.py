import pandas as pd
from pathlib import Path
import re


# =========================
# 1. 경로 설정
# =========================

INPUT_DIR = Path("크롤링/input_csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "merged_filtered_reviews.csv"


# =========================
# 2. 컬럼명 통일 함수
# =========================

def standardize_columns(df):
    """
    파일마다 컬럼명이 조금씩 다를 수 있으므로
    가능한 컬럼명을 하나의 표준 컬럼명으로 통일한다.
    """

    rename_map = {
        # 리뷰 내용
        "review": "review_text",
        "content": "review_text",
        "text": "review_text",
        "리뷰": "review_text",
        "리뷰내용": "review_text",
        "리뷰 내용": "review_text",
        "review_text": "review_text",

        # 계정명
        "user": "user_id",
        "user_id": "user_id",
        "account": "user_id",
        "account_id": "user_id",
        "계정": "user_id",
        "계정 ID": "user_id",
        "작성자": "user_id",

        # 계정 리뷰 수
        "account_review_count": "account_review_count",
        "계정 리뷰 수": "account_review_count",
        "계정의 리뷰 수": "account_review_count",
        "작성자 리뷰 수": "account_review_count",

        # 방문 횟수
        "visit_count": "visit_count",
        "방문 횟수": "visit_count",
        "방문횟수": "visit_count",

        # 사진 유무
        "has_photo": "has_photo",
        "photo": "has_photo",
        "사진 유무": "has_photo",
        "사진유무": "has_photo",

        # 별점
        "rating": "rating",
        "별점": "rating",
        "score": "rating",

        # 계정 평균 별점
        "account_avg_rating": "account_avg_rating",
        "계정 평균 별점": "account_avg_rating",
        "작성자 평균 별점": "account_avg_rating",

        # 날짜
        "date": "date",
        "visit_date": "date",
        "방문 날짜": "date",
        "작성일": "date",

        # 인증 수단
        "auth_method": "auth_method",
        "인증 수단": "auth_method",
        "인증수단": "auth_method",
    }

    df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})

    return df


# =========================
# 3. 값 정리 함수
# =========================

def parse_visit_count(value):
    """
    '1번째 방문', '2번째 방문' 같은 값을 숫자로 바꾼다.
    값이 없으면 0으로 처리한다.
    """
    if pd.isna(value):
        return 0

    value = str(value)
    match = re.search(r"(\d+)", value)

    if match:
        return int(match.group(1))

    return 0


def parse_bool_photo(value):
    """
    사진 유무 값을 1/0으로 바꾼다.
    """
    if pd.isna(value):
        return 0

    value = str(value).strip().lower()

    true_values = ["true", "1", "yes", "y", "있음", "사진있음", "o", "유"]
    false_values = ["false", "0", "no", "n", "없음", "사진없음", "x", "무"]

    if value in true_values:
        return 1
    elif value in false_values:
        return 0
    else:
        return 0


def to_number(value, default=0):
    """
    숫자로 바꿀 수 있으면 숫자로 바꾸고,
    안 되면 기본값을 넣는다.
    """
    try:
        if pd.isna(value):
            return default
        return float(str(value).replace(",", "").strip())
    except:
        return default


# =========================
# 4. 진정성 필터 함수
# =========================

def calculate_trust_score(row):
    """
    리뷰 하나마다 진정성 점수를 계산한다.
    점수가 높을수록 실제 경험 리뷰일 가능성이 높다고 본다.
    """

    score = 0

    review_text = str(row.get("review_text", ""))
    review_length = len(review_text)

    account_review_count = row.get("account_review_count", 0)
    visit_count = row.get("visit_count", 0)
    has_photo = row.get("has_photo", 0)
    rating = row.get("rating", 0)
    account_avg_rating = row.get("account_avg_rating", 0)

    # -------------------------
    # 1) 리뷰 길이
    # -------------------------
    if review_length >= 80:
        score += 3
    elif review_length >= 40:
        score += 2
    elif review_length >= 15:
        score += 1
    elif review_length <= 5:
        score -= 3
    else:
        score -= 1

    # -------------------------
    # 2) 메뉴 언급
    # -------------------------
    menu_words = [
        "곱창", "막창", "대창", "삼겹살", "고기", "국밥", "라멘", "파스타",
        "피자", "치킨", "볶음밥", "찌개", "전골", "냉면", "만두", "튀김",
        "커피", "디저트", "케이크", "빵", "샐러드", "초밥", "회", "탕"
    ]

    if any(word in review_text for word in menu_words):
        score += 2

    # -------------------------
    # 3) 구체적 경험 표현
    # -------------------------
    experience_words = [
        "웨이팅", "예약", "직원", "친절", "불친절", "재방문", "포장",
        "매장", "분위기", "자리", "주차", "화장실", "가격", "양이",
        "식감", "냄새", "느끼", "고소", "맵", "짜", "달", "싱겁",
        "바삭", "부드럽", "질기", "신선", "잡내"
    ]

    if any(word in review_text for word in experience_words):
        score += 2

    # -------------------------
    # 4) 광고/이벤트 의심 표현
    # -------------------------
    suspicious_words = [
        "맛있어요", "맛있습니다", "좋아요", "최고예요", "추천합니다",
        "또 올게요", "재방문할게요", "친절해요", "대박", "굿",
        "완전 추천", "강추", "존맛", "맛집"
    ]

    short_positive = review_length <= 15 and any(word in review_text for word in suspicious_words)

    if short_positive:
        score -= 3

    # -------------------------
    # 5) 사진 유무
    # -------------------------
    if has_photo == 1:
        score += 1

    # -------------------------
    # 6) 방문 횟수
    # -------------------------
    if visit_count >= 2:
        score += 2
    elif visit_count == 1:
        score += 1

    # -------------------------
    # 7) 계정 리뷰 수
    # -------------------------
    if account_review_count >= 1000:
        score -= 2
    elif account_review_count >= 500:
        score -= 1
    elif 5 <= account_review_count <= 200:
        score += 1

    # -------------------------
    # 8) 별점 패턴
    # -------------------------
    if rating == 5 and review_length <= 10:
        score -= 2

    if account_avg_rating >= 4.8 and rating == 5:
        score -= 1

    return score


def classify_review(score):
    """
    score를 기준으로 최종 라벨을 나눈다.
    1 = 진짜 리뷰 가능성 높음
    0 = 낮은 신뢰도 리뷰 가능성 높음
    """

    if score >= 4:
        return 1
    else:
        return 0


def classify_trust_level(score):
    """
    성능 확인용 구간.
    최종 제거 여부는 pred_label을 쓰면 된다.
    """

    if score >= 6:
        return "high"
    elif score >= 3:
        return "ambiguous"
    else:
        return "low"


# =========================
# 5. CSV 전체 읽고 합치기
# =========================

def read_csv_safely(file_path):
    """
    CSV 인코딩이 파일마다 다를 수 있으므로
    여러 인코딩을 순서대로 시도한다.
    """

    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[ERROR] {file_path.name} 읽기 실패: {e}")
            return None

    print(f"[ERROR] 인코딩 문제로 읽기 실패: {file_path.name}")
    return None


def merge_all_csv():
    csv_files = list(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        print("[ERROR] input_csv 폴더 안에 CSV 파일이 없습니다.")
        return None

    all_dfs = []

    print(f"[INFO] 발견된 CSV 파일 수: {len(csv_files)}개")

    for file_path in csv_files:
        print(f"[INFO] 읽는 중: {file_path.name}")

        df = read_csv_safely(file_path)

        if df is None:
            continue

        df = standardize_columns(df)

        # 원본 파일명 남기기
        df["source_file"] = file_path.name

        # 파일명에서 식당명 추정
        df["restaurant_name"] = file_path.stem

        all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] 읽을 수 있는 CSV 파일이 없습니다.")
        return None

    merged_df = pd.concat(all_dfs, ignore_index=True)

    print(f"[INFO] 전체 합쳐진 리뷰 수: {len(merged_df)}개")

    return merged_df


# =========================
# 6. 필수 컬럼 보정
# =========================

def prepare_dataframe(df):
    """
    없는 컬럼은 기본값으로 생성한다.
    """

    required_columns = [
        "review_text",
        "user_id",
        "account_review_count",
        "visit_count",
        "has_photo",
        "rating",
        "account_avg_rating",
        "date",
        "auth_method"
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = None

    df["review_text"] = df["review_text"].fillna("").astype(str)

    df["review_length"] = df["review_text"].apply(len)
    df["visit_count"] = df["visit_count"].apply(parse_visit_count)
    df["has_photo"] = df["has_photo"].apply(parse_bool_photo)
    df["account_review_count"] = df["account_review_count"].apply(lambda x: to_number(x, 0))
    df["rating"] = df["rating"].apply(lambda x: to_number(x, 0))
    df["account_avg_rating"] = df["account_avg_rating"].apply(lambda x: to_number(x, 0))

    return df


# =========================
# 7. 실행
# =========================

def main():
    merged_df = merge_all_csv()

    if merged_df is None:
        return

    merged_df = prepare_dataframe(merged_df)

    print("[INFO] 진정성 필터 적용 중...")

    merged_df["trust_score"] = merged_df.apply(calculate_trust_score, axis=1)
    merged_df["pred_label"] = merged_df["trust_score"].apply(classify_review)
    merged_df["trust_level"] = merged_df["trust_score"].apply(classify_trust_level)

    merged_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print()
    print("===== 처리 완료 =====")
    print(f"저장 파일: {OUTPUT_FILE}")
    print(f"전체 리뷰 수: {len(merged_df)}개")
    print()
    print("===== pred_label 분포 =====")
    print(merged_df["pred_label"].value_counts())
    print()
    print("===== trust_level 분포 =====")
    print(merged_df["trust_level"].value_counts())
    print()
    print("pred_label 의미:")
    print("1 = 진짜 리뷰 가능성 높음")
    print("0 = 낮은 신뢰도 리뷰 가능성 높음")


if __name__ == "__main__":
    main()