# =========================================================
# 리뷰 진정성 패턴 분석 프로젝트
# =========================================================

# ---------------------------------------------------------
# [실행 전 해야 할 것]
#
# 1. 아래 두 CSV 파일을
#    이 .py 파일과 같은 폴더에 넣기
#
# - naver_reviews_이태리국시_2차.csv
# - naver_reviews_금금_2차.csv
#
# 2. 터미널에서 필요한 라이브러리 설치
#
# pip install pandas konlpy matplotlib scikit-learn
#
# ---------------------------------------------------------


# =========================================================
# 1. 라이브러리 불러오기
# =========================================================

import pandas as pd
import re

from collections import Counter

from konlpy.tag import Okt
JVM_PATH = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.31.11-hotspot\bin\server\jvm.dll"

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 2. CSV 파일 불러오기
# =========================================================

df1 = pd.read_csv("라벨링/naver_reviews_이태리국시_2차.csv")
df2 = pd.read_csv("라벨링/naver_reviews_금금_2차.csv")


# =========================================================
# 3. 식당 이름 추가
# =========================================================

df1["store"] = "이태리국시"
df2["store"] = "금금"


# =========================================================
# 4. 두 데이터 합치기
# =========================================================

df = pd.concat([df1, df2], ignore_index=True)

print("\n데이터 합치기 완료")
print("총 리뷰 수:", len(df))


# =========================================================
# 5. 필요한 컬럼만 사용
# =========================================================

df = df[
    [
        "store",
        "account_id",
        "account_review",
        "visit_date",
        "visit_count",
        "verification",
        "review",
        "review_text"
    ]
]


# =========================================================
# 6. 결측 제거
# =========================================================

df = df.dropna(subset=["review"])


# =========================================================
# 7. 중복 제거
# =========================================================

df = df.drop_duplicates(subset=["review"])


# =========================================================
# 8. 텍스트 전처리
# =========================================================

def clean_text(text):

    text = str(text)

    # 특수문자 제거
    text = re.sub(r"[^가-힣0-9\s]", "", text)

    return text.strip()

df["review"] = df["review"].apply(clean_text)


# =========================================================
# 9. 리뷰 길이 계산
# =========================================================

# review_text 컬럼이 이상할 경우 대비

df["review_length"] = df["review"].str.len()

print("\n평균 리뷰 길이")

print(
    df.groupby("store")["review_length"].mean()
)


# =========================================================
# 10. 리뷰 길이 분포 시각화
# =========================================================

plt.figure(figsize=(8,5))

for store_name in df["store"].unique():

    subset = df[df["store"] == store_name]

    plt.hist(
        subset["review_length"],
        bins=30,
        alpha=0.5,
        label=store_name
    )

plt.title("Review Length Distribution")
plt.xlabel("Review Length")
plt.ylabel("Count")
plt.legend()

plt.savefig("review_length_distribution.png")
plt.close()


# =========================================================
# 11. 계정 활동성 분석
# =========================================================

print("\n평균 계정 리뷰 수")

print(
    df.groupby("store")["account_review"].mean()
)


# =========================================================
# 12. 방문 횟수 분석
# =========================================================

print("\n방문 횟수 분포")

print(
    df["visit_count"].value_counts()
)


# =========================================================
# 13. 인증 방식 분석
# =========================================================

print("\n인증 방식 분포")

print(
    df["verification"].value_counts()
)


# =========================================================
# 14. 형태소 분석
# =========================================================

okt = Okt(jvmpath=JVM_PATH)

stopwords = [
    "정말",
    "진짜",
    "너무",
    "그냥",
    "완전"
]

def extract_words(text):

    nouns = okt.nouns(str(text))

    nouns = [
        n for n in nouns
        if len(n) >= 2 and n not in stopwords
    ]

    return nouns


# =========================================================
# 15. 자주 등장하는 단어 분석
# =========================================================

for store_name in df["store"].unique():

    print("\n==============================")
    print(store_name, "상위 단어")
    print("==============================")

    subset = df[df["store"] == store_name]

    words = []

    for review in subset["review"]:

        words.extend(extract_words(review))

    counter = Counter(words)

    print(counter.most_common(20))


# =========================================================
# 16. 중복 리뷰 분석
# =========================================================

print("\n중복 리뷰 TOP 20")

duplicates = (
    df["review"]
    .value_counts()
    .head(20)
)

print(duplicates)


# =========================================================
# 17. 리뷰 유사도 분석
# =========================================================

print("\n리뷰 유사도 분석")

sample_df = df.sample(
    n=min(100, len(df)),
    random_state=42
)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(sample_df["review"])

similarity_matrix = cosine_similarity(X)

similarities = []

for i in range(len(similarity_matrix)):

    sim_scores = list(similarity_matrix[i])

    sim_scores.pop(i)

    similarities.append(max(sim_scores))

sample_df["max_similarity"] = similarities

print("\n평균 최대 유사도")

print(
    sample_df.groupby("store")["max_similarity"].mean()
)


# =========================================================
# 18. 메뉴 언급 여부 분석
# =========================================================

menu_words = [
    "파스타",
    "뇨끼",
    "피자",
    "묵은지",
    "고기",
    "면",    
]

def menu_mention(text):

    for word in menu_words:

        if word in str(text):

            return 1

    return 0

df["menu_mention"] = df["review"].apply(menu_mention)

print("\n메뉴 언급 비율")

print(
    df.groupby("store")["menu_mention"].mean()
)


# =========================================================
# 19. 진정성 점수 계산
# =========================================================

def authenticity_score(row):

    score = 10

    # 리뷰 길이 짧음
    if row["review_length"] <= 10:
        score -= 2

    # 계정 리뷰 수 적음
    if row["account_review"] <= 5:
        score -= 1

    # 방문 횟수 1회
    if row["visit_count"] == "1번째 방문":
        score -= 1

    # 메뉴 언급 없음
    if row["menu_mention"] == 0:
        score -= 1

    return score

df["authenticity_score"] = df.apply(
    authenticity_score,
    axis=1
)


# =========================================================
# 20. 진정성 점수 결과
# =========================================================

print("\n평균 진정성 점수")

print(
    df.groupby("store")["authenticity_score"].mean()
)


# =========================================================
# 21. 저진정성 리뷰 추출
# =========================================================

low_auth = df[
    df["authenticity_score"] <= 5
]

print("\n저진정성 의심 리뷰 예시")

print(
    low_auth[
        [
            "store",
            "review",
            "authenticity_score"
        ]
    ].head(20)
)


# =========================================================
# 22. 결과 저장
# =========================================================

df.to_csv(
    "review_analysis_result.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n분석 완료")
print("결과 파일 저장 완료")
print("파일명: review_analysis_result.csv")