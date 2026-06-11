import pandas as pd
from pathlib import Path
from collections import defaultdict

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# =========================================================
# 1. 경로 설정
# =========================================================
# 이 파이썬 파일과 입력 CSV 파일들을 같은 폴더에 넣고 실행하세요.

BASE_DIR = Path(__file__).resolve().parent

review_file = BASE_DIR / "final_high_trust_reviews_pos.csv"
dict_file = BASE_DIR / "통합_감성사전_v9.csv"

# 출력 파일
output_file = BASE_DIR / "final_high_trust_reviews_with_sentiment_star_v12_bert_adjusted.csv"
platform_output = BASE_DIR / "platform_category_sentiment_summary_v12_bert_adjusted.csv"
store_output = BASE_DIR / "store_category_sentiment_summary_v12_bert_adjusted.csv"
review_score_output = BASE_DIR / "review_sentiment_scores_v12_bert_adjusted.csv"

# =========================================================
# BERT 보정 설정
# =========================================================
# 카카오 실제 별점과 사전 기반 감성별점 차이가 이 값 이상인 리뷰만 BERT를 돌립니다.
BERT_DIFF_THRESHOLD = 2.0

# BERT가 positive면 +, negative면 -로 움직일 폭입니다.
# 너무 세게 보정하고 싶으면 1.0~1.5, 약하게 보정하고 싶으면 0.5로 바꾸면 됩니다.
BERT_ADJUST_STEP = 1.0

# BERT 확신도가 이 값 미만이면 애매한 결과로 보고 보정하지 않습니다.
BERT_CONFIDENCE_THRESHOLD = 0.60

# 기존에 쓰던 한국어 감성분석 모델
BERT_MODEL_NAME = "WhitePeak/bert-base-cased-Korean-sentiment"

# 맨 오른쪽에 추가될 최종 열 이름
BERT_STAR_COL = "sentiment_star_bert_adjusted"

# =========================================================
# 2. CSV 안전하게 불러오기
# =========================================================

def read_csv_safely(path):
    """utf-8-sig, utf-8, cp949, euc-kr 순서로 CSV를 읽습니다."""
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as e:
            last_error = e

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"CSV 인코딩을 읽지 못했습니다: {path} / 마지막 오류: {last_error}",
    )


df, review_enc = read_csv_safely(review_file)
sent_dict, dict_enc = read_csv_safely(dict_file)

print(f"[입력 리뷰 파일 인코딩] {review_enc}")
print(f"[입력 사전 파일 인코딩] {dict_enc}")

# =========================================================
# 3. 감성사전 정리
# =========================================================

required_cols = ["word", "category", "polarity", "score"]
missing_cols = [col for col in required_cols if col not in sent_dict.columns]

if missing_cols:
    raise ValueError(f"감성사전에 필요한 열이 없습니다: {missing_cols}")

sent_dict = sent_dict[required_cols].copy()
sent_dict = sent_dict.dropna(subset=required_cols)

sent_dict["word"] = sent_dict["word"].astype(str).str.strip()
sent_dict["category"] = sent_dict["category"].astype(str).str.strip()
sent_dict["polarity"] = sent_dict["polarity"].astype(str).str.strip()

# polarity 오타 보정
sent_dict["polarity"] = sent_dict["polarity"].replace({"negetive": "negative"})

# 빈 단어 제거
sent_dict = sent_dict[sent_dict["word"] != ""].copy()

# score 숫자 변환
sent_dict["score"] = pd.to_numeric(sent_dict["score"], errors="coerce")
sent_dict = sent_dict.dropna(subset=["score"])
sent_dict["score"] = sent_dict["score"].astype(int)

# 완전 중복 제거
sent_dict = sent_dict.drop_duplicates(subset=["word", "category", "polarity", "score"])

# =========================================================
# 3-1. 같은 단어가 같은 카테고리에 여러 점수로 들어간 경우 정리
# =========================================================
# 예:
# 훨씬 맛있다 / food / positive / 1
# 훨씬 맛있다 / food / positive / 2
# 이런 경우 둘 다 점수화되지 않도록 절댓값이 큰 점수 하나만 남김.

sent_dict["abs_score"] = sent_dict["score"].abs()

sent_dict = (
    sent_dict
    .sort_values(
        by=["word", "category", "polarity", "abs_score"],
        ascending=[True, True, True, False]
    )
    .drop_duplicates(subset=["word", "category", "polarity"], keep="first")
    .drop(columns=["abs_score"])
    .reset_index(drop=True)
)

# =========================================================
# 4. 리뷰 데이터 기본 확인
# =========================================================

if "review_text" not in df.columns:
    raise ValueError("리뷰 파일에 'review_text' 열이 없습니다.")

if "tokens" not in df.columns:
    raise ValueError("리뷰 파일에 'tokens' 열이 없습니다. 먼저 Okt 토큰 생성 코드를 실행해야 합니다.")

df["review_text"] = df["review_text"].fillna("").astype(str)
df["tokens"] = df["tokens"].fillna("").astype(str)

if "rating" in df.columns:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
else:
    print("[주의] rating 열이 없습니다. diff, diff_abs 계산은 생략됩니다.")

# =========================================================
# 5. 카테고리 설정
# =========================================================

categories = ["food", "price", "service", "atmosphere", "general"]

dict_categories = sorted(sent_dict["category"].unique())
unknown_categories = [cat for cat in dict_categories if cat not in categories]

if unknown_categories:
    print("[주의] categories 목록에 없는 카테고리가 사전에 있습니다:")
    print(unknown_categories)
    print("이 카테고리들은 점수 계산에서 제외됩니다.")

# =========================================================
# 6. 사전 구조 만들기
# =========================================================

sentiment_map = defaultdict(list)

for _, row in sent_dict.iterrows():
    word = row["word"]
    category = row["category"]
    score = row["score"]

    sentiment_map[word].append((category, score))

max_ngram = max(len(str(word).split()) for word in sentiment_map.keys())

print(f"[사전 유효 행 수] {len(sent_dict):,}")
print(f"[사전 고유 표현 수] {len(sentiment_map):,}")
print(f"[최대 n-gram 어절 수] {max_ngram}")

# =========================================================
# 7. 유틸 함수
# =========================================================

def clean_token(token):
    """
    혹시 tokens에 품사 태그가 남아 있을 경우 제거.
    예: 맛있다/Adjective -> 맛있다
    이미 품사 태그가 없다면 그대로 반환.
    """
    token = str(token).strip()
    if "/" in token:
        return token.rsplit("/", 1)[0]
    return token


def make_keyword_variants(keyword):
    """
    감성어의 원문 변형을 생성.
    예:
    맛있다 -> 맛있, 맛있는, 맛있고, 맛있습니다
    친절하다 -> 친절, 친절한, 친절하고, 친절함
    약하다 -> 약, 약한, 약하고, 약함
    """
    keyword = str(keyword).strip()
    compact_keyword = keyword.replace(" ", "")

    variants = set()
    variants.add(keyword)
    variants.add(compact_keyword)

    for part in keyword.split():
        variants.add(part)
        variants.add(part.replace(" ", ""))

    base_variants = set(variants)

    for kw in base_variants:
        if not kw:
            continue

        if kw.endswith("하다"):
            stem = kw[:-2]
            variants.add(stem)
            variants.add(stem + "하")
            variants.add(stem + "한")
            variants.add(stem + "하고")
            variants.add(stem + "하지만")
            variants.add(stem + "하지")
            variants.add(stem + "하지는")
            variants.add(stem + "하지도")
            variants.add(stem + "함")
            variants.add(stem + "했다")
            variants.add(stem + "했")
            variants.add(stem + "해")

        if kw.endswith("다"):
            stem = kw[:-1]
            variants.add(stem)
            variants.add(stem + "는")
            variants.add(stem + "고")
            variants.add(stem + "게")
            variants.add(stem + "지만")
            variants.add(stem + "지는")
            variants.add(stem + "지도")
            variants.add(stem + "은")
            variants.add(stem + "한")
            variants.add(stem + "합니다")
            variants.add(stem + "했습니다")
            variants.add(stem + "습니다")
            variants.add(stem + "스러울")
            variants.add(stem + "스럽")
            variants.add(stem + "었다")
            variants.add(stem + "었음")
            variants.add(stem + "웠다")
            variants.add(stem + "웠음")

    return [v for v in variants if v]


def has_negative_context(review_text, keyword, window=35):
    """
    긍정 감성어 주변에 부정/대조 표현이 있으면 True 반환.

    예:
    - 설렁탕이 맛있는 것도 아니고
    - 반찬이 신선하고 다양한 것도 아니고
    - 친절하다기보다는 거리감이 있었다
    - 시원한물이 아닌 미지근한물
    """
    if pd.isna(review_text):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    keyword_variants = make_keyword_variants(keyword)

    negative_patterns = [
        # 직접 부정
        "아닌", "아니", "아니라", "아니고", "아니다", "아님",
        "않", "못",
        "안함", "안 함", "안하", "안 하",
        "하지 않", "지는 않", "진 않", "지도 않", "지도 안",

        # 별로/부족/실망 계열
        "별로", "그닥", "그다지", "별루",
        "기대 이하", "잘 모르", "모르겠",
        "실망", "아쉽", "아쉬", "비추",

        # '~것도 아니고' 계열
        "것도 아니고", "것도 아니", "것도 아니다", "것도 아닌", "것도 아님",
        "것은 아니", "건 아니", "게 아니",
        "것 같지는 않", "같지는 않", "같진 않",

        # 없음 계열
        "없습니다", "없었", "없다", "없고", "없는", "없네요", "없음",

        # 대조 표현
        "기보다는", "라기보다는", "이라기보다는",
        "하다기보다는", "하기보다는", "보다는",

        # 서비스/분위기 부정 맥락
        "거리감", "불편", "편하게 식사하기에는", "편하지",
    ]

    compact_negative_patterns = [p.replace(" ", "") for p in negative_patterns]

    direct_negative_suffixes = [
        "아닌", "아니", "아니고", "아니다", "아님",
        "않", "못", "없", "별로",
        "기보다는", "라기보다는", "이라기보다는",
        "하다기보다는", "하기보다는", "보다는",
        "것도아니고", "것도아니", "것도아니다", "것도아닌",
        "건아니", "게아니",
        "지는않", "진않", "지도않",
    ]

    # 붙어 있는 표현 직접 탐지
    for kw in keyword_variants:
        compact_kw = kw.replace(" ", "")
        for suffix in direct_negative_suffixes:
            if compact_kw + suffix in compact_text:
                return True

    # 원문 기준 주변 문맥 검사
    for kw in keyword_variants:
        start_idx = 0

        while True:
            idx = text.find(kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)

            context = text[start:end]

            if any(pattern in context for pattern in negative_patterns):
                return True

            start_idx = idx + len(kw)

    # 공백 제거 문장 기준 주변 문맥 검사
    for kw in keyword_variants:
        compact_kw = kw.replace(" ", "")
        start_idx = 0

        while True:
            idx = compact_text.find(compact_kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(compact_text), idx + len(compact_kw) + window)

            context = compact_text[start:end]

            if any(pattern in context for pattern in compact_negative_patterns):
                return True

            start_idx = idx + len(compact_kw)

    return False


def has_negative_token_context(tokens, start_i, end_i, window=5):
    """
    토큰 기준 부정 문맥 검사.
    감성어가 매칭된 위치 주변, 특히 뒤쪽에 부정 표현이 있는지 확인합니다.

    예:
    - 설렁탕 맛있다 것 아니다
    - 맛있다 것 아니다
    - 신선하다 다양하다 것 아니다
    - 친절하다 기 보다는
    """
    left = max(0, start_i - 2)
    right = min(len(tokens), end_i + window)

    context_tokens = tokens[left:right]
    context = " ".join(context_tokens)
    compact_context = context.replace(" ", "")

    negative_patterns = [
        "아니다", "아니고", "아닌", "아니", "아님",
        "않다", "않음", "않고", "않은", "않",
        "안함", "안 하다", "안하다",
        "못하다", "못",

        "것 아니다", "것 아니고", "것 아니",
        "것도 아니다", "것도 아니고", "것도 아니",
        "건 아니다", "건 아니고", "건 아니",
        "게 아니다", "게 아니고", "게 아니",

        "기 보다는", "기보다는",
        "라기 보다는", "라기보다는",
        "하다기 보다는", "하다기보다는",
        "하기 보다는", "하기보다는",
        "보다는",

        "별로", "그닥", "그다지",
        "기대 이하", "모르겠다", "모르겠",
        "실망", "아쉽", "아쉬움", "비추",

        "없다", "없음", "없고", "없네요", "없습니다",
    ]

    compact_negative_patterns = [p.replace(" ", "") for p in negative_patterns]

    if any(p in context for p in negative_patterns):
        return True

    if any(p in compact_context for p in compact_negative_patterns):
        return True

    return False


def has_comparative_or_ironic_context(review_text, keyword, tokens=None, start_i=None, end_i=None, window=55):
    """
    긍정어가 현재 식당이 아니라 비교 대상/대체재/반어 문맥에 쓰였는지 검사.

    예:
    - 일반 고깃집 된찌가 훨씬 맛있다
    - 컵라면에 삼김만 먹어도 이거보단 만족스럽다
    - 이걸 먹고 맛있다고 하는 사람들은 이해불가
    - 차라리 다른 데가 낫다
    """
    if pd.isna(review_text):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    keyword_variants = make_keyword_variants(keyword)

    comparative_patterns = [
        # 현재 식당보다 다른 대상이 낫다는 표현
        "이거보단", "이것보단", "이거보다", "이것보다",
        "여기보다", "여기보단",
        "이집보다", "이 집보다",
        "이 식당보다", "이곳보다",

        # 대체재 비교
        "차라리", "컵라면", "삼김", "편의점", "씨유", "CU",
        "라면에삼김", "컵라면에삼김",

        # 다른 음식점/다른 음식 비교
        "일반고깃집", "일반 고깃집",
        "서비스로나오는", "서비스로 나오는",
        "다른곳", "다른 곳",
        "다른집", "다른 집",
        "다른식당", "다른 식당",
        "건너편", "맞은편", "옆집",

        # 비교 결론
        "훨씬낫", "훨씬 낫",
        "더낫", "더 낫",
        "훨씬맛있", "훨씬 맛있",
        "더맛있", "더 맛있",
        "만족스러울것", "만족스러울 것",
    ]

    ironic_patterns = [
        "맛있다고하는사람", "맛있다고 하는 사람",
        "맛있다고하는사람들", "맛있다고 하는 사람들",
        "맛있다고하는분", "맛있다고 하는 분",
        "맛있다고느끼", "맛있다고 느끼",
        "맛있다고생각", "맛있다고 생각",
    ]

    ironic_negative_patterns = [
        "이해불가", "이해 불가",
        "대체뭘", "대체 뭘",
        "뭘드시고", "뭘 드시고",
        "살아계시는",
        "아무리생각해도", "아무리 생각해도",
        "이해안", "이해 안",
    ]

    improvement_patterns = [
        "제발", "바꾸세요", "고치세요", "개선",
        "문제", "심각",
        "맛이없", "맛 없", "맛없",
    ]

    compact_comparative_patterns = [p.replace(" ", "") for p in comparative_patterns]
    compact_ironic_patterns = [p.replace(" ", "") for p in ironic_patterns]
    compact_ironic_negative_patterns = [p.replace(" ", "") for p in ironic_negative_patterns]
    compact_improvement_patterns = [p.replace(" ", "") for p in improvement_patterns]

    # 원문 주변 문맥 검사
    for kw in keyword_variants:
        start_idx = 0

        while True:
            idx = text.find(kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)

            context = text[start:end]
            compact_context = context.replace(" ", "")

            if any(p in context for p in comparative_patterns) or any(p in compact_context for p in compact_comparative_patterns):
                return True

            has_ironic = any(p in context for p in ironic_patterns) or any(p in compact_context for p in compact_ironic_patterns)
            has_ironic_negative = any(p in context for p in ironic_negative_patterns) or any(p in compact_context for p in compact_ironic_negative_patterns)

            if has_ironic and has_ironic_negative:
                return True

            has_improvement = any(p in context for p in improvement_patterns) or any(p in compact_context for p in compact_improvement_patterns)

            if has_improvement and ("제발" in context or "제발" in compact_context):
                return True

            start_idx = idx + len(kw)

    # 공백 제거 전체 문장 기준 검사
    for kw in keyword_variants:
        compact_kw = kw.replace(" ", "")
        start_idx = 0

        while True:
            idx = compact_text.find(compact_kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(compact_text), idx + len(compact_kw) + window)

            context = compact_text[start:end]

            if any(p in context for p in compact_comparative_patterns):
                return True

            has_ironic = any(p in context for p in compact_ironic_patterns)
            has_ironic_negative = any(p in context for p in compact_ironic_negative_patterns)

            if has_ironic and has_ironic_negative:
                return True

            has_improvement = any(p in context for p in compact_improvement_patterns)

            if has_improvement and "제발" in context:
                return True

            start_idx = idx + len(compact_kw)

    # 토큰 기준 주변 문맥 검사
    if tokens is not None and start_i is not None and end_i is not None:
        left = max(0, start_i - 8)
        right = min(len(tokens), end_i + 8)

        token_context = " ".join(tokens[left:right])
        compact_token_context = token_context.replace(" ", "")

        if any(p in token_context for p in comparative_patterns) or any(p in compact_token_context for p in compact_comparative_patterns):
            return True

        has_ironic = any(p in token_context for p in ironic_patterns) or any(p in compact_token_context for p in compact_ironic_patterns)
        has_ironic_negative = any(p in token_context for p in ironic_negative_patterns) or any(p in compact_token_context for p in compact_ironic_negative_patterns)

        if has_ironic and has_ironic_negative:
            return True

        has_improvement = any(p in token_context for p in improvement_patterns) or any(p in compact_token_context for p in compact_improvement_patterns)

        if has_improvement and ("제발" in token_context or "제발" in compact_token_context):
            return True

    return False


def has_positive_preference_context(review_text, keyword, window=45):
    """
    부정어가 취향/선호/만족 문맥에서 긍정적으로 쓰였는지 검사.

    예:
    - 탄력이 약한 편인데 이 점이 마음에 든다
    - 슴슴한 맛이 개인적으로 좋았다
    - 자극적이지 않아서 좋다
    """
    if pd.isna(review_text):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    keyword_variants = make_keyword_variants(keyword)

    positive_preference_patterns = [
        "마음에 든다",
        "마음에 들",
        "마움에 든다",   # 오타 대응
        "마움에 들",
        "맘에 든다",
        "맘에 들",

        "개인적으로 좋",
        "개인적으로 가장",
        "개인적으로 선호",
        "가장 선호",
        "선호하는",
        "선호",

        "취향",
        "만족감",
        "만족",
        "좋았다",
        "좋을",
        "좋은",
        "좋습니다",
        "도전해봐도 좋",

        "이 점이",
        "이점이",
        "이 부분이",
        "이부분이",
    ]

    compact_positive_patterns = [p.replace(" ", "") for p in positive_preference_patterns]

    for kw in keyword_variants:
        start_idx = 0

        while True:
            idx = text.find(kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)

            context = text[start:end]
            compact_context = context.replace(" ", "")

            if any(p in context for p in positive_preference_patterns):
                return True

            if any(p in compact_context for p in compact_positive_patterns):
                return True

            start_idx = idx + len(kw)

    return False


def has_other_target_negative_context(review_text, keyword, window=60):
    """
    부정어가 현재 리뷰 대상 식당이 아니라 다른 가게/다른 대상에 향하는지 검사.

    예:
    - 다른 가게들은 탄성이 높아서 아쉬웠음
    - 다른 평냉집은 만족감을 느끼지 못했다
    - 다른 집은 별로였다
    - 다른 곳은 아쉬웠다
    """
    if pd.isna(review_text):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    keyword_variants = make_keyword_variants(keyword)

    other_target_patterns = [
        "다른 가게", "다른가게",
        "다른 집", "다른집",
        "다른 곳", "다른곳",
        "다른 식당", "다른식당",
        "다른 평냉집", "다른평냉집",
        "다른 평양냉면집", "다른평양냉면집",
        "타 가게", "타가게",
        "타 식당", "타식당",
        "남의 가게",
        "근처 가게",
        "주변 가게",
    ]

    other_negative_patterns = [
        "아쉬웠",
        "아쉬움",
        "아쉽",
        "별로",
        "실망",
        "만족감을 느끼진 못",
        "만족감을 느끼지 못",
        "만족감 못",
        "못했다",
        "못했",
        "부족",
        "별루",
        "그닥",
        "그다지",
    ]

    compact_other_target_patterns = [p.replace(" ", "") for p in other_target_patterns]
    compact_other_negative_patterns = [p.replace(" ", "") for p in other_negative_patterns]

    for kw in keyword_variants:
        start_idx = 0

        while True:
            idx = text.find(kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)

            context = text[start:end]
            compact_context = context.replace(" ", "")

            has_other_target = (
                any(p in context for p in other_target_patterns)
                or any(p in compact_context for p in compact_other_target_patterns)
            )

            has_other_negative = (
                any(p in context for p in other_negative_patterns)
                or any(p in compact_context for p in compact_other_negative_patterns)
            )

            if has_other_target and has_other_negative:
                return True

            start_idx = idx + len(kw)

    return False


def sentiment_to_star(score, score_min=-5, score_max=4):
    """
    category_total_score를 1~5점 별점으로 변환.
    score_min 이하이면 1점, score_max 이상이면 5점으로 고정.
    """
    if pd.isna(score):
        return pd.NA

    if score_max == score_min:
        return pd.NA

    normalized = (score - score_min) / (score_max - score_min)
    normalized = max(0, min(1, normalized))

    return round(1 + normalized * 4, 2)

# =========================================================
# 8. 리뷰별 감성점수 계산
# =========================================================

def analyze_review(tokens_text, review_text):
    raw_tokens = str(tokens_text).split()
    tokens = [clean_token(t) for t in raw_tokens if clean_token(t) != ""]

    scores = {cat: 0 for cat in categories}
    matched = {cat: [] for cat in categories}

    matched_terms = set()
    used_token_idx = set()

    # 긴 표현부터 먼저 매칭
    for n in range(max_ngram, 0, -1):
        for i in range(len(tokens) - n + 1):
            idx_range = set(range(i, i + n))

            # 이미 긴 표현으로 잡힌 토큰이면 스킵
            if used_token_idx & idx_range:
                continue

            term = " ".join(tokens[i:i + n])

            if term not in sentiment_map:
                continue

            # 같은 표현 반복 점수화를 막음
            if term in matched_terms:
                continue

            for category, score in sentiment_map[term]:
                if category in scores:

                    neg_context_found = False
                    comparative_context_found = False
                    positive_preference_found = False
                    other_target_negative_found = False

                    # =================================================
                    # A. 긍정어 보정
                    # =================================================
                    if score > 0:
                        # 1. 원문 기반 부정 문맥 검사
                        if has_negative_context(review_text, term):
                            neg_context_found = True

                        # 2. term 내부 단어 부정 문맥 검사
                        if not neg_context_found:
                            for part in term.split():
                                if has_negative_context(review_text, part):
                                    neg_context_found = True
                                    break

                        # 3. 토큰 기반 부정 문맥 검사
                        if not neg_context_found:
                            if has_negative_token_context(tokens, i, i + n, window=5):
                                neg_context_found = True

                        # 4. 비교/반어/대상 전환 문맥 검사
                        if not neg_context_found:
                            if has_comparative_or_ironic_context(
                                review_text=review_text,
                                keyword=term,
                                tokens=tokens,
                                start_i=i,
                                end_i=i + n,
                                window=55
                            ):
                                comparative_context_found = True

                    # =================================================
                    # B. 부정어 보정
                    # =================================================
                    if score < 0:
                        # 1. 부정어가 취향 긍정 문맥이면 무효화
                        if has_positive_preference_context(review_text, term):
                            positive_preference_found = True

                        if not positive_preference_found:
                            for part in term.split():
                                if has_positive_preference_context(review_text, part):
                                    positive_preference_found = True
                                    break

                        # 2. 부정어가 현재 식당이 아니라 다른 가게를 향하면 무효화
                        if not positive_preference_found:
                            if has_other_target_negative_context(review_text, term):
                                other_target_negative_found = True

                        if not positive_preference_found and not other_target_negative_found:
                            for part in term.split():
                                if has_other_target_negative_context(review_text, part):
                                    other_target_negative_found = True
                                    break

                    # =================================================
                    # C. 최종 점수 반영
                    # =================================================
                    if score > 0 and neg_context_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[neg_context]")

                    elif score > 0 and comparative_context_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[comparative_context]")

                    elif score < 0 and positive_preference_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[positive_preference]")

                    elif score < 0 and other_target_negative_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[other_target_negative]")

                    else:
                        adjusted_score = score
                        matched[category].append(f"{term}:{score}[token]")

                    scores[category] += adjusted_score

            matched_terms.add(term)
            used_token_idx.update(idx_range)

    result = {}

    for cat in categories:
        result[f"{cat}_score"] = scores[cat]
        result[f"{cat}_matched_words"] = " ".join(matched[cat])

        if scores[cat] > 0:
            result[f"{cat}_label"] = "positive"
        elif scores[cat] < 0:
            result[f"{cat}_label"] = "negative"
        else:
            result[f"{cat}_label"] = "neutral"

    result["category_total_score"] = sum(scores.values())

    if result["category_total_score"] > 0:
        result["category_total_label"] = "positive"
    elif result["category_total_score"] < 0:
        result["category_total_label"] = "negative"
    else:
        result["category_total_label"] = "neutral"

    return pd.Series(result)


result_df = df.apply(
    lambda row: analyze_review(row["tokens"], row["review_text"]),
    axis=1
)

df = pd.concat([df, result_df], axis=1)


# =========================================================
# 8-A. BERT 기반 감성별점 보정 함수
# =========================================================

def normalize_bert_label(label):
    """
    transformers 모델별 label 표기가 달라서 한 번에 정리합니다.
    예: positive, POSITIVE, LABEL_1, 1, negative, LABEL_0, 0
    """
    label = str(label).strip().lower()

    if "positive" in label or label in ["pos", "1", "label_1"]:
        return "positive"

    if "negative" in label or label in ["neg", "0", "label_0"]:
        return "negative"

    return "unknown"


def clip_star(value):
    """별점을 1~5 범위로 고정하고 소수 둘째 자리까지 반올림합니다."""
    if pd.isna(value):
        return pd.NA
    return round(max(1.0, min(5.0, float(value))), 2)


def apply_bert_adjustment(df):
    """
    카카오 리뷰 중 실제 별점(rating)과 사전 기반 감성별점(sentiment_star)의 차이가
    BERT_DIFF_THRESHOLD 이상인 행만 BERT로 재판정합니다.

    저장되는 최종 결과에는 BERT_STAR_COL 한 열만 추가합니다.
    - 대상 아님: 기존 sentiment_star 그대로 복사
    - BERT positive: sentiment_star + BERT_ADJUST_STEP
    - BERT negative: sentiment_star - BERT_ADJUST_STEP
    - BERT unknown / 확신도 낮음 / 에러: 기존 sentiment_star 유지
    """
    if BERT_STAR_COL in df.columns:
        df = df.drop(columns=[BERT_STAR_COL])

    df[BERT_STAR_COL] = df["sentiment_star"].apply(clip_star)

    required = ["platform", "rating", "sentiment_star", "diff_abs", "review_text"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"[주의] BERT 보정에 필요한 열이 없어 보정을 건너뜁니다: {missing}")
        return df

    if pipeline is None:
        raise ImportError(
            "transformers가 설치되어 있지 않습니다. 아래 명령으로 설치한 뒤 다시 실행하세요:\n"
            "pip install transformers torch tqdm"
        )

    target_mask = (
        df["platform"].astype(str).str.lower().eq("kakao")
        & df["rating"].notna()
        & df["sentiment_star"].notna()
        & df["diff_abs"].notna()
        & (df["diff_abs"] >= BERT_DIFF_THRESHOLD)
    )

    target_indices = df.index[target_mask].tolist()
    print(f"\n[BERT 보정 대상] {len(target_indices):,}건 / 전체 {len(df):,}건")

    if not target_indices:
        return df

    classifier = pipeline(
        "sentiment-analysis",
        model=BERT_MODEL_NAME,
        tokenizer=BERT_MODEL_NAME,
        device=-1,          # CPU. GPU 사용 가능하면 0으로 변경
        truncation=True,
        max_length=512,
    )

    iterator = target_indices
    if tqdm is not None:
        iterator = tqdm(target_indices, desc="BERT 보정 진행 중")

    adjusted_count = 0
    kept_low_confidence = 0
    kept_unknown = 0
    error_count = 0

    for idx in iterator:
        text = str(df.at[idx, "review_text"]).strip()

        if text == "":
            continue

        try:
            result = classifier(text[:512])[0]
            raw_label = result.get("label", "")
            bert_score = float(result.get("score", 0.0))
            clean_label = normalize_bert_label(raw_label)

            old_star = df.at[idx, "sentiment_star"]

            if clean_label == "unknown":
                kept_unknown += 1
                new_star = old_star
            elif bert_score < BERT_CONFIDENCE_THRESHOLD:
                kept_low_confidence += 1
                new_star = old_star
            elif clean_label == "positive":
                new_star = old_star + BERT_ADJUST_STEP
                adjusted_count += 1
            elif clean_label == "negative":
                new_star = old_star - BERT_ADJUST_STEP
                adjusted_count += 1
            else:
                new_star = old_star

            df.at[idx, BERT_STAR_COL] = clip_star(new_star)

        except Exception as e:
            error_count += 1
            df.at[idx, BERT_STAR_COL] = clip_star(df.at[idx, "sentiment_star"])

    print(f"[BERT 실제 보정] {adjusted_count:,}건")
    print(f"[BERT 확신도 낮아 유지] {kept_low_confidence:,}건")
    print(f"[BERT label unknown으로 유지] {kept_unknown:,}건")
    print(f"[BERT 오류로 유지] {error_count:,}건")

    return df

# =========================================================
# 8-1. sentiment_star, diff, diff_abs 계산
# =========================================================

df["sentiment_star"] = df["category_total_score"].apply(sentiment_to_star)

if "rating" in df.columns:
    df["diff"] = df["rating"] - df["sentiment_star"]
    df["diff_abs"] = df["diff"].abs()

    # rating이 없는 행에서는 diff_abs에 sentiment_star를 넣음
    df["diff_abs"] = df["diff_abs"].fillna(df["sentiment_star"])

# =========================================================
# 8-2. 카카오 mismatch 리뷰만 BERT로 감성별점 보정
# =========================================================
df = apply_bert_adjustment(df)

# =========================================================
# 9. 플랫폼별 요약
# =========================================================

if "platform" in df.columns:
    agg_dict = {
        "review_count": ("review_text", "count"),
        "avg_total_score": ("category_total_score", "mean"),
        "avg_sentiment_star": ("sentiment_star", "mean"),
        "positive_ratio": ("category_total_label", lambda x: (x == "positive").mean()),
        "neutral_ratio": ("category_total_label", lambda x: (x == "neutral").mean()),
        "negative_ratio": ("category_total_label", lambda x: (x == "negative").mean()),
    }

    if "rating" in df.columns:
        agg_dict["avg_rating"] = ("rating", "mean")
        agg_dict["avg_diff_abs"] = ("diff_abs", "mean")

    for cat in categories:
        agg_dict[f"avg_{cat}_score"] = (f"{cat}_score", "mean")

    platform_summary = df.groupby("platform").agg(**agg_dict).reset_index()

    platform_cols = (
        ["platform", "review_count"]
        + [f"avg_{cat}_score" for cat in categories]
        + ["avg_total_score", "avg_sentiment_star"]
    )

    if "rating" in df.columns:
        platform_cols += ["avg_rating", "avg_diff_abs"]

    platform_cols += ["positive_ratio", "neutral_ratio", "negative_ratio"]

    platform_summary = platform_summary[platform_cols]
    platform_summary.to_csv(platform_output, index=False, encoding="utf-8-sig")

# =========================================================
# 10. 식당별 요약
# =========================================================

if "store_name" in df.columns and "platform" in df.columns:
    agg_dict = {
        "review_count": ("review_text", "count"),
        "avg_total_score": ("category_total_score", "mean"),
        "avg_sentiment_star": ("sentiment_star", "mean"),
        "positive_ratio": ("category_total_label", lambda x: (x == "positive").mean()),
        "neutral_ratio": ("category_total_label", lambda x: (x == "neutral").mean()),
        "negative_ratio": ("category_total_label", lambda x: (x == "negative").mean()),
    }

    if "rating" in df.columns:
        agg_dict["avg_rating"] = ("rating", "mean")
        agg_dict["avg_diff_abs"] = ("diff_abs", "mean")

    for cat in categories:
        agg_dict[f"avg_{cat}_score"] = (f"{cat}_score", "mean")

    store_summary = df.groupby(["store_name", "platform"]).agg(**agg_dict).reset_index()

    store_cols = (
        ["store_name", "platform", "review_count"]
        + [f"avg_{cat}_score" for cat in categories]
        + ["avg_total_score", "avg_sentiment_star"]
    )

    if "rating" in df.columns:
        store_cols += ["avg_rating", "avg_diff_abs"]

    store_cols += ["positive_ratio", "neutral_ratio", "negative_ratio"]

    store_summary = store_summary[store_cols]
    store_summary.to_csv(store_output, index=False, encoding="utf-8-sig")

# =========================================================
# 11. 리뷰별 감성점수 확인용 파일 저장
# =========================================================

review_score_cols = [
    "platform",
    "store_name",
    "review_text",
    "tokens",
]

if "rating" in df.columns:
    review_score_cols += ["rating"]

review_score_cols += [f"{cat}_score" for cat in categories]

review_score_cols += [
    "category_total_score",
    "category_total_label",
    "sentiment_star",
]

if "rating" in df.columns:
    review_score_cols += [
        "diff",
        "diff_abs",
    ]

review_score_cols += [f"{cat}_label" for cat in categories]
review_score_cols += [f"{cat}_matched_words" for cat in categories]

review_score_cols = [col for col in review_score_cols if col in df.columns]

df[review_score_cols].to_csv(
    review_score_output,
    index=False,
    encoding="utf-8-sig",
)

# =========================================================
# 12. 전체 결과 저장
# =========================================================

# 요청사항: 최종 CSV의 맨 오른쪽에 BERT 보정 감성별점 열만 오도록 정렬
if BERT_STAR_COL in df.columns:
    other_cols = [col for col in df.columns if col != BERT_STAR_COL]
    df = df[other_cols + [BERT_STAR_COL]]

df.to_csv(output_file, index=False, encoding="utf-8-sig")

# =========================================================
# 13. 실행 결과 출력
# =========================================================

print("\n완료:", output_file)
print("리뷰별 감성점수 확인 파일:", review_score_output)

if "platform" in df.columns:
    print("플랫폼 요약:", platform_output)

if "store_name" in df.columns and "platform" in df.columns:
    print("식당 요약:", store_output)

print("\n[사용 카테고리]")
print(categories)

print("\n[사전에 들어 있는 카테고리]")
print(dict_categories)

print("\n[감성분석 결과 미리보기]")

preview_cols = [
    "review_text",
    "tokens",
]

if "rating" in df.columns:
    preview_cols += ["rating"]

preview_cols += [f"{cat}_score" for cat in categories]

preview_cols += [
    "category_total_score",
    "category_total_label",
    "sentiment_star",
    BERT_STAR_COL,
]

if "rating" in df.columns:
    preview_cols += [
        "diff",
        "diff_abs",
    ]

preview_cols += [f"{cat}_matched_words" for cat in categories]

preview_cols = [col for col in preview_cols if col in df.columns]

print(df[preview_cols].head(20).to_string())