import re
import json
import pandas as pd
from pathlib import Path
from collections import Counter

# =========================================================
# 0. 설정
# =========================================================

INPUT_REVIEW_FILE = "all_크롤링.csv"
CUSTOM_DICT_FILE = "aspect_sentiment_dictionary_final_updated.csv"
KNU_DICT_FILE = "SentiWord_info.json"

OUTPUT_DIR = Path("final_outputs_context_readable")
OUTPUT_DIR.mkdir(exist_ok=True)

ASPECT_SAMPLE_DIR = OUTPUT_DIR / "aspect_samples"
ASPECT_TERM_DIR = OUTPUT_DIR / "aspect_terms"
ASPECT_SAMPLE_DIR.mkdir(exist_ok=True)
ASPECT_TERM_DIR.mkdir(exist_ok=True)

REVIEW_COL = "review_text"
RESTAURANT_COL = "store_name"
PLATFORM_COL = "platform"

SAMPLE_SIZE = 1000
RANDOM_STATE = 42

USE_KNU_WORD_ROOT = True
KNU_ROOT_WEIGHT = 0.65
USE_EXTRA_CUSTOM_TERMS = True
CUSTOM_WEIGHTS = [0.3, 0.5, 1.0]
REPRESENTATIVE_CUSTOM_WEIGHT = 0.5

# filter5.py 결과가 같은 폴더에 있으면 trust_weight까지 결합함
USE_TRUST_WEIGHT_IF_AVAILABLE = True
TRUST_RESULT_FILE = Path("filter_result_v19_trust_level_eval") / "all_filter_result.csv"


# =========================================================
# 1. 전처리 함수
# =========================================================

POS_EMOJIS = [
    "😀", "😃", "😄", "😁", "😆", "😊", "🙂", "😍", "🥰", "😘",
    "😋", "😎", "👍", "👏", "🙌", "❤️", "❤", "♥", "♡",
    "💕", "💖", "💗", "💓", "💞", "💘", "💝", "💟", "❣️",
    "💛", "💚", "💙", "💜", "🧡", "🖤", "🤍", "🤎", "💯", "🔥", "✨"
]
NEG_EMOJIS = ["😞", "😔", "😟", "😢", "😭", "😡", "😠", "🤬", "👎", "💔", "😩", "😫", "😤"]
LAUGH_EMOJIS = ["😂", "🤣"]


def basic_clean(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4}\b", " ", text)
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def replace_emojis(text):
    for emoji in POS_EMOJIS:
        text = text.replace(emoji, " EMO_POS ")
    for emoji in NEG_EMOJIS:
        text = text.replace(emoji, " EMO_NEG ")
    for emoji in LAUGH_EMOJIS:
        text = text.replace(emoji, " EMO_LAUGH ")
    return text


def normalize_korean_emotion_tokens(text):
    text = re.sub(r"ㅎ{2,}", " ㅎㅎ ", text)
    text = re.sub(r"ㅋ{2,}", " ㅋㅋ ", text)
    text = re.sub(r"ㅠ{2,}", " ㅠㅠ ", text)
    text = re.sub(r"ㅜ{2,}", " ㅜㅜ ", text)
    return text


def normalize_punctuation(text):
    text = re.sub(r"!{2,}", " EXCLAM ", text)
    text = re.sub(r"\?{2,}", " QUESTION ", text)
    text = re.sub(r"~{2,}", " TILDE ", text)
    text = text.replace("!", " EXCLAM ")
    text = text.replace("?", " QUESTION ")
    return text


def normalize_repeated_chars(text):
    return re.sub(r"([가-힣a-zA-Z])\1{2,}", r"\1\1", text)


def remove_unnecessary_symbols(text):
    text = re.sub(r"[^가-힣a-zA-Z0-9\s_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentiment_preprocess(text):
    text = basic_clean(text)
    text = replace_emojis(text)
    text = normalize_korean_emotion_tokens(text)
    text = normalize_punctuation(text)
    text = normalize_repeated_chars(text)
    text = remove_unnecessary_symbols(text)
    return text


def extract_sentiment_preprocess_features(text):
    if pd.isna(text):
        text = ""
    text = str(text)
    return pd.Series({
        "has_emo_pos": int(any(e in text for e in POS_EMOJIS)),
        "has_emo_neg": int(any(e in text for e in NEG_EMOJIS)),
        "has_laugh": int(any(e in text for e in LAUGH_EMOJIS)),
        "has_cry": int(bool(re.search(r"ㅠ{2,}|ㅜ{2,}", text))),
        "has_exclamation": int("!" in text),
        "has_question": int("?" in text),
        "has_number": int(bool(re.search(r"\d", text))),
        "original_char_len": len(text),
    })


# =========================================================
# 2. Aspect 후보/빈출 표현 추출
# =========================================================

ASPECT_KEYWORDS = {
    "food": ["맛", "맛있", "맛없", "메뉴", "음식", "요리", "양", "식감", "간", "재료", "고기", "면", "밥", "국물", "소스", "튀김", "디저트", "커피", "반찬", "구성", "파스타", "피자", "스테이크", "샐러드", "빵", "김치", "냉면", "평냉", "감자탕", "보쌈", "잡내", "느끼", "질기", "짜", "싱겁", "맵"],
    "service": ["서비스", "친절", "불친절", "응대", "직원", "사장님", "알바", "주문", "서빙", "안내", "대처", "설명", "접객", "태도", "구워주", "사과", "늦게", "웨이팅", "대기", "예약", "빨리"],
    "price": ["가격", "가성비", "비싸", "비쌈", "저렴", "싸다", "값", "금액", "돈", "원", "가격대", "할인", "비용", "양 대비", "돈 아깝", "아깝", "착하고"],
    "atmosphere": ["분위기", "인테리어", "매장", "내부", "공간", "자리", "테이블", "청결", "위생", "음악", "조명", "깔끔", "조용", "시끄럽", "화장실", "쾌적", "차분", "넓", "좁", "데이트", "혼밥", "주차", "위치"],
}
ASPECT_NAMES_KR = {"food": "음식", "service": "서비스", "price": "가격", "atmosphere": "분위기"}

STOPWORDS = {
    "그리고", "근데", "그런데", "하지만", "그냥", "먹었어요", "먹었습니다", "갔어요", "갔습니다",
    "있어요", "있습니다", "같아요", "합니다", "했어요", "했습니다", "되어", "있는", "없는",
    "EXCLAM", "QUESTION", "TILDE", "EMO_NEG", "EMO_LAUGH",
}


def contains_any_keyword(text, keywords):
    if pd.isna(text):
        return False
    text = str(text)
    return any(keyword in text for keyword in keywords)


def get_matched_keywords(text, keywords):
    if pd.isna(text):
        return ""
    text = str(text)
    return ", ".join([keyword for keyword in keywords if keyword in text])


def make_aspect_samples(df):
    print("\n[2단계] Aspect별 후보 리뷰 추출 시작")
    summary = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        aspect_kr = ASPECT_NAMES_KR[aspect]
        mask = df["review_sentiment"].apply(lambda x: contains_any_keyword(x, keywords))
        aspect_df = df[mask].copy()
        aspect_df[f"{aspect_kr}_matched_keywords"] = aspect_df["review_sentiment"].apply(lambda x: get_matched_keywords(x, keywords))
        total_count = len(aspect_df)
        sample_df = aspect_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy() if total_count > SAMPLE_SIZE else aspect_df.copy()
        output_file = ASPECT_SAMPLE_DIR / f"{aspect}_sample_reviews.csv"
        sample_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        summary.append({"aspect": aspect, "aspect_kr": aspect_kr, "matched_review_count": total_count, "sample_count": len(sample_df), "output_file": str(output_file)})
        print(f"[완료] {aspect_kr}: 후보 {total_count:,}개 중 {len(sample_df):,}개 저장")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(ASPECT_SAMPLE_DIR / "aspect_sample_summary.csv", index=False, encoding="utf-8-sig")
    return summary_df


def simple_tokenize(text):
    if pd.isna(text):
        return []
    text = str(text)
    text = re.sub(r"[^가-힣a-zA-Z0-9_ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [token for token in text.split() if len(token) >= 2 and token not in STOPWORDS]


def make_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def save_counter(counter, output_file, top_n=300):
    pd.DataFrame([{"term": term, "count": count} for term, count in counter.most_common(top_n)]).to_csv(output_file, index=False, encoding="utf-8-sig")


def extract_terms_by_aspect():
    print("\n[3단계] Aspect별 빈출 표현 추출 시작")
    for aspect in ASPECT_KEYWORDS.keys():
        input_file = ASPECT_SAMPLE_DIR / f"{aspect}_sample_reviews.csv"
        if not input_file.exists():
            print(f"[경고] 파일 없음: {input_file}")
            continue
        df = pd.read_csv(input_file, encoding="utf-8-sig")
        if "review_sentiment" not in df.columns:
            print(f"[경고] review_sentiment 컬럼 없음: {input_file}")
            continue
        unigram_counter, bigram_counter, trigram_counter = Counter(), Counter(), Counter()
        for text in df["review_sentiment"].dropna():
            tokens = simple_tokenize(text)
            unigram_counter.update(tokens)
            bigram_counter.update(make_ngrams(tokens, 2))
            trigram_counter.update(make_ngrams(tokens, 3))
        save_counter(unigram_counter, ASPECT_TERM_DIR / f"{aspect}_unigram_top.csv")
        save_counter(bigram_counter, ASPECT_TERM_DIR / f"{aspect}_bigram_top.csv")
        save_counter(trigram_counter, ASPECT_TERM_DIR / f"{aspect}_trigram_top.csv")
        print(f"[완료] {aspect}: unigram / bigram / trigram 저장")


# =========================================================
# 3. 파일/사전 로드
# =========================================================

def read_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return pd.DataFrame(data["data"])
            return pd.DataFrame([data])
        raise ValueError("JSON 구조를 DataFrame으로 변환할 수 없습니다.")
    raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")


def find_column(df, candidates):
    cols = list(df.columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    normalized = {str(col).strip().lower(): col for col in cols}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


BAD_ROOT_TERMS = {"것", "그", "이", "저", "나", "너", "수", "데", "듯", "때", "가", "간", "게", "함", "되", "하", "있", "없", "내", "더", "곳", "중", "못", "안", "잘", "및", "등"}


def clean_term(term):
    if pd.isna(term):
        return ""
    term = str(term).strip()
    return re.sub(r"\s+", " ", term)


def is_valid_knu_term(term):
    return not (term == "" or term.lower() == "nan" or len(term) < 2 or term in BAD_ROOT_TERMS)


def load_knu_dictionary(path):
    df = read_file(path)
    df.columns = df.columns.astype(str).str.strip()
    print("\n[확인] KNU 사전 컬럼:", list(df.columns))
    word_col = find_column(df, ["word", "term", "ngram", "단어", "표현", "SentiWord"])
    root_col = find_column(df, ["word_root", "root", "어근"])
    polarity_col = find_column(df, ["polarity", "score", "value", "감성점수", "감성값", "max.value", "MaxValue"])
    if word_col is None:
        raise ValueError(f"KNU 사전에서 word 컬럼을 찾지 못했습니다. 현재 컬럼: {list(df.columns)}")
    if polarity_col is None:
        raise ValueError(f"KNU 사전에서 polarity 컬럼을 찾지 못했습니다. 현재 컬럼: {list(df.columns)}")
    rows = []
    for _, row in df.iterrows():
        polarity = pd.to_numeric(row[polarity_col], errors="coerce")
        if pd.isna(polarity) or int(polarity) == 0:
            continue
        polarity = int(polarity)
        word = clean_term(row[word_col])
        if is_valid_knu_term(word):
            rows.append({"term": word, "polarity": polarity, "source": "knu_word", "weight": 1.0, "is_root": False})
        if USE_KNU_WORD_ROOT and root_col is not None:
            root = clean_term(row[root_col])
            if is_valid_knu_term(root):
                rows.append({"term": root, "polarity": polarity, "source": "knu_word_root", "weight": KNU_ROOT_WEIGHT, "is_root": True})
    knu = pd.DataFrame(rows)
    if len(knu) == 0:
        raise ValueError("KNU 사전에서 사용할 수 있는 항목이 없습니다.")
    knu["term_len"] = knu["term"].str.len()
    knu["abs_polarity"] = knu["polarity"].abs()
    knu["source_priority"] = knu["source"].map({"knu_word": 1, "knu_word_root": 2})
    knu = knu.sort_values(by=["term", "source_priority", "abs_polarity", "term_len"], ascending=[True, True, False, False])
    knu = knu.drop_duplicates(subset=["term"], keep="first")
    knu = knu.sort_values("term_len", ascending=False)
    print(f"[확인] KNU 감성사전 로드 완료: {len(knu):,}개")
    print(knu["source"].value_counts())
    return knu[["term", "polarity", "source", "weight", "is_root"]]


EXTRA_CUSTOM_TERMS = [
    {"term": "굿", "aspect": "food", "polarity": 1, "source": "extra_zero_fix"},
    {"term": "굳", "aspect": "food", "polarity": 1, "source": "extra_zero_fix"},
    {"term": "잘먹었어요", "aspect": "food", "polarity": 1, "source": "extra_zero_fix"},
    {"term": "잘 먹었어요", "aspect": "food", "polarity": 1, "source": "extra_zero_fix"},
    {"term": "잘먹었습니다", "aspect": "food", "polarity": 1, "source": "extra_zero_fix"},
    {"term": "잘 먹었습니다", "aspect": "food", "polarity": 1, "source": "extra_zero_fix"},
    {"term": "맛잇", "aspect": "food", "polarity": 1, "source": "extra_zero_fix"},
    {"term": "쩔어요", "aspect": "food", "polarity": 2, "source": "extra_zero_fix"},
    {"term": "쩔어", "aspect": "food", "polarity": 2, "source": "extra_zero_fix"},
    {"term": "예술입니다", "aspect": "food", "polarity": 2, "source": "extra_zero_fix"},
    {"term": "예술이에요", "aspect": "food", "polarity": 2, "source": "extra_zero_fix"},
    {"term": "또 갈", "aspect": "service", "polarity": 1, "source": "extra_zero_fix"},
    {"term": "재방문", "aspect": "service", "polarity": 1, "source": "extra_zero_fix"},
    {"term": "EMO_POS", "aspect": "atmosphere", "polarity": 1, "source": "extra_zero_fix"},
]


def load_custom_dictionary(path):
    df = read_file(path)
    df.columns = df.columns.astype(str).str.strip()
    print("\n[확인] 보완사전 컬럼:", list(df.columns))
    required = ["term", "aspect", "polarity"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"보완 감성사전에 필요한 컬럼이 없습니다: {missing}")
    custom = df[required].copy()
    custom["term"] = custom["term"].astype(str).str.strip()
    custom["aspect"] = custom["aspect"].astype(str).str.strip()
    custom["polarity"] = pd.to_numeric(custom["polarity"], errors="coerce").fillna(0).astype(int)
    custom["source"] = "manual_custom_dict"
    custom = custom[(custom["term"] != "") & (custom["term"].str.lower() != "nan") & (custom["polarity"] != 0) & (custom["aspect"].isin(["food", "price", "service", "atmosphere"]))].copy()
    if USE_EXTRA_CUSTOM_TERMS:
        custom = pd.concat([custom, pd.DataFrame(EXTRA_CUSTOM_TERMS)], ignore_index=True)
    custom = custom.drop_duplicates(subset=["term", "aspect"], keep="first")
    custom["term_len"] = custom["term"].str.len()
    custom = custom.sort_values("term_len", ascending=False)
    print(f"[확인] 보완 감성사전 로드 완료: {len(custom):,}개")
    print(custom["source"].value_counts())
    return custom[["term", "aspect", "polarity", "source"]]


# =========================================================
# 4. 문맥 보정 + 겹침 제거
# =========================================================

INTENSIFIERS = ["진짜", "정말", "너무", "완전", "엄청", "매우", "되게", "겁나", "짱", "대박", "핵", "개", "존", "미친", "미쳤"]
DIMINISHERS = ["조금", "살짝", "약간", "좀", "그냥", "그럭저럭"]
NEGATORS = ["않", "안 ", "아니", "못", "별로", "그닥", "그다지"]
CONTRAST_WORDS = ["하지만", "근데", "그런데", "다만", "그래도", "그러나", "반면"]
SENTENCE_BOUNDARY_PATTERN = r"[.!?。~]|EXCLAM|QUESTION|TILDE"
POSITIVE_WITHOUT_PATTERNS = ["잡내 없이", "잡내없이", "잡내 안", "잡내가 안", "잡내도 안", "냄새 없이", "냄새없이", "비린내 없이", "비린내없이", "웨이팅 없이", "웨이팅없이", "대기 없이", "대기없이", "기다림 없이", "기다림없이", "불편함 없이", "불편함없이", "부담 없이", "부담없이", "자극적이지 않", "자극적이지않"]
NEGATION_EXCEPTION_POSITIVE = ["나쁘지 않", "나쁘지않", "불친절하지 않", "불친절하지않", "비싸지 않", "비싸지않", "부담스럽지 않", "부담스럽지않", "짜지 않", "짜지않", "느끼하지 않", "느끼하지않", "질기지 않", "질기지않", "맵지 않", "맵지않", "싱겁지 않", "싱겁지않"]


def get_clause_bounds(text, start, end):
    left_text = text[:start]
    right_text = text[end:]
    left_bound = 0
    for match in re.finditer(SENTENCE_BOUNDARY_PATTERN, left_text):
        left_bound = match.end()
    right_bound = len(text)
    right_match = re.search(SENTENCE_BOUNDARY_PATTERN, right_text)
    if right_match:
        right_bound = end + right_match.start()
    return left_bound, right_bound


def root_match_allowed(text, start, end, term):
    after = text[end:end + 4]
    if after == "" or after[0].isspace():
        return True
    allowed_suffix_starts = ["아", "어", "었", "았", "고", "게", "네", "죠", "지", "긴", "은", "는", "을", "를", "면", "며", "서", "던", "듯", "음", "습니다", "어요", "아요", "네요", "지만", "지는", "진", "더라", "구", "용", "여", "데", "더니"]
    return any(after.startswith(suf) for suf in allowed_suffix_starts)


def get_context_multiplier(text, term_start, term_end, base_polarity):
    text = "" if pd.isna(text) else str(text)
    clause_start, clause_end = get_clause_bounds(text, term_start, term_end)
    left = text[max(clause_start, term_start - 20):term_start]
    right = text[term_end:min(clause_end, term_end + 20)]
    window = text[max(clause_start, term_start - 20):min(clause_end, term_end + 20)]
    clause = text[clause_start:clause_end]
    multiplier = 1.0

    if any(word in left for word in INTENSIFIERS):
        multiplier *= 1.5
    if any(word in left for word in DIMINISHERS):
        multiplier *= 0.5

    if any(pattern in window for pattern in NEGATION_EXCEPTION_POSITIVE):
        return abs(multiplier) * -1 if base_polarity < 0 else abs(multiplier)

    if any(pattern in window for pattern in POSITIVE_WITHOUT_PATTERNS):
        return abs(multiplier) * -1 if base_polarity < 0 else abs(multiplier)

    negation_window = right[:12] + left[-6:]
    if any(neg in negation_window for neg in NEGATORS):
        multiplier *= -1

    term_pos_in_clause = term_start - clause_start
    term_end_in_clause = term_end - clause_start

    nearest_left_contrast = -1
    for contrast in CONTRAST_WORDS:
        pos = clause.rfind(contrast, 0, term_pos_in_clause)
        if pos > nearest_left_contrast:
            nearest_left_contrast = pos
    if nearest_left_contrast != -1 and term_pos_in_clause - nearest_left_contrast <= 50:
        multiplier *= 1.3

    nearest_right_contrast = None
    for contrast in CONTRAST_WORDS:
        pos = clause.find(contrast, term_end_in_clause)
        if pos != -1 and (nearest_right_contrast is None or pos < nearest_right_contrast):
            nearest_right_contrast = pos
    if nearest_right_contrast is not None and nearest_right_contrast - term_end_in_clause <= 50:
        multiplier *= 0.7

    return multiplier


def find_term_matches(text, term, is_root=False):
    if not term:
        return []
    matches = []
    for match in re.finditer(re.escape(term), text):
        start, end = match.start(), match.end()
        if is_root and not root_match_allowed(text, start, end, term):
            continue
        matches.append((start, end))
    return matches


def spans_overlap(span1, span2):
    s1, e1 = span1
    s2, e2 = span2
    return not (e1 <= s2 or e2 <= s1)


def select_non_overlapping_candidates(candidates):
    sorted_candidates = sorted(candidates, key=lambda x: (-x["term_len"], x["priority"], x["start"]))
    selected = []
    selected_spans = []
    for cand in sorted_candidates:
        span = (cand["start"], cand["end"])
        if any(spans_overlap(span, selected_span) for selected_span in selected_spans):
            continue
        selected.append(cand)
        selected_spans.append(span)
    return sorted(selected, key=lambda x: x["start"])


def score_review_with_context(text, knu_df, custom_df):
    text = "" if pd.isna(text) else str(text)
    candidates = []

    for term, aspect, polarity, source in custom_df[["term", "aspect", "polarity", "source"]].itertuples(index=False):
        if term not in text:
            continue
        for start, end in find_term_matches(text, term, is_root=False):
            ctx = get_context_multiplier(text, start, end, polarity)
            adj = polarity * ctx
            candidates.append({"dict_type": "custom", "term": term, "aspect": aspect, "base_polarity": polarity, "adjusted_polarity": adj, "source": source, "term_source": "custom", "weight": 1.0, "context_multiplier": ctx, "start": start, "end": end, "term_len": len(term), "priority": 0})

    for term, polarity, source, weight, is_root in knu_df[["term", "polarity", "source", "weight", "is_root"]].itertuples(index=False):
        if term not in text:
            continue
        for start, end in find_term_matches(text, term, is_root=is_root):
            ctx = get_context_multiplier(text, start, end, polarity)
            adj = polarity * weight * ctx
            priority = 1 if source == "knu_word" else 2
            candidates.append({"dict_type": "knu", "term": term, "aspect": "", "base_polarity": polarity, "adjusted_polarity": adj, "source": source, "term_source": source, "weight": weight, "context_multiplier": ctx, "start": start, "end": end, "term_len": len(term), "priority": priority})

    selected = select_non_overlapping_candidates(candidates)
    aspects = ["food", "price", "service", "atmosphere"]
    knu_total, custom_total = 0.0, 0.0
    knu_count, custom_count = 0, 0
    knu_terms, custom_terms = [], []
    aspect_scores = {a: 0.0 for a in aspects}
    aspect_counts = {a: 0 for a in aspects}

    for cand in selected:
        if cand["dict_type"] == "knu":
            knu_total += cand["adjusted_polarity"]
            knu_count += 1
            knu_terms.append(f"{cand['term']}:{cand['base_polarity']}:{cand['term_source']}:w{round(cand['weight'], 2)}:ctx{round(cand['context_multiplier'], 2)}:adj{round(cand['adjusted_polarity'], 2)}")
        else:
            custom_total += cand["adjusted_polarity"]
            custom_count += 1
            aspect = cand["aspect"]
            if aspect in aspect_scores:
                aspect_scores[aspect] += cand["adjusted_polarity"]
                aspect_counts[aspect] += 1
            custom_terms.append(f"{cand['term']}:{cand['aspect']}:{cand['base_polarity']}:{cand['source']}:ctx{round(cand['context_multiplier'], 2)}:adj{round(cand['adjusted_polarity'], 2)}")

    result = {
        "knu_sentiment_score": knu_total / (knu_count + 1),
        "knu_total_polarity": knu_total,
        "knu_matched_count": knu_count,
        "knu_matched_terms": "; ".join(knu_terms),
        "custom_sentiment_score": custom_total / (custom_count + 1),
        "custom_total_polarity": custom_total,
        "custom_matched_count": custom_count,
        "custom_matched_terms": "; ".join(custom_terms),
        "selected_match_count": len(selected),
        "raw_candidate_count": len(candidates),
        "removed_overlap_count": len(candidates) - len(selected),
    }
    for aspect in aspects:
        result[f"{aspect}_score_raw"] = aspect_scores[aspect]
        result[f"{aspect}_matched_count"] = aspect_counts[aspect]
        result[f"{aspect}_score_norm"] = aspect_scores[aspect] / aspect_counts[aspect] if aspect_counts[aspect] > 0 else 0
    return pd.Series(result)


# =========================================================
# 5. readable 열
# =========================================================

ASPECT_KR = {"food": "음식", "price": "가격", "service": "서비스", "atmosphere": "분위기", "": "일반"}
SOURCE_KR = {"manual_custom_dict": "수동 보완사전", "extra_zero_fix": "0점 보완표현", "knu_word": "KNU 원표현", "knu_word_root": "KNU 어근", "custom": "보완사전", "word": "KNU 원표현", "word_root": "KNU 어근"}


def polarity_to_kr(value):
    try:
        value = float(value)
    except Exception:
        return "감성값 확인 필요"
    if value > 0:
        return f"긍정(+{value:g})"
    if value < 0:
        return f"부정({value:g})"
    return "중립(0)"


def context_to_kr(ctx):
    try:
        ctx = float(ctx)
    except Exception:
        return "문맥보정 없음"
    if abs(ctx - 1.0) < 1e-9:
        return "기본"
    if ctx > 1.0:
        return f"강화({ctx:g}배)"
    if 0 < ctx < 1.0:
        return f"약화({ctx:g}배)"
    if ctx < 0:
        return f"반전({ctx:g}배)"
    return "문맥보정 0"


def parse_custom_matched_terms_readable(terms):
    if pd.isna(terms) or str(terms).strip() == "":
        return ""
    readable = []
    for item in str(terms).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        term = parts[0] if len(parts) > 0 else ""
        aspect = parts[1] if len(parts) > 1 else ""
        polarity = parts[2] if len(parts) > 2 else ""
        source = parts[3] if len(parts) > 3 else ""
        ctx = None
        adj = None
        for part in parts:
            if part.startswith("ctx"):
                ctx = part.replace("ctx", "")
            elif part.startswith("adj"):
                adj = part.replace("adj", "")
        aspect_text = ASPECT_KR.get(aspect, aspect)
        source_text = SOURCE_KR.get(source, source)
        polarity_text = polarity_to_kr(polarity)
        ctx_text = context_to_kr(ctx) if ctx is not None else "기본"
        try:
            adj_text = f"반영 {float(adj):+.2f}" if adj is not None else "반영점수 없음"
        except Exception:
            adj_text = f"반영 {adj}"
        readable.append(f"{term}({aspect_text}, {polarity_text}, {source_text}, {ctx_text}, {adj_text})")
    return " / ".join(readable)


def parse_knu_matched_terms_readable(terms):
    if pd.isna(terms) or str(terms).strip() == "":
        return ""
    readable = []
    for item in str(terms).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        term = parts[0] if len(parts) > 0 else ""
        polarity = parts[1] if len(parts) > 1 else ""
        source = parts[2] if len(parts) > 2 else ""
        weight = None
        ctx = None
        adj = None
        for part in parts:
            if part.startswith("w"):
                weight = part.replace("w", "")
            elif part.startswith("ctx"):
                ctx = part.replace("ctx", "")
            elif part.startswith("adj"):
                adj = part.replace("adj", "")
        source_text = SOURCE_KR.get(source, source)
        polarity_text = polarity_to_kr(polarity)
        weight_text = f"사전가중 {weight}" if weight is not None else "사전가중 없음"
        ctx_text = context_to_kr(ctx) if ctx is not None else "기본"
        try:
            adj_text = f"반영 {float(adj):+.2f}" if adj is not None else "반영점수 없음"
        except Exception:
            adj_text = f"반영 {adj}"
        readable.append(f"{term}({source_text}, {polarity_text}, {weight_text}, {ctx_text}, {adj_text})")
    return " / ".join(readable)


def add_readable_matched_terms(df):
    df["custom_matched_terms_readable"] = df["custom_matched_terms"].apply(parse_custom_matched_terms_readable)
    df["knu_matched_terms_readable"] = df["knu_matched_terms"].apply(parse_knu_matched_terms_readable)
    return df


# =========================================================
# 6. 최종 점수/요약
# =========================================================

def add_final_scores(df):
    for weight in CUSTOM_WEIGHTS:
        col = f"final_score_w{str(weight).replace('.', '')}"
        df[col] = df["knu_sentiment_score"] + weight * df["custom_sentiment_score"]
    representative_col = f"final_score_w{str(REPRESENTATIVE_CUSTOM_WEIGHT).replace('.', '')}"
    df["final_sentiment_score"] = df[representative_col]
    return df


def classify_sentiment(score):
    if score >= 0.3:
        return "positive"
    if score <= -0.3:
        return "negative"
    return "neutral_mixed"


def classify_zero_reason(row):
    if abs(row["final_sentiment_score"]) > 1e-9:
        return "non_zero"
    if row["knu_matched_count"] == 0 and row["custom_matched_count"] == 0:
        return "no_dictionary_match"
    return "polarity_cancelled"


def add_merge_key(df, text_col):
    platform = df[PLATFORM_COL].astype(str).str.strip() if PLATFORM_COL in df.columns else pd.Series(["unknown"] * len(df), index=df.index)
    store = df[RESTAURANT_COL].astype(str).str.strip() if RESTAURANT_COL in df.columns else pd.Series(["unknown"] * len(df), index=df.index)
    text = df[text_col].astype(str).str.strip()
    return platform + "||" + store + "||" + text


def attach_trust_weight(df):
    if not USE_TRUST_WEIGHT_IF_AVAILABLE or not TRUST_RESULT_FILE.exists():
        df["trust_weight_available"] = 0
        df["trust_weight"] = 1.0
        df["trust_level"] = "not_available"
        df["trust_adjusted_sentiment_score"] = df["final_sentiment_score"]
        if USE_TRUST_WEIGHT_IF_AVAILABLE:
            print(f"[안내] trust 결과 파일이 없어 결합을 건너뜁니다: {TRUST_RESULT_FILE}")
        return df
    trust_df = pd.read_csv(TRUST_RESULT_FILE, encoding="utf-8-sig")
    required = ["review_text", "trust_weight", "trust_level", "trust_score"]
    missing = [c for c in required if c not in trust_df.columns]
    if missing:
        print(f"[경고] trust 결과 파일에 필요한 컬럼이 없습니다: {missing}")
        df["trust_weight_available"] = 0
        df["trust_weight"] = 1.0
        df["trust_level"] = "not_available"
        df["trust_adjusted_sentiment_score"] = df["final_sentiment_score"]
        return df
    df["_merge_key"] = add_merge_key(df, REVIEW_COL)
    trust_df["_merge_key"] = add_merge_key(trust_df, "review_text")
    keep = ["_merge_key", "trust_weight", "trust_level", "trust_score", "trust_reasons"]
    keep = [c for c in keep if c in trust_df.columns]
    trust_small = trust_df[keep].drop_duplicates(subset=["_merge_key"], keep="first")
    before = len(df)
    df = df.merge(trust_small, on="_merge_key", how="left")
    if len(df) != before:
        print("[경고] trust_weight 병합 후 행 수가 달라졌습니다.")
    df["trust_weight_available"] = df["trust_weight"].notna().astype(int)
    df["trust_weight"] = df["trust_weight"].fillna(1.0)
    df["trust_level"] = df["trust_level"].fillna("not_available")
    df["trust_adjusted_sentiment_score"] = df["final_sentiment_score"] * df["trust_weight"]
    df = df.drop(columns=["_merge_key"], errors="ignore")
    print("[완료] trust_weight 결합 완료")
    print(df["trust_weight_available"].value_counts())
    return df


def make_restaurant_summary(df):
    if RESTAURANT_COL not in df.columns:
        return None
    agg = {
        "review_count": (REVIEW_COL, "count"),
        "final_sentiment_mean": ("final_sentiment_score", "mean"),
        "trust_adjusted_sentiment_mean": ("trust_adjusted_sentiment_score", "mean"),
        "knu_sentiment_mean": ("knu_sentiment_score", "mean"),
        "custom_sentiment_mean": ("custom_sentiment_score", "mean"),
        "knu_matched_total": ("knu_matched_count", "sum"),
        "custom_matched_total": ("custom_matched_count", "sum"),
        "removed_overlap_total": ("removed_overlap_count", "sum"),
        "food_score_mean": ("food_score_norm", "mean"),
        "price_score_mean": ("price_score_norm", "mean"),
        "service_score_mean": ("service_score_norm", "mean"),
        "atmosphere_score_mean": ("atmosphere_score_norm", "mean"),
        "trust_weight_mean": ("trust_weight", "mean"),
    }
    for weight in CUSTOM_WEIGHTS:
        col = f"final_score_w{str(weight).replace('.', '')}"
        agg[f"{col}_mean"] = (col, "mean")
    summary = df.groupby(RESTAURANT_COL).agg(**agg).reset_index()
    summary = summary.sort_values(by=["trust_adjusted_sentiment_mean", "review_count"], ascending=[False, False])
    output = OUTPUT_DIR / "restaurant_sentiment_summary_final.csv"
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[완료] 식당별 최종 요약 저장: {output}")
    return summary


def make_platform_summary(df):
    if PLATFORM_COL not in df.columns:
        return None
    agg = {
        "review_count": (REVIEW_COL, "count"),
        "final_sentiment_mean": ("final_sentiment_score", "mean"),
        "trust_adjusted_sentiment_mean": ("trust_adjusted_sentiment_score", "mean"),
        "knu_sentiment_mean": ("knu_sentiment_score", "mean"),
        "custom_sentiment_mean": ("custom_sentiment_score", "mean"),
        "knu_matched_total": ("knu_matched_count", "sum"),
        "custom_matched_total": ("custom_matched_count", "sum"),
        "removed_overlap_total": ("removed_overlap_count", "sum"),
        "food_score_mean": ("food_score_norm", "mean"),
        "price_score_mean": ("price_score_norm", "mean"),
        "service_score_mean": ("service_score_norm", "mean"),
        "atmosphere_score_mean": ("atmosphere_score_norm", "mean"),
        "trust_weight_mean": ("trust_weight", "mean"),
    }
    for weight in CUSTOM_WEIGHTS:
        col = f"final_score_w{str(weight).replace('.', '')}"
        agg[f"{col}_mean"] = (col, "mean")
    summary = df.groupby(PLATFORM_COL).agg(**agg).reset_index()
    output = OUTPUT_DIR / "platform_sentiment_summary_final.csv"
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[완료] 플랫폼별 최종 요약 저장: {output}")
    return summary


def make_value_counts_file(df, col, filename):
    out = df[col].value_counts().reset_index()
    out.columns = [col, "count"]
    out.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
    return out


def make_custom_term_match_summary(df):
    rows = []
    for terms in df["custom_matched_terms"].dropna():
        if str(terms).strip() == "":
            continue
        for item in str(terms).split(";"):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            rows.append({
                "term": parts[0] if len(parts) > 0 else "",
                "aspect": parts[1] if len(parts) > 1 else "",
                "base_polarity": parts[2] if len(parts) > 2 else "",
                "source": parts[3] if len(parts) > 3 else "",
            })
    if not rows:
        print("[주의] 보완사전 매칭 표현이 없습니다.")
        return None
    summary = pd.DataFrame(rows).groupby(["term", "aspect", "base_polarity", "source"]).size().reset_index(name="matched_count").sort_values("matched_count", ascending=False)
    output = OUTPUT_DIR / "custom_dictionary_matched_term_counts.csv"
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[완료] 보완사전 표현별 매칭 빈도 저장: {output}")
    return summary


def make_context_summary(df):
    rows = []
    for col in ["knu_matched_terms", "custom_matched_terms"]:
        for terms in df[col].dropna():
            if str(terms).strip() == "":
                continue
            for item in str(terms).split(";"):
                match = re.search(r"ctx(-?\d+\.?\d*)", item)
                if match:
                    rows.append({"source_col": col, "ctx": float(match.group(1))})
    if not rows:
        print("[안내] 문맥 보정 정보가 없습니다.")
        return None
    summary = pd.DataFrame(rows).groupby(["source_col", "ctx"]).size().reset_index(name="count").sort_values(["source_col", "count"], ascending=[True, False])
    output = OUTPUT_DIR / "context_multiplier_summary.csv"
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"[완료] 문맥 보정 요약 저장: {output}")
    return summary


def make_zero_examples(df):
    df[df["zero_reason"] == "no_dictionary_match"].head(300).to_csv(OUTPUT_DIR / "zero_examples_no_dictionary_match.csv", index=False, encoding="utf-8-sig")
    df[df["zero_reason"] == "polarity_cancelled"].head(300).to_csv(OUTPUT_DIR / "zero_examples_polarity_cancelled.csv", index=False, encoding="utf-8-sig")
    print("[완료] 0점 예시 파일 저장")


# =========================================================
# 7. 메인 실행
# =========================================================

def main():
    print("[시작] 최종 문맥보정+readable 감성분석 파이프라인 실행")
    input_path = Path(INPUT_REVIEW_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 리뷰 파일을 찾을 수 없습니다: {INPUT_REVIEW_FILE}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"[확인] 원본 리뷰 수: {len(df):,}")
    print(f"[확인] 컬럼 목록: {list(df.columns)}")
    if REVIEW_COL not in df.columns:
        raise ValueError(f"'{REVIEW_COL}' 컬럼이 없습니다. REVIEW_COL 설정을 확인하세요.")

    print(f"[확인] 리뷰 텍스트 결측 수: {df[REVIEW_COL].isna().sum():,}")
    df[REVIEW_COL] = df[REVIEW_COL].fillna("")

    print("\n[1단계] 리뷰 전처리 시작")
    df["review_original"] = df[REVIEW_COL]
    df["review_sentiment"] = df["review_original"].apply(sentiment_preprocess)
    df = pd.concat([df, df["review_original"].apply(extract_sentiment_preprocess_features)], axis=1)
    preprocessed_file = OUTPUT_DIR / "reviews_preprocessed.csv"
    df.to_csv(preprocessed_file, index=False, encoding="utf-8-sig")
    print(f"[완료] 전처리 결과 저장: {preprocessed_file}")

    make_aspect_samples(df)
    extract_terms_by_aspect()

    print("\n[4단계] 감성사전 로드")
    knu_df = load_knu_dictionary(KNU_DICT_FILE)
    custom_df = load_custom_dictionary(CUSTOM_DICT_FILE)

    print("\n[5단계] KNU + 보완사전 + 문맥 보정 + 겹침 제거 적용 중...")
    print("※ 시간이 걸릴 수 있습니다.")
    score_df = df["review_sentiment"].apply(lambda text: score_review_with_context(text, knu_df, custom_df))
    df = pd.concat([df, score_df], axis=1)

    print("\n[6단계] 최종 점수/라벨/readable 열 생성")
    df = add_final_scores(df)
    df["final_sentiment_label"] = df["final_sentiment_score"].apply(classify_sentiment)
    df["zero_reason"] = df.apply(classify_zero_reason, axis=1)
    df = attach_trust_weight(df)
    df = add_readable_matched_terms(df)

    output_file = OUTPUT_DIR / "reviews_scored_context_readable.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"[완료] 리뷰별 최종 결과 저장: {output_file}")

    print("\n[7단계] 요약표 생성")
    restaurant_summary = make_restaurant_summary(df)
    platform_summary = make_platform_summary(df)
    label_summary = make_value_counts_file(df, "final_sentiment_label", "final_sentiment_label_summary.csv")
    zero_summary = make_value_counts_file(df, "zero_reason", "zero_reason_summary.csv")
    custom_term_summary = make_custom_term_match_summary(df)
    context_summary = make_context_summary(df)
    make_zero_examples(df)

    print("\n[완료] 전체 파이프라인 종료")
    print(f"[결과 폴더] {OUTPUT_DIR.resolve()}")
    print("\n[확인할 파일]")
    print(f"1. 리뷰별 결과: {OUTPUT_DIR / 'reviews_scored_context_readable.csv'}")
    print(f"2. 플랫폼 요약: {OUTPUT_DIR / 'platform_sentiment_summary_final.csv'}")
    print(f"3. 식당별 요약: {OUTPUT_DIR / 'restaurant_sentiment_summary_final.csv'}")
    print(f"4. 문맥 보정 요약: {OUTPUT_DIR / 'context_multiplier_summary.csv'}")
    print(f"5. 보완사전 매칭 빈도: {OUTPUT_DIR / 'custom_dictionary_matched_term_counts.csv'}")
    print(f"6. 0점 원인: {OUTPUT_DIR / 'zero_reason_summary.csv'}")

    print("\n[플랫폼별 요약 미리보기]")
    if platform_summary is not None:
        print(platform_summary)
    print("\n[감성 라벨 분포]")
    print(label_summary)
    print("\n[0점 원인 요약]")
    print(zero_summary)
    print("\n[문맥 보정 요약]")
    if context_summary is not None:
        print(context_summary.head(20))
    print("\n[보완사전 매칭 표현 상위 20개]")
    if custom_term_summary is not None:
        print(custom_term_summary.head(20))
    print("\n[식당별 요약 미리보기]")
    if restaurant_summary is not None:
        print(restaurant_summary.head(10))


if __name__ == "__main__":
    main()
