
그다음 바로 복붙해서 실행:

```python
import re
import time
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# =========================
# 설정
# =========================
URL = "https://place.map.kakao.com/27306859#review"
CUTOFF_DATE = datetime(2024, 1, 1)
OUTFILE = "kakao_reviews_27306859.csv"
HEADLESS = False
CLICK_MORE_MAX = 300
WAIT_SEC = 12


# =========================
# 셀렉터 후보들
# =========================
REVIEW_ITEM_SELECTORS = [
    "ul.list_evaluation > li",
    ".list_evaluation > li",
    "ul[class*='evaluation'] > li",
    "ul[class*='review'] > li",
    ".review_list > li",
    "div[class*='review'] ul > li",
]

USER_NAME_SELECTORS = [
    ".link_user",
    ".txt_user",
    ".name_user",
    ".txt_name",
    "a[href*='/profile/']",
]

PROFILE_LINK_SELECTORS = [
    "a[href*='/profile/']",
    ".link_user[href]",
    ".txt_user[href]",
]

REVIEW_TEXT_SELECTORS = [
    ".txt_comment span",
    ".txt_comment",
    ".desc_review",
    ".review_cont",
    ".txt_evaluation",
    ".wrap_comment",
]

KEYWORD_SELECTORS = [
    ".list_keyword span",
    ".wrap_keyword span",
    "[class*='keyword'] span",
    "[class*='keyword'] a",
    "[class*='keyword'] li",
]

PHOTO_SELECTORS = [
    ".list_photo img",
    ".wrap_photo img",
    ".thumb_photo img",
    "[class*='photo'] img",
    ".img_review",
]

STAR_SELECTORS = [
    ".grade_star em",
    ".num_rate",
    "[class*='star'] em",
    "[class*='rate'] em",
]

REVIEW_TAB_XPATHS = [
    "//a[contains(@href, '#review')]",
    "//a[contains(., '후기')]",
    "//a[contains(., '리뷰')]",
    "//button[contains(., '후기')]",
    "//button[contains(., '리뷰')]",
]

LATEST_SORT_XPATHS = [
    "//*[self::a or self::button][contains(., '최신순')]",
    "//*[self::a or self::button][contains(., '최근순')]",
]

MORE_BUTTON_XPATHS = [
    "//*[self::a or self::button][contains(., '후기 더보기')]",
    "//*[self::a or self::button][contains(., '리뷰 더보기')]",
    "//*[self::a or self::button][contains(., '더보기')]",
]


# =========================
# 유틸
# =========================
def clean_text(s: str, keep_newline: bool = False) -> str:
    if s is None:
        return ""
    s = s.replace("\xa0", " ").replace("\u200b", " ")
    if keep_newline:
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n+", "\n", s)
        return s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_driver(headless: bool = False):
    options = Options()
    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--lang=ko-KR")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.implicitly_wait(2)
    return driver


def get_best_elements(scope, selectors):
    best = []
    for sel in selectors:
        try:
            found = scope.find_elements(By.CSS_SELECTOR, sel)
            found = [x for x in found if x.is_displayed()]
            if len(found) > len(best):
                best = found
        except Exception:
            pass
    return best


def get_first_element(scope, selectors):
    for sel in selectors:
        try:
            els = scope.find_elements(By.CSS_SELECTOR, sel)
            for el in els:
                if el.is_displayed():
                    return el
        except Exception:
            pass
    return None


def get_first_text(scope, selectors):
    el = get_first_element(scope, selectors)
    if el:
        return clean_text(el.text)
    return ""


def get_first_attr(scope, selectors, attr_name):
    el = get_first_element(scope, selectors)
    if el:
        try:
            return el.get_attribute(attr_name) or ""
        except Exception:
            return ""
    return ""


def click_any_xpath(driver, xpaths, wait_sec=WAIT_SEC):
    for xp in xpaths:
        try:
            els = driver.find_elements(By.XPATH, xp)
            for el in els:
                if el.is_displayed():
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
                    time.sleep(0.3)
                    driver.execute_script("arguments[0].click();", el)
                    return True
        except Exception:
            pass
    return False


def open_review_tab(driver):
    time.sleep(2)
    click_any_xpath(driver, REVIEW_TAB_XPATHS)
    time.sleep(1.5)


def click_latest_sort(driver):
    click_any_xpath(driver, LATEST_SORT_XPATHS)
    time.sleep(1)


def find_review_items(driver):
    items = get_best_elements(driver, REVIEW_ITEM_SELECTORS)
    # 텍스트가 거의 없는 li 제거
    filtered = []
    for item in items:
        try:
            txt = clean_text(item.text, keep_newline=True)
            if txt:
                filtered.append(item)
        except StaleElementReferenceException:
            continue
    return filtered


def parse_visit_date(text: str):
    if not text:
        return None

    text = clean_text(text, keep_newline=True)

    patterns = [
        r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일",
        r"(\d{4})[./-]\s*(\d{1,2})[./-]\s*(\d{1,2})",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            y, mth, d = map(int, m.groups())
            try:
                return datetime(y, mth, d)
            except ValueError:
                return None
    return None


def extract_visit_count(text: str) -> str:
    m = re.search(r"(\d+)\s*번째\s*방문", text)
    return m.group(1) if m else ""


def extract_auth_method(lines) -> str:
    for i, line in enumerate(lines):
        if line in ["인증수단", "인증 수단", "인증"]:
            if i + 1 < len(lines):
                return lines[i + 1]
    joined = "\n".join(lines)
    m = re.search(r"인증(?:\s*수단)?\s*[:：]?\s*\n?\s*([^\n]+)", joined)
    return clean_text(m.group(1)) if m else ""


def extract_keywords(card, lines) -> str:
    kws = []
    for sel in KEYWORD_SELECTORS:
        try:
            els = card.find_elements(By.CSS_SELECTOR, sel)
            for el in els:
                t = clean_text(el.text)
                if t and t not in kws:
                    kws.append(t)
        except Exception:
            pass

    if kws:
        return " | ".join(kws)

    # 셀렉터 실패 시 텍스트 기반 보조 파싱
    for i, line in enumerate(lines):
        if line == "키워드" or line.startswith("키워드"):
            tmp = []
            for j in range(i + 1, min(i + 8, len(lines))):
                lj = lines[j]
                if any(x in lj for x in ["방문", "인증", "별점", "후기", "리뷰", "사진"]):
                    break
                if 1 <= len(lj) <= 20:
                    tmp.append(lj)
            if tmp:
                return " | ".join(tmp)

    return ""


def extract_review_text(card, lines) -> str:
    text = get_first_text(card, REVIEW_TEXT_SELECTORS)
    if text:
        return text

    # 보조: 가장 긴 문장을 리뷰로 추정
    bad_tokens = ["방문일", "인증", "영수증", "예약", "후기", "리뷰", "별점", "레벨", "Lv", "키워드"]
    candidates = [x for x in lines if len(x) >= 5 and not any(tok in x for tok in bad_tokens)]
    if candidates:
        return max(candidates, key=len)
    return ""


def extract_review_star(card, raw_text: str) -> str:
    text = get_first_text(card, STAR_SELECTORS)
    if text:
        m = re.search(r"([0-5](?:\.\d+)?)", text)
        if m:
            return m.group(1)

    m = re.search(r"별점\s*([0-5](?:\.\d+)?)", raw_text)
    return m.group(1) if m else ""


def extract_account_review_count(raw_text: str) -> str:
    patterns = [
        r"(?:후기|리뷰)\s*(\d+)",
        r"(\d+)\s*(?:개의\s*)?(?:후기|리뷰)",
    ]
    for p in patterns:
        m = re.search(p, raw_text)
        if m:
            return m.group(1)
    return ""


def extract_account_avg_star(raw_text: str) -> str:
    patterns = [
        r"평균\s*별점\s*([0-5](?:\.\d+)?)",
        r"별점\s*평균\s*([0-5](?:\.\d+)?)",
        r"평균평점\s*([0-5](?:\.\d+)?)",
    ]
    for p in patterns:
        m = re.search(p, raw_text)
        if m:
            return m.group(1)
    return ""


def extract_level(raw_text: str) -> str:
    patterns = [
        r"(Lv\.?\s*\d+)",
        r"(LEVEL\s*\d+)",
        r"(레벨\s*\d+)",
    ]
    for p in patterns:
        m = re.search(p, raw_text, re.IGNORECASE)
        if m:
            return clean_text(m.group(1))
    return ""


def extract_account_id(card, lines) -> str:
    user = get_first_text(card, USER_NAME_SELECTORS)
    if user:
        return user

    # 보조: 첫 줄을 닉네임으로 추정
    if lines:
        first = lines[0]
        if not any(x in first for x in ["방문", "인증", "후기", "리뷰", "별점", "키워드"]):
            return first
    return ""


def extract_profile_url(card) -> str:
    href = get_first_attr(card, PROFILE_LINK_SELECTORS, "href")
    if href:
        return urljoin("https://place.map.kakao.com", href)
    return ""


def has_review_photo(card) -> str:
    for sel in PHOTO_SELECTORS:
        try:
            if card.find_elements(By.CSS_SELECTOR, sel):
                return "Y"
        except Exception:
            pass
    return "N"


def scrape_profile_stats(driver, profile_url: str):
    # 프로필 접근 실패해도 전체 수집은 계속 진행
    result = {
        "계정의 리뷰 수": "",
        "계정의 별점 평균": "",
        "카카오맵 리뷰다는 사람 레벨": "",
    }

    if not profile_url:
        return result

    current = driver.current_window_handle

    try:
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])
        driver.get(profile_url)
        time.sleep(2.5)

        body_text = clean_text(driver.find_element(By.TAG_NAME, "body").text, keep_newline=True)

        rc = extract_account_review_count(body_text)
        avg = extract_account_avg_star(body_text)
        lvl = extract_level(body_text)

        if rc:
            result["계정의 리뷰 수"] = rc
        if avg:
            result["계정의 별점 평균"] = avg
        if lvl:
            result["카카오맵 리뷰다는 사람 레벨"] = lvl

    except Exception:
        pass
    finally:
        try:
            driver.close()
        except Exception:
            pass
        try:
            driver.switch_to.window(current)
        except Exception:
            pass

    return result


def expand_reviews_until_cutoff(driver):
    prev_count = -1
    stagnant = 0

    for _ in range(CLICK_MORE_MAX):
        items = find_review_items(driver)
        cur_count = len(items)

        if cur_count == 0:
            time.sleep(1)
            continue

        # 마지막 리뷰 날짜가 cutoff 이전이면 중단
        try:
            last_text = clean_text(items[-1].text, keep_newline=True)
            last_dt = parse_visit_date(last_text)
            if last_dt and last_dt < CUTOFF_DATE:
                break
        except Exception:
            pass

        if cur_count == prev_count:
            stagnant += 1
        else:
            stagnant = 0

        if stagnant >= 3:
            break

        clicked = click_any_xpath(driver, MORE_BUTTON_XPATHS)
        if not clicked:
            # 스크롤 한 번 더 내려보고 재시도
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            clicked = click_any_xpath(driver, MORE_BUTTON_XPATHS)

        if not clicked:
            break

        prev_count = cur_count
        time.sleep(1.5)


# =========================
# 메인
# =========================
driver = build_driver(HEADLESS)
wait = WebDriverWait(driver, WAIT_SEC)

try:
    driver.get(URL)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(2)

    open_review_tab(driver)
    click_latest_sort(driver)
    expand_reviews_until_cutoff(driver)

    review_items = find_review_items(driver)
    print(f"[INFO] 화면상 리뷰 카드 수: {len(review_items)}")

    rows = []
    profile_cache = {}

    for idx, card in enumerate(review_items, start=1):
        try:
            raw_text = clean_text(card.text, keep_newline=True)
            lines = [clean_text(x) for x in raw_text.split("\n") if clean_text(x)]

            visit_date_dt = parse_visit_date(raw_text)
            if visit_date_dt and visit_date_dt < CUTOFF_DATE:
                continue

            account_id = extract_account_id(card, lines)
            profile_url = extract_profile_url(card)

            account_review_count = extract_account_review_count(raw_text)
            account_avg_star = extract_account_avg_star(raw_text)
            user_level = extract_level(raw_text)

            # 카드에서 안 잡히는 값은 프로필 페이지에서 보조 수집
            if profile_url and (not account_review_count or not account_avg_star or not user_level):
                if profile_url not in profile_cache:
                    profile_cache[profile_url] = scrape_profile_stats(driver, profile_url)

                prof = profile_cache[profile_url]
                if not account_review_count:
                    account_review_count = prof.get("계정의 리뷰 수", "")
                if not account_avg_star:
                    account_avg_star = prof.get("계정의 별점 평균", "")
                if not user_level:
                    user_level = prof.get("카카오맵 리뷰다는 사람 레벨", "")

            review_text = extract_review_text(card, lines)
            keywords = extract_keywords(card, lines)
            visit_count = extract_visit_count(raw_text)
            auth_method = extract_auth_method(lines)
            review_star = extract_review_star(card, raw_text)
            photo_yn = has_review_photo(card)

            row = {
                "계정 ID": account_id,
                "계정의 리뷰 수": account_review_count,
                "방문 날짜": visit_date_dt.strftime("%Y-%m-%d") if visit_date_dt else "",
                "방문 횟수": visit_count,
                "인증 수단": auth_method,
                "키워드": keywords,
                "리뷰 내용": review_text,
                "리뷰 내 사진 유무": photo_yn,
                "별점": review_star,
                "계정의 별점 평균": account_avg_star,
                "카카오맵 리뷰다는 사람 레벨": user_level,
                "리뷰 글자 수": len(review_text),
            }
            rows.append(row)

            if idx % 20 == 0:
                print(f"[INFO] 처리 중... {idx}개 카드 확인")

        except Exception as e:
            print(f"[WARN] 카드 파싱 실패: {e}")
            continue

    df = pd.DataFrame(rows)

    # 중복 제거
    if not df.empty:
        df = df.drop_duplicates(
            subset=["계정 ID", "방문 날짜", "리뷰 내용"],
            keep="first"
        ).reset_index(drop=True)

    df.to_csv(OUTFILE, index=False, encoding="utf-8-sig")

    print(f"[DONE] 저장 완료: {len(df)}개 -> {OUTFILE}")
    print(df.head(10))

finally:
    driver.quit()