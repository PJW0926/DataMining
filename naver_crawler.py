import re
import time
import pandas as pd

from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


# =====================================
# 0. 기본 설정
# =====================================
NAVER_MAP_URL = "https://map.naver.com/p/entry/place/37126807?c=15.00,0,0,0,dh&placePath=/review?additionalHeight=76&fromPanelNum=1&locale=ko&svcName=map_pcv5"

START_DATE = datetime(2024, 1, 1)

# 수집 제어
SLEEP = 1.2
MAX_IDLE_ROUNDS = 8         # 새 리뷰가 안 늘어나는 라운드가 이 횟수 이상이면 종료
SAFETY_MAX_ROUNDS = 300     # 무한루프 방지용
OUTPUT_CSV = "naver_reviews.csv"
TEMP_CSV = "naver_reviews_temp.csv"


# =====================================
# 1. 드라이버 생성
# =====================================
def create_driver():
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    driver.implicitly_wait(2)
    return driver


# =====================================
# 2. 보조 함수
# =====================================
def safe_text(parent, by, selector, default=""):
    try:
        return parent.find_element(by, selector).text.strip()
    except Exception:
        return default


def safe_texts(parent, by, selector):
    try:
        return [e.text.strip() for e in parent.find_elements(by, selector) if e.text.strip()]
    except Exception:
        return []


def exists(parent, by, selector):
    try:
        parent.find_element(by, selector)
        return True
    except Exception:
        return False


def normalize_date_string(y, m, d):
    return f"{int(y):04d}.{int(m):02d}.{int(d):02d}"


def extract_visit_date_from_text(text):
    """
    카드 전체 텍스트에서 방문 날짜 추출
    우선순위:
    1) YYYY.MM.DD
    2) YYYY년 M월 D일
    """
    m1 = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", text)
    if m1:
        y, m, d = m1.groups()
        return normalize_date_string(y, m, d)

    m2 = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", text)
    if m2:
        y, m, d = m2.groups()
        return normalize_date_string(y, m, d)

    return ""


def is_valid_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%Y.%m.%d")
        return date_obj >= START_DATE
    except Exception:
        return False


def parse_visit_info_from_text(full_text):
    visit_date = extract_visit_date_from_text(full_text)

    visit_count = ""
    count_match = re.search(r"(\d+번째 방문)", full_text)
    if count_match:
        visit_count = count_match.group(1)

    auth_method = ""
    auth_candidates = ["영수증", "예약", "주문", "포장", "배달"]
    for candidate in auth_candidates:
        if candidate in full_text:
            auth_method = candidate
            break

    return visit_date, visit_count, auth_method


def extract_account_review_count(full_text):
    """
    예: '리뷰 106 사진 499 팔로워 1' -> 106
    """
    match = re.search(r"리뷰\s*([\d,]+)", full_text)
    if match:
        return match.group(1).replace(",", "")
    return ""


def count_review_chars(review_text):
    return len(review_text)


def extract_review_text(full_text):
    """
    카드 전체 텍스트에서 리뷰 본문 추출
    구조가 조금씩 달라도 최대한 본문만 남기도록 처리
    """
    try:
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        if not lines:
            return ""

        # 아래와 같은 UI/메타 영역이 나오기 전까지만 본문 후보로 사용
        stop_tokens = {
            "반응 남기기", "접기", "방문일", "인증 수단", "메뉴", "재방문", "영수증", "예약", "주문", "포장", "배달"
        }

        # 일반적으로 상단 2~3줄은 닉네임/리뷰수/팔로우 등
        # 여기서는 너무 공격적으로 자르지 않고, 메타로 보이는 줄만 제외
        filtered = []

        for line in lines:
            if line in stop_tokens:
                break

            # 상단 UI 제거
            if line == "팔로우":
                continue
            if re.fullmatch(r"리뷰\s*[\d,]+.*", line):
                continue
            if re.fullmatch(r"사진\s*[\d,]+.*", line):
                continue
            if re.fullmatch(r"팔로워\s*[\d,]+.*", line):
                continue

            # 방문 상황/혼밥/대기시간 등 키워드성 메타 제거
            meta_like_keywords = [
                "방문", "이용", "대기 시간", "연인", "배우자", "친구", "가족",
                "아이", "혼자", "점심", "저녁", "아침", "예약", "주문", "포장", "배달"
            ]
            if any(k in line for k in meta_like_keywords):
                # 날짜 줄은 뒤에서 따로 처리하므로 제외
                if re.search(r"\d{4}[.\년]\s*\d{1,2}[.\월]\s*\d{1,2}", line):
                    continue

            filtered.append(line)

        # 첫 줄이 닉네임일 가능성이 높으면 제거
        # 닉네임은 보통 짧고, 뒤에 본문이 이어짐
        if filtered:
            if len(filtered[0]) <= 20 and len(filtered) >= 2:
                # 너무 짧고 문장 느낌이 약하면 닉네임으로 간주
                if not re.search(r"[.!?]|[가-힣]{6,}", filtered[0]):
                    filtered = filtered[1:]

        review_text = " ".join(filtered).strip()

        # 날짜/방문횟수/인증수단 같은 텍스트 제거
        review_text = re.sub(r"\b\d{4}\.\d{1,2}\.\d{1,2}\b", " ", review_text)
        review_text = re.sub(r"\b\d{4}년\s*\d{1,2}월\s*\d{1,2}일\b", " ", review_text)
        review_text = re.sub(r"\b\d+번째 방문\b", " ", review_text)
        review_text = re.sub(r"\b(영수증|예약|주문|포장|배달)\b", " ", review_text)

        review_text = re.sub(r"\s+", " ", review_text).strip()

        # 너무 짧은 UI 찌꺼기는 제거
        if review_text in {"", "팔로우", "리뷰", "사진", "반응 남기기"}:
            return ""

        return review_text

    except Exception:
        return ""


def make_review_key(row):
    return (
        row.get("계정 ID", "").strip(),
        row.get("방문 날짜", "").strip(),
        row.get("리뷰 내용", "").strip()
    )


# =====================================
# 3. iframe 진입
# =====================================
def switch_to_entry_iframe(driver):
    driver.switch_to.default_content()

    for iframe_id in ["entryIframe", "searchIframe"]:
        try:
            WebDriverWait(driver, 5).until(
                EC.frame_to_be_available_and_switch_to_it((By.ID, iframe_id))
            )
            print(f"[INFO] iframe 진입 성공: {iframe_id}")
            return
        except TimeoutException:
            continue

    raise Exception("iframe을 찾지 못했습니다. 네이버 지도 구조를 다시 확인해야 합니다.")


# =====================================
# 4. 리뷰 탭 클릭
# =====================================
def click_review_tab(driver):
    candidate_xpaths = [
        "//a[contains(., '리뷰')]",
        "//button[contains(., '리뷰')]",
        "//span[contains(., '리뷰')]"
    ]

    for xpath in candidate_xpaths:
        try:
            elem = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            driver.execute_script("arguments[0].click();", elem)
            time.sleep(2)
            print("[INFO] 리뷰 탭 클릭 성공")
            return
        except Exception:
            continue

    print("[WARN] 리뷰 탭 클릭 실패 - 이미 리뷰 탭일 수도 있음")


# =====================================
# 5. 리뷰 영역 찾기 / 스크롤
# =====================================
def get_review_scroll_box(driver):
    """
    네이버 지도 리뷰 리스트가 들어있는 실제 스크롤 박스를 최대한 유연하게 찾는다.
    """
    candidates = [
        "div[role='main']",
        "div.place_section_content",
        "div.place_section",
        "div#_pcmap_list_scroll_container",
        "div[class*='review']",
        "body"
    ]

    for selector in candidates:
        try:
            elem = driver.find_element(By.CSS_SELECTOR, selector)
            return elem
        except Exception:
            continue

    return driver.find_element(By.TAG_NAME, "body")


def click_more_buttons(driver):
    """
    리뷰 더보기 / 펼쳐서 더보기 버튼을 가능한 만큼 클릭
    """
    xpaths = [
        "//a[contains(., '펼쳐서 더보기')]",
        "//button[contains(., '펼쳐서 더보기')]",
        "//a[contains(., '더보기')]",
        "//button[contains(., '더보기')]"
    ]

    total_clicked = 0

    for xpath in xpaths:
        try:
            buttons = driver.find_elements(By.XPATH, xpath)
            for btn in buttons:
                try:
                    if btn.is_displayed() and btn.is_enabled():
                        driver.execute_script("arguments[0].click();", btn)
                        time.sleep(0.15)
                        total_clicked += 1
                except Exception:
                    pass
        except Exception:
            pass

    return total_clicked


def scroll_once(driver):
    scroll_box = get_review_scroll_box(driver)

    before = driver.execute_script("return arguments[0].scrollTop;", scroll_box)
    height = driver.execute_script("return arguments[0].scrollHeight;", scroll_box)

    driver.execute_script(
        "arguments[0].scrollTop = arguments[0].scrollHeight;",
        scroll_box
    )
    time.sleep(SLEEP)

    after = driver.execute_script("return arguments[0].scrollTop;", scroll_box)
    return before, after, height


# =====================================
# 6. 리뷰 카드 파싱
# =====================================
def get_review_cards(driver):
    selectors = [
        "li.EjjAW",
        "li.place_apply_pui",
        "div[class*='place_apply_pui']",
        "li[class*='pui']"
    ]

    for selector in selectors:
        cards = driver.find_elements(By.CSS_SELECTOR, selector)
        if cards:
            print(f"[DEBUG] selector={selector} / count={len(cards)}")
            return cards

    return []


def parse_one_card(card):
    full_text = card.text.strip()
    if not full_text:
        return None

    # 계정 ID
    account_id = safe_text(
        card, By.CSS_SELECTOR,
        ".pui__NMi-Dp, .pui__uslU0d, .place_bluelink, [class*='nick'], [class*='name']"
    )

    # fallback: 첫 줄을 계정명 후보로 사용
    if not account_id:
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]
        if lines:
            account_id = lines[0]

    account_review_count = extract_account_review_count(full_text)
    visit_date, visit_count, auth_method = parse_visit_info_from_text(full_text)

    if not visit_date or not is_valid_date(visit_date):
        return None

    keywords = safe_texts(
        card, By.CSS_SELECTOR,
        ".pui__jhpEyP span, .keyword_tag, [class*='tag']"
    )
    keywords_joined = ", ".join(keywords)

    review_text = extract_review_text(full_text)
    review_char_count = count_review_chars(review_text)

    has_photo = exists(
        card, By.CSS_SELECTOR,
        "img, .place_thumb, .review_photo"
    )

    row = {
        "계정 ID": account_id,
        "계정의 리뷰 수": account_review_count,
        "방문 날짜": visit_date,
        "방문 횟수": visit_count,
        "인증 수단": auth_method,
        "키워드": keywords_joined,
        "리뷰 내용": review_text,
        "리뷰 글자 수": review_char_count,
        "리뷰 내 사진 유무": has_photo
    }

    # 리뷰 내용이 비어 있어도, 최소 메타는 남길 수 있게 허용
    if not row["계정 ID"] and not row["리뷰 내용"]:
        return None

    return row


def collect_visible_reviews(driver, collected_dict):
    cards = get_review_cards(driver)
    new_count = 0
    parsed_count = 0

    for idx, card in enumerate(cards, start=1):
        try:
            row = parse_one_card(card)
            if not row:
                continue

            parsed_count += 1
            key = make_review_key(row)

            if key not in collected_dict:
                collected_dict[key] = row
                new_count += 1

        except Exception as e:
            print(f"[WARN] 카드 파싱 실패: {e}")

    print(f"[INFO] 현재 화면 파싱 성공: {parsed_count}개 / 신규 누적: {new_count}개 / 총 누적: {len(collected_dict)}개")
    return new_count


# =====================================
# 7. 전체 수집 루프
# =====================================
def collect_all_reviews(driver):
    collected = {}
    idle_rounds = 0

    for round_idx in range(1, SAFETY_MAX_ROUNDS + 1):
        print(f"\n[INFO] ===== 수집 라운드 {round_idx} =====")

        # 현재 보이는 카드들 먼저 수집
        click_count = click_more_buttons(driver)
        if click_count:
            print(f"[INFO] 더보기 클릭 수: {click_count}")

        new_count = collect_visible_reviews(driver, collected)

        if len(collected) > 0 and len(collected) % 50 == 0:
            pd.DataFrame(list(collected.values())).to_csv(
                TEMP_CSV, index=False, encoding="utf-8-sig"
            )
            print(f"[INFO] 중간 저장 완료: {len(collected)}개")

        # 신규 수집이 없으면 idle 증가
        if new_count == 0:
            idle_rounds += 1
        else:
            idle_rounds = 0

        # 스크롤
        before, after, height = scroll_once(driver)
        print(f"[INFO] scrollTop: {before} -> {after} / scrollHeight={height}")

        # 다시 보이는 카드 수집
        click_count_2 = click_more_buttons(driver)
        if click_count_2:
            print(f"[INFO] 스크롤 후 더보기 클릭 수: {click_count_2}")

        new_count_after_scroll = collect_visible_reviews(driver, collected)

        if new_count_after_scroll == 0:
            idle_rounds += 1
        else:
            idle_rounds = 0

        if len(collected) > 0 and len(collected) % 50 == 0:
            pd.DataFrame(list(collected.values())).to_csv(
                TEMP_CSV, index=False, encoding="utf-8-sig"
            )
            print(f"[INFO] 중간 저장 완료: {len(collected)}개")

        # 종료 조건
        if idle_rounds >= MAX_IDLE_ROUNDS:
            print("[INFO] 새 리뷰가 더 이상 늘지 않아 수집 종료")
            break

    return list(collected.values())


# =====================================
# 8. 메인 실행
# =====================================
def main():
    driver = create_driver()

    try:
        print("[INFO] 페이지 접속 중...")
        driver.get(NAVER_MAP_URL)
        time.sleep(3)

        switch_to_entry_iframe(driver)
        click_review_tab(driver)

        data = collect_all_reviews(driver)
        print(f"[INFO] 총 수집된 리뷰 수(중복 제거 전): {len(data)}")

        df = pd.DataFrame(data)

        if not df.empty:
            df = df.drop_duplicates(
                subset=["계정 ID", "방문 날짜", "리뷰 내용"],
                keep="first"
            ).sort_values(by=["방문 날짜", "계정 ID"], ascending=[False, True])

        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[DONE] 최종 저장 완료: {len(df)}개 -> {OUTPUT_CSV}")
        print(df.head())

    finally:
        driver.quit()


if __name__ == "__main__":
    main()