from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

NAVER_URL = "https://map.naver.com/p/entry/place/37126807?placePath=/review"

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get(NAVER_URL)
time.sleep(4)

# iframe 목록 확인
iframes = driver.find_elements(By.TAG_NAME, 'iframe')
print(f"iframe 개수: {len(iframes)}")
for i, f in enumerate(iframes):
    print(f"  [{i}] id={f.get_attribute('id')} src={f.get_attribute('src')[:80] if f.get_attribute('src') else 'None'}")

# iframe 진입 시도
try:
    driver.switch_to.frame(iframes[1])
    time.sleep(2)
    print("\n=== iframe[1] HTML (앞 3000자) ===")
    print(driver.page_source[:3000])
except Exception as e:
    print(f"iframe 진입 실패: {e}")

input("확인 완료 후 엔터 누르세요...")
driver.quit()