from datetime import datetime

# =====================================
# 수집 대상 설정
# =====================================
NAVER_MAP_URL = (
    "https://map.naver.com/p/search/%EA%B5%B4%EB%8B%A4%EB%A6%AC%EC%8B%9D%EB%8B%B9/place/11707324?c=15.00,0,0,0,dh&placePath=/review?bk_query=%EA%B5%B4%EB%8B%A4%EB%A6%AC%EC%8B%9D%EB%8B%B9&entry=bmp&fromPanelNum=2&locale=ko&searchText=%EA%B5%B4%EB%8B%A4%EB%A6%AC%EC%8B%9D%EB%8B%B9&svcName=map_pcv5&timestamp=202605011305&entry=bmp&fromPanelNum=2&timestamp=202605011305&locale=ko&svcName=map_pcv5&searchText=%EA%B5%B4%EB%8B%A4%EB%A6%AC%EC%8B%9D%EB%8B%B9&from=map"
)

# 수집할 리뷰 날짜 범위 (이 날짜 이전 리뷰가 나오면 수집 중단)
START_DATE = datetime(2025, 1, 1)

# =====================================
# 크롤링 동작 설정
# =====================================
SLEEP             = 1.2  # 스크롤 후 대기 시간 (초)
MAX_IDLE_ROUNDS   = 8    # 신규 리뷰 없을 때 허용 횟수 (초과 시 종료)
SAFETY_MAX_ROUNDS = 300  # 무한루프 방지용 최대 라운드
MAX_REVIEWS       = 200  # 최대 수집 리뷰 수 (0 또는 None이면 제한 없음)

# =====================================
# 출력 파일 경로
# =====================================
OUTPUT_CSV = "naver_reviews.csv"
TEMP_CSV   = "naver_reviews_temp.csv"  # 크롤링 중간 임시 저장