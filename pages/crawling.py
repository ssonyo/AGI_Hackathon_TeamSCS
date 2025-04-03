import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import re
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def extract_article_no(url):
    match = re.search(r'articleNo=(\d+)', url)
    if match:
        return match.group(1)
    return None

def parse_table_to_dict(table):
    """<table> 태그를 받아 dict로 변환"""
    info = {}
    rows = table.select("tbody tr")
    for row in rows:
        cells = row.find_all(["th", "td"])
        for i in range(0, len(cells) - 1, 2):
            key = cells[i].get_text(strip=True)
            value = cells[i + 1].get_text(strip=True)
            info[key] = value
    return info

def get_property_details(property_list):
    results = []
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # ← 최신 크롬은 이렇게
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option("detach", True)

    chrome_options.add_experimental_option("excludeSwitches",["enable-logging"])

    service = Service(executable_path=ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, 10)

    try:
        for article_no in property_list:
            try:
                url = f"https://land.naver.com/info/printArticleDetailInfo.naver?atclNo={article_no}&atclRletTypeCd=E03&rletTypeCd=E03"
                driver.get(url)
                time.sleep(1)

                soup = BeautifulSoup(driver.page_source, 'html.parser')

                tables = soup.find_all("table", summary=True)
                main_info = {}
                detail_info = {}

                for table in tables:
                    summary = table.get("summary", "")
                    if "매물정보" in summary:
                        main_info = parse_table_to_dict(table)
                    elif "매물세부정보" in summary:
                        detail_info = parse_table_to_dict(table)

                results.append({
                    "article_no": article_no,
                    "main_info": main_info,
                    "detail_info": detail_info
                })

            except Exception as e:
                print(f"{article_no} 매물 상세 추출 실패:", e)
                continue
    finally:
        driver.quit()

    return results



# def get_property_details(property_list):
#     results = []
#     driver = webdriver.Chrome()
#     wait = WebDriverWait(driver, 10)

#     try:
#         for article_no in property_list:
#             try:
#                 url = f"https://land.naver.com/info/printArticleDetailInfo.naver?atclNo={article_no}&atclRletTypeCd=E03&rletTypeCd=E03"
#                 driver.get(url)
#                 time.sleep(1)

#                 soup = BeautifulSoup(driver.page_source, 'html.parser')
#                 details = soup.text.strip()
#                 results.append({
#                     "article_no": article_no,
#                     "details": details
#                 })

#             except Exception as e:
#                 print(f"{article_no} 매물 상세 추출 실패:", e)
#                 continue
#     finally:
#         driver.quit()

#     return results



def get_property_list(latitude, longitude):
    """특정 위도, 경도의 매물 목록을 가져오는 함수"""
    results = []
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # ← 최신 크롬은 이렇게
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option("detach", True)

    chrome_options.add_experimental_option("excludeSwitches",["enable-logging"])

    service = Service(executable_path=ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # 웹드라이버 설정
        driver = webdriver.Chrome()
        wait = WebDriverWait(driver, 10)
        
        # 네이버 부동산 URL 생성
        url = f"https://new.land.naver.com/offices?ms={latitude},{longitude},16&a=TJ&b=A1&e=RETAIL"
        driver.get(url)
        time.sleep(2)  # 페이지 로딩 대기
        
        # 스크롤 컨테이너 찾기
        scroll_container = driver.find_element(By.CSS_SELECTOR, ".item_list.item_list--article")

        # 초기 높이 설정
        prev_height = driver.execute_script("return arguments[0].scrollHeight", scroll_container)
        same_count = 0  # 변화 없는 횟수

        while True:
            # 내부 컨테이너를 스크롤
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scroll_container)
            time.sleep(1.5)

            new_height = driver.execute_script("return arguments[0].scrollHeight", scroll_container)

            if new_height == prev_height:
                same_count += 1
            else:
                same_count = 0

            if same_count >= 5:
                break

            prev_height = new_height

        # 매물 목록 가져오기
        items = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".item_list.item_list--article .item")))
        
        
        for item in items:
            try:
                item.click()
                time.sleep(1)
                
                article_no = extract_article_no(driver.current_url)
                if article_no:
                    # 여기서 상세 정보도 가져오기
                    # driver 전달
                    results.append(article_no)
                    print(f"매물 저장: {article_no}")
                
                # # 이전 페이지로 돌아가기
                # driver.back()
                # time.sleep(1)
                
            except Exception as e:
                print(f"매물 처리 중 오류 발생: {e}")
                continue
                
        return results
                
    except Exception as e:
        print(f"페이지 처리 중 오류 발생: {e}")
        return results
        
    finally:
        if driver:
            driver.quit()

# 사용 예시
latitude = 37.430218  # 예시 위도
longitude = 127.002425  # 예시 경도
property_list = get_property_list(latitude, longitude)
print(f"총 {len(property_list)}개의 매물이 발견되었습니다.")

# a = get_property_details(property_list)
# for i in a:
#     print(i)


        
