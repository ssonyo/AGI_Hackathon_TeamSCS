from bs4 import BeautifulSoup
import re
import json
import requests
import os
import time
from tqdm import tqdm
from urllib.parse import urljoin, unquote

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# 다운로드 폴더 경로 (절대 경로)
download_dir = "/Users/jungyang/Desktop/YBIGTA/2025-1/해커톤/AGI_Hackathon_TeamSCS/EUM_crawling/goshi"

def get_link():
    link_for_each_eum = []
    for i in tqdm(range(1, 11)):
        link = f"https://www.eum.go.kr/web/gs/gv/gvGosiList.jsp?listSize=50&pageNo={i}&zonenm=&startdt=&enddt=&chrgorg=&selSggCd=11&select2=1100000000&select_3=&gosino=&gosichrg=&prj_nm=&prj_cat_cd=&geul_yn=Y&gihyung_yn=&silsi_yn=&mobile_yn="
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')

        for tr in soup.find_all('tr', class_="center"):
            date = ""
            td_list = tr.find_all('td', class_="left")
            td_date = tr.find_all('td', class_="mb")
            for td in td_date:
                # 'left mb' 클래스를 가진 td 제외
                if (td.get('class') != ['left', 'mb']) and (date == ""):
                    date = td.text

            for td in td_list:
                a_tag = td.find('a', href=True)
                if a_tag:
                    full_url = urljoin(link, a_tag['href'])
                    link_for_each_eum.append({"url": full_url, "date": date})

    return link_for_each_eum

def get_data(one_eum: dict):
    link = one_eum["url"]
    date = one_eum["date"]
    try:
        chrome_options = Options()
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        })
        # headless 모드를 사용하려면 아래 주석 해제
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(link)
        time.sleep(2)  # 페이지 로딩 대기

        download_links = driver.find_elements(By.CSS_SELECTOR, 'a.link.font_blue04')
        print(f"[{link}] Found {len(download_links)} file(s)")

        for i, a in enumerate(download_links):
            try:
                # 다운로드 전 현재 폴더 내 파일 목록 확인
                before_files = set(os.listdir(download_dir))
                
                # JavaScript를 이용하여 hover와 클릭 이벤트를 강제 발생
                driver.execute_script("""
                    // 마우스 오버(hover) 이벤트 강제 발생
                    var mouseOverEvent = new MouseEvent('mouseover', {
                        bubbles: true,
                        cancelable: true,
                        view: window
                    });
                    arguments[0].dispatchEvent(mouseOverEvent);
                    
                    // 클릭 이벤트 강제 발생
                    var clickEvent = new MouseEvent('click', {
                        bubbles: true,
                        cancelable: true,
                        view: window
                    });
                    arguments[0].dispatchEvent(clickEvent);
                """, a)
                print(f" ✅ Forced hover & click on file {i+1}")
                # 파일명에서 괄호와 용량 정보를 제거하되, 확장자는 유지
                original_file_name = a.text.strip()
                if "(" in original_file_name:
                    # 마지막 괄호와 그 안의 용량 정보만 제거
                    parts = original_file_name.rsplit("(", 1)
                    if "KB" in parts[1] or "MB" in parts[1]:
                        original_file_name = parts[0].strip()
                
                # URL 디코딩
                original_file_name = unquote(original_file_name)

                # 다운로드 완료를 확인하기 위한 대기 (임시파일(.crdownload) 배제)
                timeout = 30
                downloaded_file = None
                while timeout > 0:
                    time.sleep(1)
                    after_files = set(os.listdir(download_dir))
                    new_files = after_files - before_files
                    # 임시파일(.crdownload)은 제외
                    new_files = {f for f in new_files if not f.endswith(".crdownload")}
                    if new_files:
                        downloaded_file = new_files.pop()
                        break
                    timeout -= 1
                
                if downloaded_file:
                    old_path = os.path.join(download_dir, downloaded_file)
                    # new_file_name = f"{date}_{unquote(downloaded_file)}"
                    new_file_name = f"{date}_{original_file_name}"
                    new_path = os.path.join(download_dir, new_file_name)
                    print(new_path)
                    os.rename(old_path, new_path)
                    print(f" 📦 Renamed file to: {new_file_name}")
                else:
                    print(" ⚠️ No new file detected after download.")
                
                time.sleep(2)  # 다음 다운로드 전 잠시 대기
            except Exception as e:
                print(f" 🚨 Error processing file {i+1}: {e}")

    except Exception as e:
        print(f"[Error] Failed to process link: {link}")
        print("Reason:", e)
    finally:
        driver.quit()

def main():
    # 링크 정보를 처음 생성하려면 아래 주석 해제 후 실행하세요.
    # links_for_each_eum = get_link()
    # with open("EUM_crawling/links_for_each_eum.json", "w") as f:
    #     json.dump(links_for_each_eum, f)

    # 저장된 JSON 파일에서 링크 정보를 읽어옵니다.
    json_file = "EUM_crawling/links_for_each_eum.json"
    with open(json_file, "r") as f:
        links_for_each_eum = json.load(f)

    for link in links_for_each_eum:
        get_data(link)

if __name__ == "__main__":
    main()
