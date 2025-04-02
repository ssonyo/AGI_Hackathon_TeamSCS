from bs4 import BeautifulSoup
import ast
import requests
import os
import time
from tqdm import tqdm
from urllib.parse import urljoin
from urllib.parse import unquote

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# 1년 할거면 10까지
# 5년 할거면 38까지
# 10년 할거면 66까지
download_dir = "/Users/jungyang/Desktop/YBIGTA/2025-1/해커톤/goshi/"

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
            # print(td_date)
            for td in td_date:
                # except the td that has class_="left mb"
                if (td.get('class') != ['left', 'mb']) and (date == ""):
                    date = td.text

            for td in td_list:
                a_tag = td.find('a', href=True)
                if a_tag:
                    full_url = urljoin(link, a_tag['href'])
                    link_for_each_eum.append({"url" : full_url, "date" : date})

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
        # chrome_options.add_argument("--headless") 
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        driver.get(link)
        time.sleep(2)

        download_links = driver.find_elements(By.CSS_SELECTOR, 'a.link.font_blue04')

        print(f"[{link}] Found {len(download_links)} file(s)")

        for i, a in enumerate(download_links):
            try:
                a.click()
                print(f" ✅ Clicked to download file {i+1}")
                time.sleep(2)  # 다운로드 대기

            except Exception as e:
                print(f" 🚨 Error clicking file {i+1}: {e}")


    except Exception as e:
        print(f"[Error] Failed to process: {link}")
        print("Reason:", e)

    finally:
        driver.quit()


def main():
    links_for_each_eum = get_link()

    for link in links_for_each_eum:
        get_data(link)


if __name__ == "__main__":
    main()


# before_files = set(os.listdir(download_dir))
# a.click()
# print(f" ✅ Clicked to download file {i+1}")
# time.sleep(5)

# after_files = set(os.listdir(download_dir))
# new_files = after_files - before_files

# if new_files:
#     downloaded_file = list(new_files)[0]
#     old_path = os.path.join(download_dir, downloaded_file)

#     new_name = f"{date}_{unquote(downloaded_file)}"
#     new_path = os.path.join(download_dir, new_name)

#     os.rename(old_path, new_path)
#     print(f" 📦 Renamed to: {new_name}")
# else:
#     print(" ⚠️ No new file detected after download.")
