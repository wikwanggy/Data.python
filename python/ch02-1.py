from selenium import webdriver
from bs4 import BeautifulSoup
import re

# 윈도우용 크롬 웹드라이버 실행 경로 (Windows)
excutable_path = "chromedriver.exe"

# 크롤링할 사이트 주소를 정의합니다.
source_url = "https://namu.wiki/RecentChanges"

# 크롬 드라이버를 사용합니다 (맥은 첫 줄, 윈도우는 두번째 줄 실행)
# driver = webdriver.Chrome(path)
driver = webdriver.Chrome(executable_path=excutable_path)

# 드라이버가 브라우징 할 페이지 소스를 입력합니다
driver.get(source_url)
req = driver.page_source

# 사이트의 html 구조에 기반하여 데이터를 파싱합니다.
soup=BeautifulSoup(req, "html.parser")
contents_table = soup.find(name="table")
table_body = contents_table.find(name="tbody")
table_rows = table_body.find_all(name="tr")

# a태그의 href 속성을 리스트로 추출하여, 크롤링 할 페이지 리스트를 생성합니다.
page_url_base = "https://namu.wiki"
page_urls = []
for index in range(0, len(table_rows)):
    first_td = table_rows[index].find_all("td")[0]
    td_url = first_td.find_all("a")
    if len(td_url) > 0:
        page_url = page_url_base + td_url[0].get("href")
        if "png" not in page_url:
            page_urls.append(page_url)

# 중복 url을 제거합니다.
page_urls = list(set(page_urls))
for page in page_urls[:3]:
    print(page)
    
# 크롤링에 사용한 브라우저를 종료합니다.
driver.close()