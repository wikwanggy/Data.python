# pip install selenium beautifulsoup4 로 라이브러리 설치
# pip install webdriver_manager로 라이브러리 설치
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 크롤링할 사이트 주소를 정의합니다.
source_url = "https://namu.wiki/RecentChanges"

# 사이트의 html 구조에 기반하여 크롤링을 수행합니다.
driver = webdriver.Chrome(ChromeDriverManager().install())  # for Windows
driver.get(source_url)
driver.implicitly_wait(10)
table_rows = driver.find_elements(By.XPATH,'//*[@id="C6Rc9QlVe"]/div[2]/div/div/div/div/div/article/div[3]/div/div/div/div[1]/div/div/table/tbody/tr/td/a')
# print(table_rows)
page_urls = []
for i in range(0,len(table_rows)):
    page_urls.append(table_rows[i].get_attribute("href"))   # a태그의 href 속성을 리스트로 추출하여, 크롤링 할 페이지 리스트를 생성합니다.

# 중복 url을 제거합니다.
page_urls = list(set(page_urls))
for page in page_urls[:3]:
    print(page)

# 크롤링에 사용한 브라우저를 종료합니다.
driver.close()