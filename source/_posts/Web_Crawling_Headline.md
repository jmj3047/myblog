---
title: Web Crawling Practice
date: 2022-04-22
categories:
  - Python
  - HTML
tags:
   - HTML
   - Web Crawling
   - BeautifulSoup
---

### 1. Crawling Headline news from Naver
```python
import requests
from bs4 import BeautifulSoup

import pandas as pd

def crawling(soup):
    div = soup.find("div", class_='list_issue')
    # print(type(div))
    print(div.find_all('a')) #list형태
    result = []
    urls = []
    for a in div.find_all("a"):
        # print(a.get_text())
        urls.append(a['href'])
        result.append(a.get_text())
    
    # print(result) 

    #save as csv file
    df = pd.DataFrame({'news_title': result,
                       "url": urls})
    print(df)
    df.to_csv("newscrawling.csv")
    
    
    print("Crawling Success!")
    

def main():
    CUSTOM_HEADER = {
        'referer' : 'https://www.naver.com/',
        'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }

    url = 'https://www.naver.com/'
    req = requests.get(url = url, headers=CUSTOM_HEADER)
    print(req.status_code)
    #200: 정상적으로 사이트 돌아가고 있음
    #404: 주소 오류
    #503: 서버 내려진 상태
    
    #print(req.text)이거를 beautifulsoup객체로 바꿔줌
    soup = BeautifulSoup(req.text, 'html.parser')
    print(type(soup))
    # print(soup.find("strong", class_='new'))
    # print(soup.find_all("span", class_='blind'))
    #print(soup.find("a", class_='link_newsstand'))
    
    crawling(soup)

if __name__ == "__main__":
    main()
```

### 2. Crawling Product List from ACBF
```python
import requests
from bs4 import BeautifulSoup

import pandas as pd

def crawling(soup):
    div = soup.find_all("div", class_='name')
    print(type(div)) #<class 'bs4.element.ResultSet'>
    # print(div)
    product_name = []
    # urls =[]
    for a in div:
        # print(a.get_text())
        # urls.append(a.get('href'))
        product_name.append(a.get_text())
    
    print(product_name)




    # df = pd.DataFrame({'news_title': product_name})
    # print(df)
    # df.to_csv("suit_product.csv")
    
    
    print("Crawling Success!")
    

def main():
    CUSTOM_HEADER = {
        'referer' : 'https://anyonecanbeafuse.com/category/%EC%88%98%ED%8A%B8/89/',
        'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }

    url = 'https://anyonecanbeafuse.com/category/%EC%88%98%ED%8A%B8/89/'
    req = requests.get(url = url, headers=CUSTOM_HEADER)
    print(req.status_code)
    #200: 정상적으로 사이트 돌아가고 있음
    #404: 주소 오류
    #503: 서버 내려진 상태
    
    #print(req.text)이거를 beautifulsoup객체로 바꿔줌
    soup = BeautifulSoup(req.text, 'html.parser')
    #print(type(soup))
    # print(soup.find_all("div", class_='name'))
    

    
    crawling(soup)

if __name__ == "__main__":
    main()
```

### 3. Crawling Music Title from Chart
```python
import requests
from bs4 import BeautifulSoup

def crawling(soup):
    tbody_df = soup.find("tbody")
    # print(tbody_df)
    
    result = []
    
    for a in tbody_df.find_all('p', class_='title'):
        # print(a.get_text())
        # print(type(a.get_text()))
        result.append(a.get_text().strip("\n"))
        
    print(result)
    print("Crawling Success!")
    

def main():
    CUSTOM_HEADER = {
        'referer' : 'https://music.bugs.co.kr/chart', #필수 아님
        'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }

    url = 'https://music.bugs.co.kr/chart'
    req = requests.get(url = url, headers=CUSTOM_HEADER)
    print(req.status_code)
    #200: 정상적으로 사이트 돌아가고 있음
    #404: 주소 오류
    #503: 서버 내려진 상태
    
    #print(req.text)이거를 beautifulsoup객체로 바꿔줌
    soup = BeautifulSoup(req.text, 'html.parser')
    print(type(soup)) #<class 'bs4.BeautifulSoup'>
    # print(soup.find_all("p", class_='title'))
    # print(soup.find_all("span", class_='blind'))
    #print(soup.find("a", class_='link_newsstand'))
    
    crawling(soup)

if __name__ == "__main__":
    main()
```