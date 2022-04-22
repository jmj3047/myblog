---
title: Basic Web Crawling
date: 2022-04-22
categories:
  - Python
  - HTML
tags:
   - HTML
   - Web Crawling
   - BeautifulSoup
---

> Crawling Tools
   -Beautifulsoup: 파이썬에서 가장 일반적인 수집도구(CSS 통해서 수집)
   -Scrapy (CSS, XPATH 형태로 수집)
   -Selenium (CSS, XPATH 통해서 데이터 수집 + Java Script)
      →자바 필요 + 몇개 설치 도구 필요

> 웹사이트 만드는 3대 조건 +1
   :HTML, CSS, JavaScript, Ajax(비동기처리)

> 웹사이트 구동방식
   :GET / POST



### Create virtual env(git bash)
```bash
pip install virtualenv
python -m virtualenv venv
source venv/Scripts/activate
```


### Installing Library
```bash
pip install beautifulsoup4
pip install numpy pandas matplotlib seaborn
pip install requests
```


---
- Reference: [https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-beautiful-soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#installing-beautiful-soup)