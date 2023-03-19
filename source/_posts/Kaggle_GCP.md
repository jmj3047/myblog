---
title: Python으로 kaggle 데이터 GCP에 적재
date: 2023-03-03
categories:
  - Data Platform/Base
  - GCP
tags: 
  - Big Query
  - kaggle
  - Python
---

## 요약

- Kaggle 데이터 다운로드
- GCP에 데이터 세트 만들고 서비스 계정 생성하기
- Python-BigQuery 연결 후 데이터 조회
- 데이터 적재 하기

## Kaggle 데이터 다운로드

- kaggle을 설치한다
    
    ```python
    !pip install kaggle
    ```
    
- kaggle의 key를 받아온다
    
    ```python
    !mkdir ~/.kaggle
    !echo '{"username":"your_id","key":"your_key"}' > ~/.kaggle/kaggle.json
    !chmod 600 ~/.kaggle/kaggle.json
    ```
    
- kaggel TOKEN 받아오기
    - kaggle에 접속한다음 프로필을 선택하고 Account를 누른다
        
        ![ ](images/Kaggle_GCP/Untitled.png)
        
    - 화면을 내리면 `API 탭`을 찾을수 있다.
        
        ![ ](images/Kaggle_GCP/Untitled%201.png)
        
    - Create New API Token을 누르고 다운 `kaggle.json` 파일을 받아줍니다.
    - 한번 발급 받으면 이전의 것은 알려주지 않기 때문에 잊어버렸다면 `Expire API TOKEN` 으로 모두 지고 새로운 토큰을 받는다.
        
        ![ ](images/Kaggle_GCP/Untitled%202.png)
        
- kaggle에서 원하는 데이터를 다운 받아준다 → competitions에서 원하는 competition을 선택하고 data 탭으로 가면 다운 받을 수 있는 api가 있다.
    
    ![ ](images/Kaggle_GCP/Untitled%203.png)
    
    ![ ](images/Kaggle_GCP/Untitled%204.png)
    
    ![ ](images/Kaggle_GCP/Untitled%205.png)
    

---

## GCP에 데이터 세트 만들고 서비스 계정 생성하기

- GCP에 프로젝트를 만들고 새로운 데이터 세트를 만들어 준다
- 중간에 Default table expiration 을 클릭하면 테이블이 며칠 후에 만료되는지도 설정 가능하다
    
    ![ ](images/Kaggle_GCP/Untitled%206.png)
    
    ![ ](images/Kaggle_GCP/Untitled%207.png)
    
- 서비스 계정 생성
    - GCP 좌측 상단 ‘탐색 메뉴’ 클릭후 ‘IAM 및 관리자’의 ‘서비스 계정’으로 이동
        
        ![ ](images/Kaggle_GCP/Untitled%208.png)
        
    - ‘+ 서비스 계정 만들기’ 클릭 후 서비스 계정 만들기 진행
        
        ![ ](images/Kaggle_GCP/Untitled%209.png)
        
        ![ ](images/Kaggle_GCP/Untitled%2010.png)
        
- 서비스 계정 키 생성 후 JSON 파일 추출
    - Email에 생성된 서비스 계정 클릭 → 페이지 상단에 KEYS → ADD KEY → Create new key
        
        ![ ](images/Kaggle_GCP/Untitled%2011.png)
        
    - config 폴더 생성 후 해당 경로에 json 파일 다운로드

        ![ ](images/Kaggle_GCP/Untitled%2012.png)

        ![ ](images/Kaggle_GCP/Untitled%2013.png)

- 서비스 계정에 빅쿼리 관련 역할 추가
    - GCP 좌측 상단 ‘탐색 메뉴’ 클릭후 ‘IAM 및 관리자’의 ‘IAM’으로 이동
        
        ![ ](images/Kaggle_GCP/Untitled%2014.png)
        
    - ‘추가’ 클릭 - 생성된 서비스 계정 이메일 추가 및 ‘BigQuery 관리자’ 역할 선택 후 저장
        
        ![ ](images/Kaggle_GCP/Untitled%2015.png)
        

---

## Python-BigQuery 연결 후 데이터 조회

- 구글 클라우드 빅쿼리 클라이언트 설치
    
    ```python
    !pip install google-cloud-bigquery
    ```
    
- 서비스 계정 키 설정 → 빅쿼리 클라이언트 정의 → 데이터 조회 쿼리 실행
    
    ![ ](images/Kaggle_GCP/Untitled%2016.png)
    

---

## 데이터 적재 하기

- 데이터를 불러온다
    
    ```python
    import pandas as pd
    
    BASE_DIR = "./"
    train = pd.read_csv(BASE_DIR + 'train.csv')
    test = pd.read_csv(BASE_DIR + 'test.csv')
    
    train.shape, test.shape
    ```
    
    ```
    ((3000888, 6), (28512, 5))
    ```
    
- 데이터를 적재 한다
    
    ```python
    from google.oauth2 import service_account
    import pandas_gbq
    
    cd = service_account.Credentials.from_service_account_file("./config/my-project-230227-c3691227da1d.json")
    project_id = 'my-project-230227'
    train_table = 'kaggle_data.train'
    test_table = 'kaggle_data.test'
    train.to_gbq(train_table,project_id,if_exists='replace',credentials=cd) 
    test.to_gbq(test_table,project_id,if_exists='replace',credentials=cd) 
    print('migration complete')
    ```
    
    ![ ](images/Kaggle_GCP/Untitled%2017.png)
    
- GCP에 들어가서 확인해 본다
    
    ![ ](images/Kaggle_GCP/Untitled%2018.png)
    
- 이렇게 테이블을 gcp에 적재하면 데이터들을 다양한 방면으로 사용이 가능하다: [Looker Studio를 이용한 시각화](https://jmj3047.github.io/2023/03/02/GCP_LookerStudio/)

---

- Reference
    - [https://dschloe.github.io/gcp/bigquery/01_settings/python_bigquery/](https://dschloe.github.io/gcp/bigquery/01_settings/python_bigquery/)
    - [https://velog.io/@skyepodium/Kaggle-API-사용법](https://velog.io/@skyepodium/Kaggle-API-%EC%82%AC%EC%9A%A9%EB%B2%95)
    - [https://wooiljeong.github.io/python/python-bigquery/](https://wooiljeong.github.io/python/python-bigquery/)