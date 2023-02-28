---
title: Big Query
date: 2022-09-29
categories:
  - Data Base
  - GCP
tags: 
  - Big Query
---

### 1. 쿼리 실행순서
    
`FROM → WHERE → GROUP BY, Aggregation → HAVING → WINDOW → QUALIFY → DISTINCT → ORDER BY → LIMIT`
    
### 2. JOIN
    
![](images/Big_Query/Untitled.png)
    
### 3. WINDOW 함수
    
![](images/Big_Query/Untitled%201.png)
    
### 4. DECLARE
- 변수를 선언 혹은 초기화할 때 사용
    `DECLARE variable_name[, ...] [variable_type] [ DEFAULT expression];`
    
    ```sql
    -- 아래 쿼리를 실행하면 값이 모두 1로 나오는걸 확인 할 수 있음
    -- DEFAULT를 안주면 NULL로 지정 됨
    DECLARE x, y, z INT64 DEFAULT 1;
    SELECT x,y,z
    
    -- 현재 날짜로 d할당
    DECLARE d DATE DEFAULT CURRENT_DATE();
    ```
    
- 위 예시 말고 쿼리의 결과를 사용해 변수를 초기화 할 수도 있음

### 5. SET
- DECLARE와 같이 사용 되어짐.
- DECLARE에서 변수 타입을 지정하고 SET으로 값 할당이 가능,
- DECLARE에서 두 과정 모두 할 수 있지만 SET은 쿼리 내 어느 위치에서나 사용 가능 함

### 6. UDF - User Define Function
- 영구 UDF는 여러 쿼리에서 재사용 할 수 있음
`CREATE TEMP FUNCTION` → 임시 UDF 생성
`CREATE FUNCTION` → 영구 UDF 생성
`CREATE OR REPLACE FUNCTION` → 영구 UDF 생성 및 수정

    ```sql
    CREATE OR REPLACE FUNCTION ps-datateam.cbt_global_ceo.함수명(변수명 변수타입)
    RETURNS INT64 --리턴 타입
    LANGUAGE js -- 작성 언어(임시는 SQL로도 가능한 듯)
    AS """
    if (mode == 'solo') {
    return 1;
    } else if (mode == 'duo') {
    return 2;
    } else if (mode == 'trio'){
    return 3;
    } else if (mode == 'squad'){
    return 4;
    } else {
    return 0;
    }
    """; -- 함수
    ```
    
### 7. 파이썬에서 빅쿼리 데이터 사용
    
```python
    from google.cloud import bigquery
    from google.oauth2 import service_account
    key_path = "키파일 경로"
    credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    client = bigquery.Client(credentials=credentials, project='프로젝트명')
    
    # 쿼리 결과를 CSV파일로 -----------------------------------------------------------------
    query = client.query("""
    QUERY
    """)
    # 쿼리 -> DataFrame
    df_result = query.to_dataframe ()
    # DataFrame -> csv
    df_result.to_csv("filename.csv", index=False, encoding='euc-kr')
    # 프로젝트>데이터세트 내의 테이블 목록 조회------------------------------------------------
    dataset_ref = client.dataset("kr_dict") # kr_dict라는 데이터 세트의 정보를 조회
    dataset = client.get_dataset(dataset_ref)
    list(client.list_tables(dataset)) # 데이터 세트의 테이블 목록을 리스트로 가져옴
```
    
---
- [Reference1](https://zzsza.github.io/bigquery/advanced/window-function.html)
- [Reference2](https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions)