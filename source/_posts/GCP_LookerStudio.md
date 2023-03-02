---
title: GCP - Looker Studio 연결하여 대시보드 작성
date: 2023-03-02
categories:
  - Data Base
  - GCP
tags: 
  - Big Query
  - Looker Studio
---

## 개요

- GCP - Looker Studio 연결해서 대시보드 작성하기
- `bigquery-public-data.ml_datasets.census_adult_income` 데이터 사용

## 목표

- 대시보드로 데이터를 시각화 하여 인사이트를 도출해본다.
- `개인의 연간 소등이 50,000달러 이상인지 예측하기` 를 위해 지표들의 상관관계를 확인해본다.

## public-dataset 가져오기

- 이전 포스트의 [0단계]https://jmj3047.github.io/2023/02/28/BQML_Pred/)와 동일하다.

## 데이터 확인 및 EDA

- Looker Studio를 사용하여 데이터를 확인하고 EDA를 하는 방법
    - 전체 데이터 가져와서 차트 만들기
    - Big Query를 사용하여 SQL로 데이터를 정제한 후 Looker Studio에 SQL로 차트 만들기

### 전체 데이터를 가져와서 차트 만들기

- Big Query를 사용할 필요가 없다.
- 버튼으로 간단하게 조작이 가능하다.
- 지표당 간단한 합계, 평균, 집계, 최솟값, 최대값, 중앙값, 표준편차, 분산 까지의 연산은 버튼식으로 수정할수 있으며, 함수 또한 필드추가의 항목으로 만드는 것이 가능하다.
- GCP에 데이터 세트가 업로드 되어 있다면, Looker Studio를 바로 열어서 `데이터 추가` 버튼을 누른다.
    
    ![ ](images/GCP_LookerStudio/Untitled.png)
    
- Big Query를 누르고 GCP 데이터 세트를 찾는다. public data set을 사용하는 경우라면 공개 데이터 집합으로 들어가야 한다.
    
    ![ ](images/GCP_LookerStudio/Untitled%201.png)
    
- 추가 버튼을 누르면 화면 왼쪽에 데이터 세트의 이름과 그 안의 스키마 정보들이 뜬다.
- 원하는 대시보드를 만들기 위해 차트나 표를 추가하여 자유롭게 대시보드를 구성하면 된다.
- 예시: [https://lookerstudio.google.com/reporting/10e2c716-6289-4e07-a2de-da6cdac415b6](https://lookerstudio.google.com/reporting/10e2c716-6289-4e07-a2de-da6cdac415b6)
    
    ![ ](images/GCP_LookerStudio/Untitled%202.png)
    
    ![ ](images/GCP_LookerStudio/Untitled%203.png)
    

### Big Query를 사용하여 SQL로 데이터를 정제한 후 Looker Studio에 SQL로 차트 만들기

- 표에 있는 지표를 가공하여 보고 싶을때 유용하다.
- 버튼으로 조작하는 것보다 더 많은 지표 표현이 가능하다.
- 유지 보수의 어려움이 있다.
- 사용할 데이터의 지표들을 확인해보면 아래처럼 나타낼 수 있다.
    - age(나이): 개인의 나이를 연단위로 나타냅니다
    - workclass(노동 계급): 개인의 고용형태
        - `Private`, `?`, `Local-gov` , `Self-emp-inc`, `Federal-gov`, `State-gov`, `Self-emp-not-inc`, `Never-worked`, `Witout-pay`
    - functional_weight: 일련의 관측결과를 바탕으로 인구조사국이 부여하는 개인의 가중치
    - education: 개인의 최종학력
    - education_num: 교육수준을 숫자로 범주화 하여 열거 합니다. 숫자가 높을수록 개인의 교육수준이 높습니다.
        - 11: Assoc_voc: 전문학교 준학사
        - 13: Bachelors: 학사
        - 9: HS-grad: 고등학교 졸업
    - marital_status: 개인의 결혼 여부 입니다.
        - `Married-civ-spouse`, `Divorced`, `Never-married`, `Separated`, `Widowed`, `Married-spouse-absent`, `Married-AF-spouse`
    - occupation: 개인의 직업입니다.
    - relationship: 가정 내 각 개인의 관계입니다.
        - `Wife`, `Own-child`, `Husband`, `Not-in-family`, `Other-relative`, `Unmarried`
    - race: 인종을 나타냅니다
        - `White`, `Asian-Pac-Islander,` `Amer-Indian-Eskimo`, `Black`, `Other`
    - sex: 개인의 성별입니다.
        - `Female`, `Male`
    - capital_gain: 개인의 자본 이익을 미국 달러로 표기 합니다.
    - capital_loss: 개인의 자본 손실을 미국 달러로 표기 합니다.
    - hours_per_week: 주당 근무시간입니다.
    - native_country: 개인의 출신 국가 입니다.
        - `?`,`Cambodia`,`Canada`,`China`,`Columbia`,`Cuba`,`Dominican-Republic`,`Ecuador`,`El-Salvador`,`England`,`France`,`Germany`,`Greece`,`Guatemala`,`Haiti`,`Holand-Netherlands`,`Honduras`,`Hong`,`Hungary`,`India`,`Iran`,`Ireland`,`Italy`,`Jamaica`,`Japan`,`Laos`,`Mexico`,`Nicaragua`,`Outlying-US(Guam-USVI-etc)`,`Peru`,`Philippines`,`Poland`,`Portugal`,`Puerto-Rico`,`Scotland`,`South`,`Taiwan`,`Thailand`,`Trinadad&Tobago`,`United-States`,`Vietnam`,`Yugoslavia`
    - income_bracket: 개인의 연간 소득이 미화 50,000달러가 넘는지 여부를 나타냅니다
- 예측을 위한 가설들을 세운다
    - 예시: native_country를 기준으로 평균 주당 근무시간과 개인의 자본 현황의 평균을 보고 싶다.
        - 이러한 데이터는 Looker studio의 기능으로 조회가 불가능하다. → GCP Big Query로 조회하기
- 실행 쿼리
    
    ```sql
    SELECT DISTINCT native_country,
     AVG(hours_per_week) as avg_hours_per_week, 
     AVG(capital_gain - capital_loss) as avg_capital
    FROM `bigquery-public-data.ml_datasets.census_adult_income`
    GROUP BY 1
    ORDER BY 1
    ```
    
    ![ ](images/GCP_LookerStudio/Untitled%204.png)
    
- 위 데이터를 Looker Studio로 옮겨서 그래프를 작성해 보자
- 데이터 추가 → 빅쿼리 → 맞춤 검색어 → 프로젝트를 선택한 후 SQL을 입력해 준다.
    - 시간 데이터가 있을 때는 기간 매개변수 사용 설정을 체크 하면 기간 컨트롤에 대한 부분을 조작할 수 있다. → 하지만 이 데이터에는 기간에 대한 내용은 없으니 추후 작성하는 것으로 한다.
    
    ![ ](images/GCP_LookerStudio/Untitled%205.png)
    
- 데이터를 추가 하고 차트를 추가하면 대시보드가 완성 된다.
    
    ![ ](images/GCP_LookerStudio/Untitled%206.png)
    

---

- Reference
    - [https://notebook.community/google/eng-edu/ml/cc/exercises/estimators/ko/intro_to_fairness](https://notebook.community/google/eng-edu/ml/cc/exercises/estimators/ko/intro_to_fairness)