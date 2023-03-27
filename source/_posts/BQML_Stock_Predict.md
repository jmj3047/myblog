---
title: BQML로 게임사 주가 예측하기
date: 2023-03-20
categories:
  - Data Platform/Base
  - GCP
tags: 
  - Big Query
  - BigQueryML
  - ML Analysis
  - Looker Studio
---

## 개요

- 데이터로 BQML을 통해서 주가 예측을 해보자
- 넥슨, 컴투스, 넷마블, nc 소프트의 주가를 예측
- 5년동안 데이터로 학습하고 넥슨 게임사의 2023-03월달의 주가 예측을 시행함
- 데이터 흐름: API 크롤링 → 빅쿼리, 빅쿼리ML → Looker Studio

## 목적

- 5년치 주가 데이터를 활용힌 넥슨 게임즈의 주가 등락 예측과 경쟁 3사와의 비교
- BQML 만으로 괜찮은 결과가 나올수 있는지 확인
- 모델들 비교를 해보며 주제에 어떤 모델이 적합할지 확인

## 데이터

- [데이터 크롤링 → 빅쿼리 적재](https://velog.io/@kallroo/gcp-%EC%A3%BC%EC%8B%9D%EB%8D%B0%EC%9D%B4%ED%84%B0-%ED%81%AC%EB%A1%A4%EB%A7%81-%ED%95%B4%EC%84%9C-bigquery%EC%97%90-%EC%A0%81%EC%9E%AC%ED%95%98%EA%B8%B0-1) 하는 방법
- 대상 회사 정의
    - 2018년 이전에 상장한 게임 회사
    - 매출 1~5위 순위 중 위 조건을 충족하는 게임사는 넥슨, 넷마블, ncsoft 였으며, 그 외 순위의 위 조건을 충족하는 컴투스까지 포함하여 총 4개 게임사
    - 게임사 선택 이유: 경쟁사가 많고, 매출 기준으로 순위권 회사들이 상장을 많이 함.
- 피처 설명
    
    ![ ](images/BQML_Stock_Predict/Untitled.png)
    
    - 5년치 데이터를 가져 올수 있는 파이썬 api중 가장 많은 데이터를 가져올수 있는 marcap data를 사용하였음
    - Adj_Close(수정종가) 피처는 yfinance에서만 있었고, 모든 날짜에 대해서 있는건 아니지만, 이를 통해서 액면분할 된 주가를 모델에 제대로 반영할수 있어서 추가 하게 되었음
    - marcap데이터 수집 당시 2월 말까지만 수집을 하였고 이로 인해 3월 예측 데이터를 finance data reader로 진행함
    - ml_label: 훈련을 수월하게 하기 위해 임의로 만듦, volume을 기준으로 training:evaluation:prediction = 8:1:1로 나누고 prediction 데이터는 3월 데이터 까지 포함되어 있음.
- 데이터 개수
    
    ![ ](images/BQML_Stock_Predict/Untitled%201.png)
    
- marcap + yfinance 의 데이터를 8:1:1로 나누고 예측 데이터에 finance data reader에서 2023-03 기간동안의 데이터를 합쳐서 학습을 진행
- 전일대비 등락률로 예측을 진행하였고 정확한 수치가 아닌 양수면 오름, 음수면 내림으로 환산하여 예측의 정확도를 판별
- 데이터 기간
    - marcap: 2018-01-02 ~ 2023-02-28
    - yfinance: 2018-01-02 ~ 2023-02-28
    - finance data reader: 2023-03-02 ~ 2023-03-21

## 모델 선택 및 비교

- Big Query ML에서 추천하는 주식 데이터 예측 모델로는 Linear Regression, Boosted Trees Regressor, Auto ML Table Regressor, DNN Regressor, Wide & Deep Regressor 이 있음
- 총 데이터가 만개를 넘지 않을정도로 적고(5104행), 시간이 오래 걸리지 않는 Linear Regression, Boosted Trees Regression 모델을 사용하여 예측
- 모델 예측 결과 Linear Regression이 모델 학습 속도는 적게 걸리나 오류나 과적합 면에서 Boosted Trees Regression의 결과가 훨씬 나음
- 실제 등락이 크게 오르거나 내려갈때의 값을 Linear Regression 모델 보다 Boosted Trees Regression 모델이 더 잘 예측함
    
    ![ ](images/BQML_Stock_Predict/Untitled%202.png)
    
- Model Evaluation 값
    
    ![ ](images/BQML_Stock_Predict/Untitled%203.png)
    
- 지표 설명
    
    ![ ](images/BQML_Stock_Predict/Untitled%204.png)
    

## 예측

- ml_ label이 prediction으로 책정된 데이터를 예측한 결과와 실제 Chages Ratio 값을 비교
- 정확한 수치를 비교하는것이 아닌 등락으로 비교 -> 비율이 양수라면 값이 오른것, 음수라면 내린것으로 간주
    
    ![ ](images/BQML_Stock_Predict/Untitled%205.png)
    
- feature importance
    
    ![ ](images/BQML_Stock_Predict/Untitled%206.png)
    

## 시각화

![ ](images/BQML_Stock_Predict/image.png)

## 결론

- 애초에 데이터부터 잘못 설계가 된 실험
- 전날 데이터를 갖고 다음날을 예측 해야 하는데 당일 데이터로 당일을 예측해 버림 → 안정확한게 더 이상한것

---
- Reference
    - [BigQuery ML을 사용하여 펭귄 체중 예측](https://cloud.google.com/bigquery-ml/docs/linear-regression-tutorial?hl=ko#step_four_evaluate_your_model)
    - [Mean_Absolute_Error](https://computer-nerd-coding.tistory.com/1)
    - [MAE,MSE,RMSE](https://mizykk.tistory.com/102)
    - [R2_Explained_Variance_Score](http://machinelearningkorea.com/2019/06/09/%EA%B2%B0%EC%A0%95%EA%B3%84%EC%88%98-r2%EC%99%80-%EC%84%A4%EB%AA%85%EB%B6%84%EC%82%B0%EC%A0%90%EC%88%98-explained-variance-score/)
