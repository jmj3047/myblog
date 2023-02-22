---
title: Difference between Normal Distribution & Standard Normal Distribution
date: 2022-11-11
categories:
  - Basic ML
tags: 
  - Normal Distribution
  - Standard Normal Distribution
  - ML Analysis
---
### Difference between Normal Distribution & Standard Normal Distribution

본 포스팅에서는 정규분포(Normal distribution)와 표준 정규 분포(Standard normal distribution)에 대해 다루도록 한다. 정규 분포의 확률밀도 함수와 예상치(평균), 분산 그리고 증명에 대해 다루며 표준정규분포에 대해서는 확률밀도함수, 누적분포함수, 그리고 표준정규분포를 이용한 정규분포의 확률계산 등의 내용이 다뤄진다. 

### 1. 정규분포(Normal distribution)

**정규 분포(Normal distribution)**는 연속확률분포 중 하나이며 광범위하게 사용된다. 

- 확률 분포 중 가장 유명하며 가장 중요하게 다루는 확률 분포이다.
- **오류 분포(Error distribution)**와 다른 많은 자연현상을 직접 모델링 하기 위한 확률 분포이다.
- 중심극한정리(Central Limit Theorm)로 인해 매우 유용하며 단순하고 정확한 근사가 가능하다.
- 정규 분포는 가우스 분포(Gaussian distribution)라고도 불린다.
- 비율과 개별적인 확률을 모델링하는데도 유용하다.
- 정규분포는 일반적으로 다음과 같이 표현된다. : **X~N(μ, σ2)**
- 아래 그림은 정규 분포의 확률밀도 함수를 보여준다

![](images/Normal_Distribution/Untitled.png)

****************************확률밀도함수(PDF)****************************
정규분포의 확률 밀도 함수는 다음과 같다

![](images/Normal_Distribution/Untitled%201.png)

********************예상치(Expectation)와 분산(Variance)********************

![](images/Normal_Distribution/Untitled%202.png)

 **********정규분포의 적률생성함수(MGF)**********

![](images/Normal_Distribution/Untitled%203.png)

********예상치와 분산의 증명********

본 증명에서는 적률생성함수(MGF)를 이용하여 증명을 수행해보도록 하겠다. 

적률생성함수는 다음과 같다

![](images/Normal_Distribution/Untitled%204.png)

적률생성함수의 미분값과 그 미분값에서 t=0인 경우는 다음과 같다

![](images/Normal_Distribution/Untitled%205.png)

적률생성함수의 2계 미분값과 그 미분값에서 t=0인 경우는 다음과 같다

![](images/Normal_Distribution/Untitled%206.png)

따라서 분산은 다음과 같이 계산된다

![](images/Normal_Distribution/Untitled%207.png)

## 표준정규분포(Standard Normal distribution)

정규 분포에서는 **μ**는 0으로 **σ2**은 1로 설정하여 표준화를 수행한 정규분포를 **표준정규분포**라고 한다.

- 즉, 예상치(평균)는 0, 분산은 1이다.
- 아래 그림은 표준 정규 분포의 확률밀도함수와 누적분포함수를 보여준다.

![](images/Normal_Distribution/Untitled%208.png)

****************************확률밀도함수(PDF)****************************

표준정규분포의 확률밀도함수는 다음과 같다. 

![](images/Normal_Distribution/Untitled%209.png)

**누적분포함수(CDF)**

표준정규분포의 누적분포함수는 다음과 같다.

![](images/Normal_Distribution/Untitled%2010.png)

**정규분포의 적률생성함수(MGF)**

![](images/Normal_Distribution/Untitled%2011.png)

### 표준정규분포를 이용한 정규분포의 확률계산

확률분포가 정규분포인 경우 표준정규분포로 치환하여 특정범위의 확률을 계산할수 있다. 

- 치환은 다음과 같이 수행 하면 된다.

![](images/Normal_Distribution/Untitled%2012.png)

표준정규분포의 누적확률분포(CDF) 값은 비교적 쉽게 계산이 가능하며 다음과 같은 방식으로 계산할 수 있다. 

- 표준 정규 분포의 누적확률 분포(CDF) 표
- Excel, R, Python 모듈 등을 활용한 데이터 취득

다음은 정규분포의 특정 범위에서에 대한 확률을 표준정규분포로 치환하는 과정을 보여준다. 

![](images/Normal_Distribution/Untitled%2013.png)

정규분포는 중앙값에서 좌우 대칭이다. 

- 따라서 중앙값에서 다음과 같이 특정 범위의 확률을 표현하는 것이 가능하다.
- 또한 표준 편차를 이용하여 표준화 시켜 특정 범위를 표현할 수 있다.

![](images/Normal_Distribution/Untitled%2014.png)


---
- Reference
    - [https://color-change.tistory.com/m/61](https://color-change.tistory.com/m/61)
    - [https://kongdols-room.tistory.com/145](https://kongdols-room.tistory.com/145)