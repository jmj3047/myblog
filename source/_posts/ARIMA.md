---
title: ARIMA model
date: 2023-05-16
categories:
  - Data Analysis
  - Model 
tags: 
  - ARIMA
  - ML Analysis
---


### 정상성(stationarity)

- 시계열은 시계열의 특징이 해당 시계열이 관측된 시간에 무관
- 추세나 계절성이 있는 시계열은 정상성을 나타내는 시계열이 아님 → 추세와 계절성은 서로 다른 시간에 시계열의 값에 영향을 줄 것이기 때문.
- 백색 잡음(white noise) 시계열: 정상성을 나타내는 시계열 → 언제 관찰하는지에 상관이 없고 시간에 따라 어떤 시점에서 보더라도 똑같이 보일것이기 때문.

![ ](images/ARIMA/Untitled.png)

- (a) 200 거래일 동안의 구글 주식 가격; (b) 200 거래일 동안의 구글 주식 가격의 일일 변동; (c) 미국의 연간 파업 수; (d) 미국에서 판매되는 새로운 단독 주택의 월별 판매액; (e) 미국에서 계란 12개의 연간 가격 (고정 달러); (f) 호주 빅토리아 주에서 매월 도살한 돼지의 전체 수; (g) 캐나다 북서부의 맥킨지 강 지역에서 연간 포획된 스라소니의 전체 수; (h) 호주 월별 맥주 생산량; (i) 호주 월별 전기 생산량.
- 분명하게 계절성이 보이는 (d), (h), (i)는 후보가 되지 못합니다. 추세가 있고 수준이 변하는 (a), (c), (e), (f), (i)도 후보가 되지 못합니다. 분산이 증가하는 (i)도 후보가 되지 못합니다. 그러면 (b)와 (g)만 정상성을 나타내는 시계열 후보로 남았습니다.
- 언뜻 보면 시계열 (g)에서 나타나는 뚜렷한 주기(cycle) 때문에 정상성을 나타내는 시계열이 아닌 것처럼 보일 수 있습니다. 하지만 이러한 주기는 불규칙적(aperiodic)입니다 — 먹이를 구하기 힘들만큼 살쾡이 개체수가 너무 많이 늘어나 번식을 멈춰서, 개체수가 작은 숫자로 줄어들고, 그 다음 먹이를 구할 수 있게 되어 개체수가 다시 늘어나는 식이기 때문입니다. 장기적으로 볼 때, 이러한 주기의 시작이나 끝은 예측할 수 없습니다. 따라서 이 시계열은 정상성을 나타내는 시계열입니다.

### ACF (Autocorrelation Function)

- 정상성 확인하는 방법 중 하나
    
    ![ ](images/ARIMA/Untitled%201.png)
    
    - x축은 lag, y축은 ACF
- lag=1: 한 시점 미룬 데이터와의 차이를 의미
- 자기 자신과 자기 자신 이전 데이터와의 correlation = Autocorrelation lag 1이다.
- lag=2: 자신 자신 데이터와 두 시점 미룬 데이터와의 correlation = Autocorrelation lag 2
- lag=20: 현재 데이터와 20 시점을 shift한 데이터와의 correlation = Autocorrelation lag 20
- 이 그래프에서는 5시점 shift한 것과 autocorrelation이 꽤 있는 것으로 보임
- ACF를 통해서 정상성을 알아보는 방법 정리
    - 일정한 패턴이 없거나 갑자기 떨어지는 패턴 => stationary
    - 일정하게 떨어지거나 올라갔다 내려갔다하면서 굉장히 천천히 떨어지는 패턴 => nonstationary

### Autoregressive (AR) Models

- dependent variable인 Y의 lag를 independent variables인 X로 사용하는 모델

![ ](images/ARIMA/Untitled%202.png)

- 첫번째 X가 바로 *yt*−1, 즉 y의 한 시점전 데이터, *yt*−2는 두 시점 전 데이터,..., *yt*−*p*는 p 시점 전 데이터를 의미한다.
- 자기자신을 X로 삼기때문에 *X*1 = *yt*−1, *X*2 =*yt*−2,..,*Xp* = *yt*−*p*로 생각하면 된다.
- *ϕ*0는 인털셉
- *yt*를 모델링할 때 *yt*의 lag된 변수들(자신의 과거 데이터)을 X 삼아서 회귀모델을 만듦 .(Auto = self라고 생각하면 됨)
- multiple regression model과 다른 점
    - 자기자신을 갖고 모델링을 하기 때문에 독립성이 없다.
    - *ϕ*0 (계수)를 추정할 때 일반적으로 사용했던 최소제곱법은 사용할 수 없다.

### Moving Average (MA) Models

![ ](images/ARIMA/Untitled%203.png)

- *yt*를 *ε*(엡실론=error)으로 모델링
- t시점의 데이터(*yt*)를 t 시점의 에러(*εt*)와 과거의 에러들로 표현
- 연속적인 에러 term으로 y와d의 관계를 모델링하는 방법

### Autoregressive and Moving Average (ARMA)

![ ](images/ARIMA/Untitled%204.png)

- AR과 MA을 합친 모델
- t시점의 데이터(*yt*)를 자기자신의 lag된 값들, t시점의 error와 전 시점의 error들로 표현함

### Autoregressive Integrated Moving Average (ARIMA)

- p: order of the AR part of the model
- d: order of differencing
- q: order of the MA part of the model
- differencing을 했다는 것을 "integrated"로 표현함
- I 라는 것은 differencing을 몇번 했는지를 의미
- p,d,q
    - AR → p
    - I → d
    - MA → q
- AR모델에서 p = independent variable의 개수를 나타냄
- MR모델에서 q, *θ*의 개수, 즉 파라미터의 개수를 나타냄
- d = 몇번 differencing을 했냐

### 시계열 데이터 모델을 구현할 때 주의해야 할 상황

- AR, MA, ARMA 이 모델들을 구현하기 위해서는 분석해야되는 데이터가 stationary해야된다. nonstationary인 경우, 이 모델들을 적용할 수 없다.
- 일상생활엔 nonstationary한 데이터들이 훨씬 더 많다. 따라서 stationary한 데이터로 바꾼 뒤에 이 모델링을 할 수 있다.
- 어떻게 nonstationary를 stationary로 바꾸는 가장 간단한 방법이 바로 differencing(차분)

### 차분(differencing)

- 현 시점 데이터에서 d 시점 이전 데이터를 뺀 것
- 연이은 관측값들의 차이를 계산하는 것

![ ](images/ARIMA/Untitled%205.png)

- 원래 데이터와 원래 데이터를 한번 shift한 것을 빼주면 결과가 나오는데 이것이 바로 첫번째 differencing한 결과이다.
    
    ![ ](images/ARIMA/Untitled%206.png)
    
    - 1차 차분이란 t시점의 데이터와 t-1시점의 데이터의 차이
    - 2차 차분이란 t시점의 데이터와 t-2시점의 데이터의 차이
    - d차 차분이란 t시점의 데이터와 t-d시점의 데이터의 차이
- X(원래 데이터)는 nonstationary여도 differencing을 한 결과(Y)는 stationary로 바뀔 확률이 매우 크다
- (a)의 구글(Google) 주식 가격이 정상성을 나타내는 시계열이 아니었지만 패널 (b)의 일별 변화는 정상성을 나타냈다는 것에 주목합시다. 이 그림은 정상성을 나타내지 않는 시계열을 정상성을 나타내도록 만드는 한 가지 방법을 나타냅니다.
    
    ![ ](images/ARIMA/Untitled%207.png)
    
- 로그 같은 변환은 시계열의 분산 변화를 일정하게 만드는데 도움이 될 수 있습니다. 차분(differencing)은 시계열의 수준에서 나타나는 변화를 제거하여 시계열의 평균 변화를 일정하게 만드는데 도움이 될 수 있습니다. 결과적으로 추세나 계절성이 제거(또는 감소)됩니다.
- 정상성을 나타내지 않는 시계열을 찾아낼 때 데이터의 시간 그래프를 살펴보는 것만큼, ACF 그래프도 유용합니다. 정상성을 나타내지 않는 데이터에서는 ACF가 느리게 감소하지만, 정상성을 나타내는 시계열에서는, ACF가 비교적 빠르게 0으로 떨어질 것입니다. 그리고 정상성을 나타내지 않는 데이터에서 r1 은 종종 큰 양수 값을 갖습니다.

![ ](images/ARIMA/Untitled%208.png)

### ARIMA - Order of Differencing

![ ](images/ARIMA/Untitled%209.png)

- 만약 original 데이터가 stationary이면 differencing은 필요없다.
- 만약 original 데이터가 constant average trend(일정하게 증가하거나 감소하는 패턴)이면 1차 차분이면 충분하다.
- 오른쪽의 그래프와 같이 더 복잡한 패턴을 가지고 있다면 2차 차분까지 가야된다.
- **대부분의 데이터가 2차 차분으로 충분하다.**
- 3차 차분까지 했을 때 stationary가 되는 데이터는 AR,MA,ARMA 모델로는 적합하지 않은 데이터라고 생각하면 됨
- **1st Differencing (1차 차분)**
    
    ![ ](images/ARIMA/Untitled%2010.png)
    
- **2nd Differencing (2차 차분)**
    
    ![ ](images/ARIMA/Untitled%2011.png)
    
- 1차와 2차의 차이가 없으므로 2차 차분까지 할 필요가 없어보임.
- nonstationary가 stationary로 변했는지 그냥 봤을 때는 잘 모르므로 ACF를 확인하자.
    
    ![ ](images/ARIMA/Untitled%2012.png)
    
    - 원래 데이터는 ACF에서 일정하게 감소하는 패턴
    - 1차 차분한 것은 일정한 패턴이 없다.

---

- Reference
    - [https://otexts.com/fppkr/stationarity.html](https://otexts.com/fppkr/stationarity.html)
    - [https://velog.io/@sjina0722/시계열분석-ARIMA-모델](https://velog.io/@sjina0722/%EC%8B%9C%EA%B3%84%EC%97%B4%EB%B6%84%EC%84%9D-ARIMA-%EB%AA%A8%EB%8D%B8)