---
title: Ensemble Model
date: 2022-09-15
categories:
  - Data Analysis
  - Model 
tags: 
  - Ensemble Model
  - ML Analysis
  - Python
---

### 1. Ensemble Model

`어떠한 한 현상에 대한 답을 얻는다고 가정해보자, 많은 경우에 한 명의 전문가보다 여려 명의 일반인들의 의견이 더 나은 경우가 있다.`

- 위 예제와 비슷하게, 하나의 좋은 모형(회귀,분류)으로부터 예측을 하는 것보다 여러 개의 모형으로부터 예측을 수집하는 것이 더 좋은 예측을 할 수 있다.
- 이러한 여러 개의 모형을 **앙상블**이라고 부르고, 여러 개의 모형을 조화롭게 학습시키는 것을 **앙상블 학습**이라고 한다.
- 그리고 6주차에서 배운 결정 트리 모형이 하나가 아니라, 훈련 세트를 무작위로 다른 서브셋으로 만들어서 결정 트리 분류기를 만들고, 많은 모형들 중에서 가장 많은 선택을 받은 클래스를 예측하는 앙상블 모형을 **랜덤포레스트**라고 한다.
- 오늘날의 랜덤포레스트 모델은 가장 강력한 머신러닝 알고리즘 하나이다.
- 그리고 머신러닝 대회에서 우승하는 솔루션들은 대부분 앙상블 방법을 사용하여서 최고 성능을 낸다.
- 뒤에서 앙상블 방법들 중 **배깅**을 설명할 것이다.
- 투표 기반 분류기
    
    ![](images/Ensemble/image.png)
    
    - 하나의 데이터셋을 여러종류의 분류기들로 훈련시켰다고 가정해보자.
    - 위에서 언급한대로 하나의 좋은 모델을 사용하는 것보다, 여러 종류의 분류기들이 가장 많이 예측한 클래스를 예측하는 것이 더 좋은 분류기를 만드는 매우 간단한 방법이다.
    - 이렇게 다수결의 투표로 정해지는 분류기를 **hard voting(집접 투표)** 분류기라고 한다.
    - 놀랍게도 위 모델 중 가장 성능이 좋은 모델의 정확도보다 다수결을 통해 예측한 앙상블 모델의 성능이 높은 경우가 많다.
    - 이렇게 랜덤 추측보다 조금 더 높은 성능을 내는 **weak learner(약한 학습기)** 가 충분히 많고 다양하다면 strong learner(강한 학습기)가 될 수 있다.
    
    ![](images/Ensemble/image2.png)
    
    
    `어떻게 약한 학습기가 강한 학습기가 되어서 더 좋은 성능을 낼 수 있을까?, 이 질문은 "큰 수의 법칙"으로 설명될 수 있다.`
    
    
    - 먼저, 50:50의 동전이 아니라, 51:49의 불균형하게 앞면과 뒷면이 나오는 동전이 있다고 가정을 해보자.
    - 이 동전을 1,000번을 던진다면 거의 앞면 510번과 뒷면 490번이 나올 것이다.
    - 수학적으로 1,000번을 던졌을 때 앞면이 더 많게 나오는 확률은 거의 75% 정도 된다.
    - 수학적으로 10,000번을 던졌을 때 앞면이 더 많게 나오는 확률은 거의 97% 정도 된다.
    - 위 수학적 계산은 이항분포의 확률 질량 함수로 계산 가능하다. ex) 1-scipy.stats.binom.cdf(499,1000,0.51) = 0.747
    - 위의 내용을 기반으로 우리의 약한 분류기(51%) 1,000개로 앙상블 모형을 구축하고, 가장 많은 클래스를 예측으로 삼는다면 75%, 10,000개로 모형을 만들면 97% 정도의 성능을 낼 수 있다.
    - 하지만..! 위의 과정은 모든 분류기가 완벽하게 독립이고, 모델의 예측 오차에 대해서 상관관계가 없을때만 가능하다.
    - 🌞 TIP : 앙상블에서 예측기가 가능한 서로 독립일 때 최고 성능을 발휘한다. 그래서 가능한 다양한 알고리즘을 사용해서 학습을 하면 다양한 종류의 오차를 만들기 때문에 앙상블 모델의 성능을 높일 수 있다.
    - 여러 종류의 알고리즘을 사용해서 **투표기반 분류기**를 만드는 예제를 해보자.
    
    ```python
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier,VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    # 데이터셋 로드
    iris = load_iris()
    X = iris.data[:,2:] # 꽃잎의 길이, 너비
    Y = iris.target
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=2021,shuffle=True)
    
    # 약한 학습기 구축
    log_model = LogisticRegression()
    rnd_model = RandomForestClassifier()
    svm_model = SVC()
    
    # 앙상블 모델 구축
    # 만약에 모든 모델이 predict_proba() 메서드가 있으면, 예측의 평균을 내어 soft voting(간접 투표)도 할수 있다.
    # 간접 투표 방식은 확률이 높은 투표에 비중을 두기 때문에 성능이 더 높다. (voting='soft' 사용)
    # svc는 기본적으로 predict_proba를 제공하지 않아, probability = True 지정 해야 사용 가능
    # 대신 svc에서 probability = True를 지정하면 교차 검증을 사용해서 확률을 추정하기 때문에 훈련 속도 느려짐
    # 대신 성능을 올라감
    voting_model = VotingClassifier(
        estimators=[('lr',log_model),('rf',rnd_model),('svc',svm_model)], # 3개의 약한 학습기
        voting='hard' # 직접 투표(hard voting)
    )
    
    # 앙상블 모델 학습
    voting_model.fit(x_train,y_train)
    
    # 모델 비교
    for model in (log_model,rnd_model,svm_model,voting_model):
      model.fit(x_train,y_train)
      y_pred = model.predict(x_test)
      print(model.__class__.__name__," : ",accuracy_score(y_test,y_pred))
    
    > LogisticRegression  :  1.0
      RandomForestClassifier  :  0.9555555555555556
      SVC  :  1.0
      VotingClassifier  :  1.0
    ```
    
### 2.배깅과 페이스팅
  - 앙상블 모형의 좋은 성능을 내기 위해서는 다양한 종류의 오차를 만들어야 하고, 그러기 위해서는 다양한 알고리즘을 사용해야 한다고 배웠다.
  - 다양한 오차를 만들기위한 다른 하나의 방법으로는 훈련 세트의 서브셋을 무작위로 구성하여 모델을 학습시키는 것이 있다. 이를 **배깅**과 **페이스팅**이라고 부른다.
  - **배깅** : 훈련 세트의 중복을 허용하여 샘플링을 하는 방식 (통계학에서는 "부트스트래핑"이라고도 부름)
  - **페이스팅** : 훈련 세트의 중복을 허용하지 않고 샘플링 하는 방식
  - 배깅은 각 예측기가 학습하는 서브셋에 다양성을 증가시키므로 페이스팅보다 편향이 조금 더 높다.
  - 하지만 배깅은 예측기들의 상관관계를 줄이므로 앙상블의 분산을 감소 시킨다.
  - 전반적으로 배깅이 더 나은 모델을 만들지만, 시간과 장비가 좋다면 교차검증으로 배깅과 페이스팅을 둘다 해보면 좋다.
  
    **1. 사이킷런의 배깅과 페이스팅**
    
    ```python
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    
    # 모델 구축
    # BaggingClassifier에서 사용한 분류기가 클래스 확률추정(predict_proba)이 가능하면 자동으로 간접 투표 사용 
    bag_model = BaggingClassifier(
        DecisionTreeClassifier(), # 약한 학습기(결정 트리)
        n_estimators=500, # 약한 학습기(결정 트리) 500개 생성
        max_samples=0.05, # 0.0~1.0 사이 실수 선택(실수 x 샘플 수) 혹은 샘플수 지정
        bootstrap=True, # True : 배깅, False : 페이스팅
        n_jobs=-1 # 훈련과 예측에 사용할 CPU 코어 수 (-1 : 가용한 모든 코어 사용)
    )
    
    # 모델 학습
    bag_model.fit(x_train,y_train)
    
    # 모델 예측
    y_pred = bag_model.predict(x_test)
    
    # 모델 평가
    print(bag_model.__class__.__name__," : ",accuracy_score(y_test,y_pred))
    > BaggingClassifier  :  0.9777777777777777
    ```
    
    ![](images/Ensemble/image3.png)
    
    - 단일 결정 트리와 배깅을 사용한 결정트리 앙상블의 결정경계를 비교해보면 트리 앙상블이 더욱 일반화가 잘 된것을 확인할 수 있다.
    
    **2. oob 평가**
    
    - 배깅(중복 허용 샘플링)을 하다보면 평균적으로 훈련 샘플의 약 63%정도만 추출되고 나머지 약 37%는 추출되지 않고, 이렇게 추출되지 않은 샘플들을 oob(out-of-bag)샘플이라고 부른다.
    - 예측기가 훈련되는 동안에는 oob샘플을 사용하지 않으므로, 검증 세트나 교차 검증을 사용하지 않고 oob샘플만을 가지고 모델 최적화를 위한 평가를 할 수 있다.
    - 앙상블의 평가는 각 예측기의 oob평가의 평균으로 확인한다.
    
    ```python
    # 모델 구축
    bag_model = BaggingClassifier(
        base_estimator = DecisionTreeClassifier(),
        n_estimators = 500,
        bootstrap = True,
        n_jobs = -1,
        oob_score = True # oob평가를 위해 True를 지정한다.
    )
    
    # 모델 학습
    bag_model.fit(x_train,y_train)
    
    # 모델 평가(oob_score_)
    print('oob_score : ',bag_model.oob_score_)
    
    # 모델 평가
    y_pred = bag_model.predict(x_test)
    print('test_score : ',accuracy_score(y_test,y_pred))
    >oob_score  :  0.9523809523809523
     test_score :  0.9333333333333333
    ```
    
### 3. 랜덤 포레스트
  - 랜덤포레스트는 일반적으로 배깅방법을 사용한 결정트리 앙상블 모델이다.
  - 그래서 BaggingClassifier에 DecisionTreeClassifier를 넣는 대신, RandomForestClassifier를 사용할 수 있다.
  - 그래서 RandomForestClassifier는 DecisionTreeClassifier와 BaggingClassifier 매개변수 모두 가지고 있다.
  - 랜덤포레스트 모델은 트리의 노드를 분할할 때 전체 특성 중에서 최선의 특성을 찾는 것이 아니라, 무작위로 선택한 특성들 중에서 최선의 특성을 찾는 방식을 채택하여 무작위성을 더 가지게 된다.
  - 이를 통해 약간의 편향은 손해보지만, 더욱 다양한 트리를 만들므로 분산을 전체적으로 낮추어서 더 훌륭한 모델을 만들 수 있다.
    
    ```python
    from sklearn.ensemble import RandomForestClassifier
    
    # 랜덤포레스트 모델 구축
    rnd_model = RandomForestClassifier(
        n_estimators = 500, # 예측기 500개
        max_leaf_nodes = 16, # 자식노드의 최대 개수 
        n_jobs = -1 # CPU 코어 구동 개수
    )
    
    # 모델 학습
    rnd_model.fit(x_train,y_train)
    
    # 모델 예측
    y_pred_rf = rnd_model.predict(x_test)
    
    # 모델 평가
    print("rnd_model : ",accuracy_score(y_pred_rf,y_test))
    > rnd_model :  0.9333333333333333
    ```
    
---
- Reference
    - [https://velog.io/@changhtun1/ensemble#-랜덤-포레스트](https://velog.io/@changhtun1/ensemble#-%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8)