---
title: Transductive learning VS Inductive Learning
date: 2024-02-20
categories:
  - Paper
  - Few Shot Learning
tags: 
  - Transductive Learning
  - Inductive Learning
---

### Inductive Learning

- 학습 데이터를 이용하여 학습한 패턴이나 지식을 습득하여 보지 못한 데이터(평가 데이터)들에 대해서예측하기 위한 방법
- 모델을 일반화하여 예측을 잘 할 수 있게 하는 방법
- supervised learning
- 학습된 모델이 이전에 보지 못한 새로운 데이터에 대해 일반화 할 수 있도록 학습하는 것을 의미
- 주어진 훈련 데이터에서 일반적인 규칙이나 패턴을 추출하여 새로운 데이터에 대한 예측을 수행하는 것을 목표
- 분류 문제에서 고양이와 개의 이미지를 사용하여 학습한 후 새로운 이미지가 고양이인지 개인지 분류 하는 것
- ***induction*** ***is reasoning*** from observed training cases to ***general rules***, which are then applied to the test cases.
- 모델을 설계한 뒤 오직 training data만을 사용하여 분류 혹은 회귀를 위한 규칙(rule)을 스스로 추론
- 관측된 데이터는 training dataset 뿐이며 학습을 통해 training dataset분포에 최적화된 모델 파라미터를 계산, 이후 학습이 끝난뒤 unlabeled testing dataset의 label을 추론하는 것
- 이미 레이블링된 훈련 데이터 셋을 활용하여 학습을 진행, 이렇게 구축된 모델은 훈련 데이터 셋에서 본 적이 없는 테스트 데이터 셋의 레이블을 예측하는데 사용

### Transductive Learning

- 주어진 데이터에 대해 예측하도록 모델을 구축
- 기존에 학습된 모델을 가지고 새로운 데이터를 예측하는 것이 아니라 새로운 데이터가 들어오면 그것을 함께 반영해 다시 모델을 구착할 수 있음
- semi supervised learning
- 훈련 데이터와 동시에 테스트 데이터를 고려하여 학습하는 방법
- 학습 데이터와 테스트 데이터를 함께 고려하여 모델을 구축하고 특정 테스트 데이터에 대한 예측을 수행
- 레이블이 없는 데이터에 대해 레이블을 예측하는 것
- In logic, statistical inference, and supervised learning, ***Transduction*** is reasoning from observed specific (training) cases to specific (test) cases.
- 사전에 미리 training dataset 뿐만 아니라 testing dataset도 알고 있는 상태이며, testing datset의 label을 알지는 못하지만 학습이 진행되는 동안 labeled data (training data)의 특징 공유 혹은 전파, 데이터 간의 연관성, 특징 패턴 등 추가적인 정보를 활용함으로써 testing dataset의 label을 추론
- 훈련데이터와 테스트 데이터를 레이블의 유무로 나누지 않고 train test split으로 레이블이 있던 없던 모두 사용하고 랜덤으로 훈련/테스트 셋을 나눔, 그리고 레이블이 없는 데이터에서 비지도학습과 같이 숨어 있는 패턴을 추출하도록 학습

![ ](images/Transductive_Inductive/Untitled.png)

---

Reference

- [https://www.inflearn.com/questions/896304/transductive-learning-amp-inductive-learning](https://www.inflearn.com/questions/896304/transductive-learning-amp-inductive-learning)
- Yones, C., Georgina Stegmayer, and Diego H. Milone. "Genome-wide pre-miRNA discovery from few labeled examples." *Bioinformatics* 34.4 (2018): 541-549
- [https://dodonam.tistory.com/476](https://dodonam.tistory.com/476)
- [https://velog.io/@kimdyun/Inductive-Transductive-Learning-차이점](https://velog.io/@kimdyun/Inductive-Transductive-Learning-%EC%B0%A8%EC%9D%B4%EC%A0%90)
- [https://blog.naver.com/song_gina/222149366893](https://blog.naver.com/song_gina/222149366893)
- [https://newsight.tistory.com/313](https://newsight.tistory.com/313)
- [https://dos-tacos.github.io/translation/transductive-learning/](https://dos-tacos.github.io/translation/transductive-learning/)