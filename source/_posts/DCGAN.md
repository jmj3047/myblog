---
title: Unsupervised representation learning with deep convolutional generative adversarial networks
date: 2022-04-21
categories:
  - Paper
  - Generative Model
tags: 
  - DCGAN
  - Convolutional Neural Networks
  - Image Classification
  - GAN
  - Deep/Machine Learning Paper Study
mathjax: true
---



Journal/Conference: ICLR
Year(published year): 2016
Author: Alec Radford, Luke Metz, Soumith Chintala
Subject: DCGAN, Generative Model

## Unsupervised representation learning with deep convolutional generative adversarial networks

> Summary

- CNN 과 GAN framework 를 결합한 DCGAN 모델을 제시

## **Introduction**

논문 당시 GAN 은 불안정한 학습과 Generator 의 오작동으로 인해 제한적으로만 쓰였습니다. 이에 논문은 이미지 생성 모델을 만들기 위한 CNN 기반 GAN framework인 DCGAN(Deep Convolutional GANs) 을 제시합니다.

- 모델 구조에 제약을 가하여 대부분의 상황에서 안정적으로 학습할 수 있게 함
- Discriminator 로 image classification 을 수행하였을 때 기타 비지도 학습 모델과 비슷한 성능을 보임
- 특정 필터가 특정 object를 그려낸다는 것을 시각화하여 제시함
- Generator 에 입력하는 noise 를 제어하여 생성되는 샘플의 다양한 속성이 변화하는 것을 탐구함

## **Approach and Model Architecture**

논문 이전에도 GAN에 CNN을 써서 이미지 품질을 높이려는 시도가 있었으나 좋은 성과를 거두지 못하였다고 설명합니다. 이후, 다양한 데이터셋에서 안정적이고 높은 해상도의 이미지를 생성하기 위한 DCGAN 모델 설계 가이드라인을 제시합니다.

![](images/DCGAN/Screen_Shot_2021-11-28_at_2.30.30_PM.png)

## **Details of Adversarial Training**

3가지 데이터셋을 사용합니다.

- Large-scale Scene Understanding(LSUN)
- Imagenet-1k
- Faces

그외에 학습 디테일을 아래와 같이 제시합니다.

- pre-processing 제외
- batch size 128
- 가중치는 N(0, 0.02) 로 초기화
- Leaky ReLU의 기울기는 0.2로 설정함
- AdamOptimizer, $\beta_1 =0.0002, \beta_2=0.9$

![](images/DCGAN/Screen_Shot_2021-11-28_at_2.33.24_PM.png)

Generator 구조의 모식도는 위와 같습니다.

![](https://greeksharifa.github.io/public/img/2019-03-18-DCGAN/02.png)

LSUN 데이터셋으로 1 epoch 를 학습시킨 후 침실을 생성했을 때의 결과입니다. 이론적으로 모델이 훈련 예시를 기억했을 수도 있으나, 작은 학습률과 미니배치를 사용했음을 감안할 때 가능성이 낮다고 설명합니다.

![](https://greeksharifa.github.io/public/img/2019-03-18-DCGAN/03.png)

LSUN 데이터셋으로 5 epoch 학습 후 침실을 생성한 결과입니다. 침대 등의 근처에서 오히려 underfitting 이 발생했음을 확인할 수 있습니다.

## **Empirical Validation of DCGANs Capabilities**

Unsupervised representation learning 알고리즘을 평가하는 일반적인 방법은 supervised 데이터셋으로 특징을 추출한 뒤 performance를 측정하는 것입니다.

![](images/DCGAN/Screen_Shot_2021-11-28_at_2.42.06_PM.png)

CIFAR-10 데이터셋에 대해 검증한 결과, 다른 방법들(K-means, Exemplar CNN 등)과 비교하여 결과에 큰 차이가 존재하지 않습니다.

![](images/DCGAN/Screen_Shot_2021-11-28_at_2.43.05_PM.png)

StreetView House Numbers dataset(SVHN) 데이터셋에서는 state-of-the-art 결과를 얻었음을 제시합니다.

## **Investigating and Visualizing the Internals of the Networks**

가장 가까운 학습 데이터 이미지를 찾거나, 최근접 픽셀/특징을 확인하거나 log-likelihood metric 으로 평가를 하는 방법은 모두 성능이 떨어지는 metric 이기에 사용하지 않았음을 언급합니다.

논문은 대신, 2개의 이미지를 생성할 때 사용한 noise 2개를 interpolation 하고, interpolated z 로 이미지를 생성한 결과를 제시합니다. 한 이미지에서 다른 이미지로 점진적으로 변해가는 모습을 관측할 수 있습니다.

![](https://greeksharifa.github.io/public/img/2019-03-18-DCGAN/04.png)

또한 노이즈 벡터 z 의 산술 연산을 통해, vec(웃는 여자) −− vec(무표정 여자) ++ vec(무표정 남자) == vec(웃는 남자) 같은 결과를 얻을 수 있었음을 제시합니다.

![](https://greeksharifa.github.io/public/img/2019-03-18-DCGAN/05.png)

![](https://greeksharifa.github.io/public/img/2019-03-18-DCGAN/06.png)

또한, 랜덤하게 생성한 필터와 학습된 필터의 activation 을 아래와 같이 시각화 하였습니다. 이해할 수 없는 feature 가 아닌 특정 object나 특징을 추출하고 있음을 확인할 수 있습니다.

![](https://greeksharifa.github.io/public/img/2019-03-18-DCGAN/07.png)

### **Conclusions and future work**

논문은 CNN 기반의 안정적인 이미지 생성모델인 DCGAN을 제안하였으며, image representation에 적합한 성능을 보임을 제시합니다. 그러나 여전히, 학습이 길어지는 경우 필터 일부가 요동치는 등의 현상을 관측하기도 하였음을 언급합니다.

---

- Link: **[Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/abs/1511.06434)**
