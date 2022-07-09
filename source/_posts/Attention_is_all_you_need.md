---
title: Attention is all you need
date: 2022-05-10
categories:
  - NLP
tags: 
  - Transformer
  - Attention
  - Deep/Machine Learning Paper Study
mathjax: true
---

Journal/Conference: NIPS
Year(published year): 2017
Author: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
Subject: NLP


# Attention is all you need

> Summary

- Attention 만으로 시퀀셜 데이터를 분석하여 병렬화와 연산 속도 향상을 가능하게 한 새로운 모델 제시
- Seq2Seq 과 Attention 을 결합한 모델(Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly
learning to align and translate. CoRR, abs/1409.0473, 2014.)에서 한층 더 발전한 모델입니다.
- Recurrent model(재귀 구조)없이 Self-attention 만으로 구성한 첫번째 모델입니다.
- 재귀 구조 제거로 모델을 병렬화(Parallelization)하여 자연 언어 처리 학습/추론 시간을 획기적으로 단축시켰습니다.

## Introduction

기존의 자연언어 처리 모델은  RNN, LSTM, GLU 모델로 대표되는 재귀 모델(Recurrent Model)을 encoder-decoder 구조로 결합하는 seq2seq 과 같은 모델을 주로 사용하였습니다.

이러한 재귀 모델은 순차 처리로 인해서 병렬화가 어렵다는 약점이 있고, 메모리 크기가 제한되어 긴 문장을 처리하기도 어렵다는 단점이 있습니다.

이를 보완하기 위한 어텐션(attention) 매커니즘이 제안되었습니다. 어텐션은 재귀 과정에서 입력에서 출력까지의 거리가 길어지는 문제를 해결할 수 있어 입출력의 전역 의존성을 높여주었지만, 재귀 모델과 결합해서만 사용되어 왔습니다.

이에 해당 논문에서는, 어텐션만으로 모델을 구성하여 쉽게 병렬화 할 수 있고 자연언어처리 과제의 성능을 높인 Transformer 모델을 제시합니다. 이는 기존의 재귀 모델과 다르게 8대의 P100 GPU로 12시간 정도만 학습했음에도 당시 기준 SOTA를 달성하였습니다.

- WMT 2014 English-to-German Translation task -> 28.4 BLEU
- WMT 2014 English-to-French Translation task -> 41.0 BLEU

## Model Architecture

Transformer 모델은 seq2seq으로 대표되는 인코더-디코더 구조를 self-attention 으로 쌓은 뒤, fully connected layer 로 출력을 생성합니다.

### Encoder and Decoder Stacks

![](images/What_is_Transformer/Untitled%201.png)

인코더(Encoder)

- $N$(=6)개의 동일한 레이어로 구성
- 각 레이어는 2개의 하위 레이어로 구성
    - Multi-head self-attention
    - position-wise fully connected feed-forward
- 하위 레이어를 거칠 때마다 Residual connection(Resnet) 과 layer normalization 을 실행
- 각 레이어 출력의 크기는 $d_{model}$(=512)로 고정

디코더(Decoder)

- 인코더와 같이 $N$(=6)개의 동일한 레이어로 구성
- 인코더와 동일한 2개의 하위 레이어에 한가지를 더 추가하여 3개의 하위 레이어로 구성
    - Multi-head self-attention
    - position-wise fully connected feed-forward
    - 인코더의 출력으로 실행하는 multi-head attention
- 순차적으로 결과를 만들어 낼 수 있도록 Self-attention 레이어에 Masking 을추가 : $i$ 번째 출력을 만들 때, $i$번째보다 앞선 출력($i-1, i-2,\dots$) 만을 참고하도록 함

#### Attention

attention은 query와 key-value pair들을 output에 맵핑해주는 함수입니다. 출력은 values들의 weighted sum으로, value에 할당된 weight는 query와 대응되는 key의 compatibility function으로 계산합니다.

#### Scaled Dot-Product Attention

![](https://media.vlpt.us/images/emeraldgoose/post/90a63976-5d30-46e9-8158-81ae01f920fe/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-08-16%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2012.23.05.png)

여기서 사용하는 attention은 Scaled Dot-Product Attention(SDPA)라 부르는데, input은 dimension이 $d_k$인 query와 key, dimension이 $d_v$인 value들로 이루어집니다.

![](https://media.vlpt.us/images/emeraldgoose/post/3e2daa40-c1bc-496d-9aa5-fd9511ec527c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-08-15%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%207.29.09.png)

모든 query와 모든 key들에 대해 dot product로 계산되는데 각각의 결과에 *dk*로 나누어진다. 다음 value의 가중치를 얻기 위해 softmax 함수를 적용합니다.

attention 함수는 additive attention과 dot-product attention이 사용됩니다.

- additive attention은 single hidden layer와 함께 feed-forward network에 compatibility function을 계산하는데 사용됩니다.
- dot-product attention이 좀 더 빠르고 실제로 space-efficient합니다. 왜냐하면 optimized matrix multiplication code를 사용해서 구현되었기 때문입니다.

$d_k$가 작은 경우 additive attention이 dot product attention보다 성능이 좋습니다.

그러나 $d_k$가 큰 값인 경우 softmax 함수에서 기울기 변화가 거의 없는 영역으로 이동하기 때문에 dot product를 사용하면서 $d_k$으로 나누어 scaling을 적용했습니다.

#### Multi-Head Attention

![](https://media.vlpt.us/images/emeraldgoose/post/04d07224-90eb-44a6-aaf6-ff37a2d9273b/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-08-16%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2012.23.13.png)

$d_{model}$ dimension의 query, key, value 들로 하나의 attention을 수행하는 대신, query, key, value들에 각각 학습된 linear projection을 $h$번 수행하는 것이 더 좋습니다.

- 즉, *Q*,*K*,*V*에 각각 다른 weight를 곱해주는 것입니다.

![](https://media.vlpt.us/images/emeraldgoose/post/e6bce6af-0b88-4a62-93f3-1dfe4ce34052/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-08-16%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2012.28.38.png)

이때, projection이라 하는 이유는 각각의 값들이 parameter matrix와 곱해졌을 때, $d_k,d_v,d_{model}$차원으로 project되기 때문입니다. query, key, value들을 병렬적으로 attention function을 거쳐 dimension이 $d_v$인 output 값으로 나오게 됩니다.

이후 여러개의 head를 concatenate하고 다시 $W^O$와 projection하여 dimension이 $d_{model}$인 output 값으로 나오게 됩니다.

#### Position-wise Feed-Forward Networks

인코더와 디코더는 fully connected feed-forward network를 가지고 있습니다.

또한, 두 번의 linear transformations과 activation function ReLU로 구성되어집니다.

![](https://media.vlpt.us/images/emeraldgoose/post/9a356af9-ecfa-43a9-af1e-7d5d0fda273c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-08-16%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.05.52.png)

각각의 position마다 같은 $W,b$ 를 사용하지만 layer가 달라지면 다른 parameter를 사용합니다.

#### Positional Encoding

모델이 recurrence와 convolution을 사용하지 않기 때문에 문장안에 상대적인 혹은 절대적인 위치의 token들에 대한 정보를 주입해야만 했습니다.

이후 positional encoding이라는 것을 encoder와 decoder stack 밑 input embedding에 더해줬습니다.

positional encoding은 $d_{model}$인 dimension을 가지고 있기 때문에 둘을 더할 수 있습니다.

![](https://media.vlpt.us/images/emeraldgoose/post/6006bc0f-d136-4391-8b96-95db49cc7a6e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-08-16%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.38.59.png)

- $pos$는 position, $i$는 dimension

$pos$는 sequence에서 단어의 위치이고 해당 단어는 $i$에 0부터 $2d_{model}$까지 대입해 dimension이 $d_{model}$인 positional encoding vector를 얻을 수 있습니다.

### Why Self-Attention

---

Self-Attention을 사용하는 첫 번째 이유는 layer마다 total computational complexity가 작기 때문입니다.

두 번째 이유는 computation의 양이 parallelized하기때문에 sequential operation의 minimum으로 측정되기 때문입니다.

![](https://media.vlpt.us/images/emeraldgoose/post/618866d8-e4c4-4420-a385-a6a16d195a79/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-08-16%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%201.34.05.png)

세 번째 이유로는 네트워크에서의 long-range dependencies사이의 path length때문입니다. long-range dependencies를 학습하는 것은 많은 문장 번역 분야에서의 key challenge가 됩니다.

input sequence와 output sequence의 길이가 길어지면 두 position간의 거리가 멀어져 long-range dependencies를 학습하는데 어려워집니다.

테이블을 보면 Recurrent layer의 경우 Sequential operation에서 $O(n)$이 필요하지만 Self-Attention의 경우 상수시간에 실행될 수 있습니다.

또한 Self-Attention은 interpretable(설명가능한) model인 것이 이점입니다.

### Traning

---

#### Optimizer

Adam optimizer 에 파라미터로 $\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$ 를 사용했습니다.

학습동안 아래의 공식을 통해 learning rate를 변화시켰습니다.

![](https://media.vlpt.us/images/emeraldgoose/post/998beece-1bd7-44ef-a451-cd6685f6e396/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-08-17%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%209.25.56.png)

이는 warmup_step에 따라 linear하게 증가시키고 step number에 따라 square root한 값을 통해 점진적으로 줄여갔습니다. 그리고 warmup_step = 4000을 사용했습니다.

#### Residual Dropout

각 sub-layer에서 input을 더하는 것과 normalization을 하기전에 output에 dropout을 설정했습니다. 또한 encoder와 decoder stacks에 embedding의 합계와 positional encoding에도 dropout을 설정했습니다.

dropout rate $P_{drop}=0.1$ 을 사용했습니다.

#### Label Smoothing

학습하는 동안 label smoothing value $\epsilon_{ls}=0.1$을 적용했습니다.

### Result

---

#### Machine Translation

![](https://media.vlpt.us/images/emeraldgoose/post/5faf7e14-34d5-4777-aad0-f3674c68c6c7/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-08-17%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%209.57.17.png)

영어→독일어 번역에서는 기존 모델들보다 높은 점수가 나왔고 영어→프랑스어 번역에서는 single 모델보다 좋고 ensemble 모델들과 비슷한 성능을 내주는 것을 볼 수 있습니다.

여기서 중요한 점은 Training Cost인데 기존 모델들보다 훨씬 적은 Cost가 들어가는 것을 볼 수 있습니다.

#### Model Variations

![](https://media.vlpt.us/images/emeraldgoose/post/979163c8-c04b-4e4a-83b5-56236105b7e0/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202021-08-17%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%209.44.26.png)

- (A)를 보면 single-head attention은 head=16일때보다 0.9 BLEU 낮고 head=32로 늘렸을 때도 head=16일때보다 BLEU가 낮습니다.
- (B)를 보면 *dk*를 낮추는 것이 model quality를 낮추게 합니다.
- (C), (D)를 보면 더 큰 모델일수록 좋고, dropout이 overfitting을 피하는데 도움이 되는 것을 볼 수 있습니다.
- (E)를 보면 sinusoidal position대신 learned positional embeddings를 넣었을 때의 결과가 base model과 동일한 결과인 것을 볼 수 있습니다.

### Conclusion

---

재귀 구조 없이 Multi-headed Self-attention 으로 인코더-디코더를 대체한 Transformer 모델을 제시하였습니다.

재귀구조가 없으므로 recurrent 또는 convolutional 레이어 기반 모델보다 빠르게 학습을 할 수 있습니다.

해당 모델은 WMT 2014 영어→독어, 영어→불어 번역 분야에서 기존 모든 앙상블 모델들을 능가하는 SOTA를 달성했습니다.

---
- Link: **[Attention is all you need](https://arxiv.org/abs/1706.03762)**