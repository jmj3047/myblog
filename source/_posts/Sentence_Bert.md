---
title: Sentence Bert
date: 2023-09-11
categories:
  - Paper
  - NLP
tags: 
  - Sentence Bert
  - Bi Encoder
  - Cross Encoder
---

# 들어가며

이 글은 Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks를
소개하고 논문의 핵심 구조인 Sbert를 코드로 구현하는 방법에 대해
설명합니다.



# Sentence Bert가 필요한 이유

Sentence Bert는 Bert을 문장 임베딩(Sentence Embedding)을 생성하는 모델로 활용할 수 있도록 Fine-tuning하는 방법(또는 모델명) 을 의미합니다. 이때 Sentence embedding라 함은 문장 정보를 벡터 공간의 위치로 표현한 값을 말하며, 문장을 벡터 공간에 배치함으로서 문장 간 비교, 클러스터링, 시각화 등 다양한 분석 기법을 이용할 수 있는 장점이 있습니다.

사실 Sbert 이전에도 Bert 모델을 활용해 Sentence Embedding을 생성하는 방법이 존재했지만, 이러한 방법은 과거 모델(Glove,Infer-Sent)의 성능에 미치지 못했습니다. 이러한 이유 때문에 Transformer 기반 모델을 활용해 문장 간 유사도를 비교하는 Task에서는 sentence embedding 방법을 사용하지 않고 주로 두 개의 문장을 모델에 넣어 Cross-Attention을 활용해 비교하는 방식을 활용했습니다. 여기서 일대일로 방식이라 하면 두 개의 문장을 하나로 묶은 Input Data를 Bert 모델에 넣은 뒤 모델 내부에서 두 문장 간 관계를 파악하고 모델의 Output 중  [CLS] 토큰을 활용해 두 문장의 유사도를 파악하는 방법을 의미합니다.

Sentence Bert 논문에서는 문장과 문장을 비교하는 Task인 Named Entity Recognition(NER), Semantic Textual Similarity(STS)를 수행하는데 Senetnece Embedding을 활용하고 있지만, Senetence Embedding은 이러한 Task 뿐만아니라 문장과 단어 간 연관성 비교를 통한 키워드 추출, 특정 문서의 카테고리 선정 등 다양한 Task에서 응용이 가능하므로 이를 기반으로한 논문이나 라이브러리가 존재합니다. 다음의 링크들은 Setnece Bert를 활용한 라이브러리 및 논문들입니다.

-   [Sbert 공식 페이지 응용 예시](https://www.sbert.net/examples/applications/)
-   [Bertopic : 토픽 추출 라이브러리](https://github.com/MaartenGr/BERTopic)
-   [keyBert : 문서 키워드 추출 라이브러리](https://github.com/MaartenGr/BERTopic)

# Cross-Encoder와 Bi-Encoder

해당 논문에서는 Bert 모델 내부의 Cross-Ateention을 활용해 문장 간 관계를 비교했던 기존 방식을 Cross-Encoder라는 용어로 사용하고 있으며, 논문에서 새롭게 소개하는 구조를 Bi-Encoder라는 용어로 사용하고 있습니다. 

Cross-Encoder와 Bi-Encoder의 구조 차이는 아래 그림과 같습니다

<p align = "center"><img src="https://yangoos57.github.io/static/812fe66e9ad7a89e832b77f4cf7a8c27/3c492/img0.png" width="500" height="500"/></p>


위 그림에 대해 설명하면, Bi-Encoder는 두 문장을 비교하기 위해 개별 문장의 Embedding 생성하는 단계 -> 모델 Output을 Pooling하여 Sentence Embedding 생성하는 단계 -> CosineSimilarity를 통해 문장과 문장 간 관계 비교를 비교하는 단계 이렇게 3번의 단계를 거칩니다. 기존 방식인 Cross-Encoder는 두 개의 문장을 Language Model에 넣어 내부에서 문장 간 문장의 관계를 비교합니다.

절차적 측면에서 보면 Cross-Encoder가 더 간단한 방법인 것 같아 보입니다. 하지만 100개 문장을 비교한다고 가정할 때 Cross-Encoder는 100개의 문장을 1:1로 비교해야 하므로 100C2회를 수행해야 하는 반면 Bi-Encoder는 일단 문장을 embedding하면 비교하는 과정 자체는 단순하므로 문장을 embedding화 하기 위해 100회만 수행하면 됩니다. 구조 자체는 Cross-Encoder가 단순해보이지만 실제로는 Bi-Encoder 방식이 효율성 면에서 훨씬 더 효과적임을 알 수 있습니다.

Cross-Encoder와 Bi-Encoder에 대해 개별적으로 알아보기 전 Cross-Encoder와 Bi-Encoder의 특징에 대해 간단히 알아보도록 하겠습니다. 먼저 Cross-Encoder는 문장 간 관계를 파악하는 성능이 우수한 장점이 있지만 앞서 설명했듯 비교해야하는 문장수가 많아질수록 연산이 급증한다는 치명적인 단점이 있습니다. 반면 Bi-Encoder는 Embedding 과정에서 정보손실이 발생하므로 성능에 있어서 Cross-Encoder에 미치지 못하지만, 실시간 문제 해결에 활용될 수 있을만한 빠른 연산 속도를 보장합니다.

이러한 특징에서 보듯 이 둘은 상호 보완적인 관계에 있습니다. Bi-Encoder는 Cross-Encoder의 느린 연산속도를 보완할 수 있고, Cross-Encoder는 Bi-Encoder의 부족한 문장 비교 성능을 보완할 수 있습니다. 실제로도 이러한
개별 특징을 활용해 검색 기능을 구현할 수도 있습니다. 아래 그림은 Bi-Encoder와 Cross-Encoder의 개별 장점을 살려 효과적인 검색을 수행할 수 있는 구조를 보여줍니다. 이 구조는 Bi-Encoder의 빠른 연산속도를 활용해 query와 유사한 문장을 추려낸 다음, Cross-Encoder를 활용해 추려낸 문장과 Query 간 연관성을 다시 계산해 순위를 메기는 방식으로 동작합니다.

> 제가 수행했던 미니프로젝트인 [Sentence Bert를 활용해 연관성 높은 도서 추천하기](https://github.com/yangoos57/Sentence_bert_from_scratch)를 읽어보면 이러한 구조를 어떻게 코드로 구현할 수 있는지 확인하실 수 있습니다.
 <p align = "center"><img src="https://yangoos57.github.io/static/31659fa96212160ec5c5ec892af7e5d1/3c492/img1.png" width="600" height="300"/></p>



## Cross-Encoder

먼저 기존 방식인 Cross-Encoder에 대해서 설명한 뒤, 논문에서 소개하는 Bi-Encoder에 대해서 설명하겠습니다.

### ❖ Cross-Encoder 구조 이해하기

Cross-Encoder 구조는 Language Model에 classification layer를 쌓은 구조입니다. 아래 그림에서 파란색 네모 박스를 Language Model이라 하며 그 위의 노란색 테두리를 Classification Layer라 합니다. Language Model은 Bert 뿐만아니라 Electra, Roberta 등 Encoder 기반 모델이면 모두 활용할 수
있습니다.


<p align = "center"><img src="https://yangoos57.github.io/static/768bc61ae0bef22c4c25914cb3393e76/3c492/img7.png" width="600" height="500"/></p>


Cross-Encoder 내부의 데이터 흐름을 보면 Language Model의 Output을 산출한 뒤 CLS Pooling을 거쳐 다시 Classification Layer의 Input Data로 활용되고 있음을 알 수 있습니다. 이때 CLS pooling이라 하면 문장의 여러 token embedding 중 \[CLS\] token embedding을 문장 embedding으로 사용하는 방식을 의미합니다. CLS Pooling을 다르게 표현하자면 문장과 문장의 관계를 나타내고 있는 정보들은 \[CLS\] token에 모두 녹아들어있으니 \[CLS\] token외 나머지는 문장 embedding으로 사용하지 않는다라는 의미로 이해하시면 되겠습니다.

Cross-Encoder의 구조는 Language Model과 Classification Head로 구성된 매우 간단한 구조이며 아래의 코드는 이러한 구조를 보여줍니다. 아래 코드에서 주목해야할 점은 arguments로 활용되는 num_labels의 존재입니다.

Cross-Encoder Class에서 num_labels가 활용되는 목적은 모델의 Loss Function을 적용하는데 있습니다. 코드 마지막 부분에서 num_labels가 활용되는 코드를 볼 수 있는데, num_labels이 1인 경우 MSE를 Loss function을 활용하고 그외인 경우 Cross Entropy를 Loss function으로 활용하고 있는 것을 확인할 수 있습니다. num_labels 값에 따라 Loss function이 달라지는 이유는 input Data로 사용되는 타입이 Numerical Data인지 Categorical Data인지 여부에 따라 사용해야하는 Loss function이 다르기 때문입니다.

 
``` python
from torch.nn import CrossEntropyLoss, MSELoss

class CrossEncoder(nn.Module):
    def __init__(self, model, num_labels) -> None:
        super().__init__()
        self.model = model
        self.model.config.num_labels = num_labels
        self.classifier = classificationHead(self.model.config)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        model = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Last-hidden-states 추출
        sequence_output = model[0]
        # classificationHead에 Last-hidden-state 대입
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.model.config.num_labels == 1:
                # Regression Model은 MSE Loss 활용
                loss_fct = MSELoss()
            else:
                # classification Model은 Cross entropy 활용
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 3), labels.view(-1))
            return {"loss": loss, "logit": logits}
        else:
            return {"logit": logits}

```



**CLS 토큰이란?**

-   BERT는 학습을 위해 기존 transformer의 input 구조를 사용하면서도 추가로 변형하여 사용합니다. Tokenization은 WorldPiece 방법을 사용하고 있습니다.
<p align = "center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpZneZ%2FbtqGg6mCUaU%2FEcXXk5nCUAdTRMK2vXORO0%2Fimg.png" width="700" height="300"/></p>

-   위 그림처럼 세 가지 임베딩(Token, Segment, Position)을 사용해서 문장을 표현합니다.

-   먼저 Token Embedding에서는 두 가지 특수 토큰(CLS, SEP)을 사용하여 문장을 구별하게 되는데요. Special Classification token(CLS)은 모든 문장의 가장 첫 번째(문장의 시작) 토큰으로 삽입됩니다. 이 토큰은 Classification task에서는 사용되지만, 그렇지 않을 경우엔 무시됩니다.
-   또, Special Separator token(SEP)을 사용하여 첫 번째 문장과 두 번째 문장을 구별합니다. 여기에 segment Embedding을 더해서 앞뒤 문장을 더욱 쉽게 구별할 수 있도록 도와줍니다. 이 토큰은 각 문장의 끝에 삽입됩니다.
-   Position Embedding은 transformer 구조에서도 사용된 방법으로 그림과 같이 각 토큰의 위치를 알려주는 임베딩입니다. 최종적으로 세 가지 임베딩을 더한 임베딩을 input으로 사용하게 됩니다.



### ❖ Classification layer 구조 이해하기

Cross-Encoder의 전체 구조와 코드를 소개했으니 이제 Classification Layer의 내부 구조에 대해서  설명하겠습니다. 아래 그림은 Classification의 내부 구조와 개별 layer를 통해 나오는 Output Tensor의 크기를 보여줍니다. layer의 최종 output의 크기는 \[1,N\]이며, 여기서 N은 num_labels과 동일한 값이자 산출해야하는 카테고리 개수를 의미합니다. 만약 Regression 유형의 output이 필요한 경우 N = 1로 설정해야 하며, k개의 카테고리를 구분해야하는 Output이 필요한 경우 N = k로 설정해야 합니다.

<p align = "center"><img src="https://yangoos57.github.io/static/0ed34c4ed6b114c93110fb7822142201/3c492/img8.png" width="600" height="500"/></p>
 
``` python
from torch import Tensor, nn
class classificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.gelu = nn.functional.gelu
        self.dropout = nn.Dropout(classifier_dropout)
        # [batch, embed_size] => [batch, num_labels]
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, features, **kwargs):
        x = features[:, 0, :] # [CLS] 토큰 추출
        x = self.dropout(x)
        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        # label 개수만큼 차원 축소 [batch, embed_size] => [batch, num_labels]
        x = self.out_proj(x)
        return x
```

### ❖ Cross-Encoder 학습

Cross-Encoder를 실제 학습하는 과정은 [Cross-Encoder 학습 튜토리얼(Jupyter Notebook)](https://github.com/yangoos57/Sentence_bert_from_scratch)을 참고하시기 바랍니다. 해당 튜토리얼은 🤗 Transformers를 활용해 작성되었으므로 Huggingface에 익숙하지 않으신 분들은 추가적으로 [링크](https://yangoos57.github.io/blog/DeepLearning/paper/Electra/electra/)를 참고하시기 바랍니다.



## Bi-Encoder

이제 Sentence Bert 논문의 핵심 구조인 Bi-Encoder에 대해 설명하도록 하겠습니다. Bi-Encoder는 문장 간 비교가 필요한 Task에 대해 훨신 높은 퍼포먼스를 보여주는 장점이 있다고 설명한 바 있습니다. 이러한 속도를 보장할 수 있는 이유는 Sentence Embedding을 활용해 문장을 벡터 공간에 위치시켜 CosineSimilarity를 활용해 계산하기 때문이었습니다.

아래 표 주황색으로 쳐져있는 실선 중 Avg. Bert Embeddings는 이전에 시도했던 Sentence Embedding 방식의 성능을 보여주며, 이러한 성능은 과거 모델인 Glove, InferSent 성능에도 미치지 못하고 있음을 확인할 수 있습니다.

반면 NLI 데이터셋으로 학습한 SentenceBert 모델의 성능은 Glove, InferSent 성능을 압도할 뿐만아니라 기존 방식의 성능 대비 약 1.8배 이상의 성능을 보여줌을 확인할 수 있습니다.

<p align = "center"><img src="https://yangoos57.github.io/static/402c52b9e63859d06e0456b99dc4b571/13ae7/img2.png" width="500" height="600"/></p>



### ❖ Sentence Bert 구조 

<p align = "center"><img src="https://yangoos57.github.io/static/39f1a72e77fc2a06fb0f0ccd8489a161/3d64b/img4.png" width="200" height="300"/></p>



 
``` python
from transformers import ElectraModel, ElectraTokenizer
import torch.nn as nn
import torch
model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
class modelWithPooling(nn.Module):
    def __init__(self, model, pooling_type="mean") -> None:
        super().__init__()
        self.model = model  # base model ex)BertModel, ElectraModel ...
        self.pooling_type = pooling_type  # pooling type 설정(기본 mean)
    def forward(self, **kwargs):
        features = self.model(**kwargs)
        # [batch_size, src_token, embed_size]
        attention_mask = kwargs["attention_mask"]
        last_hidden_state = features["last_hidden_state"]
        if self.pooling_type == "cls":
            """
            [cls] 부분만 추출
            """
            cls_token = last_hidden_state[:, 0]  # [batch_size, embed_size]
            result = cls_token
        if self.pooling_type == "max":
            """
            문장 내 토큰 중 가장 값이 큰 token만 추출
            """
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            # Set padding tokens to large negative value
            last_hidden_state[input_mask_expanded == 0] = -1e9
            max_over_time = torch.max(last_hidden_state, 1)[0]
            result = max_over_time
        if self.pooling_type == "mean":
            """
            문장 내 토큰을 합한 뒤 평균
            """
            # padding 부분 찾기 = [batch_size, src_token, embed_size]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            # padding인 경우 0 아닌 경우 1곱한 뒤 총합 = [batch_size, embed_size]
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            # 평균 내기위한 token 개수
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            result = sum_embeddings / sum_mask
        #  input.shape : [batch_size, src_token, embed_size] => output.shape : [batch_size, embed_size]
        return {"sentence_embedding": result}


```

### ❖ Sbert 학습 구조 : Categorical Data를 학습하는 경우 

Sbert는 학습에 활용될 데이터셋에 따라 학습 구조가 달라집니다. 따라서 자신이 활용할 데이터셋이 numerical 데이터셋인지, categorical 데이터셋인지 구분을 해야합니다. 먼저 categorical 데이터 유형에 대해서 설명하겠습니다. 예제에서 활용하는 데이터셋은 자연어추론(NLI) 데이터셋이며 구조는 아래와 같습니다.

> {\'sen1\': \'그리고 그가 말했다, \"엄마, 저 왔어요.\"\',
> \'sen2\': \'그는 학교 버스가 그를 내려주자마자 엄마에게 전화를
> 걸었다.\',
> \'gold_label\': \'neutral\'}

categorical 데이터로 Sbert를 학습하는 구조는 아래와 같습니다. 1차로 SBert 모델을 통해 산출한 embedding vector를 각각 U,V라 할 때 U,V,\|U-V\|를 하나의 Tensor로 concat을 수행합니다. 그 다음 softmax Classifier를 통해 entailment, neutral, contradition을 판단하고 Loss를 구해 학습을 진행합니다.


<p align = "center"><img src="https://yangoos57.github.io/static/4ce257bd3b28eebd860c628554145582/e17e5/img5.png" width="300" height="400"/></p>




#### ❖ categorical Data 학습 구조 


 
``` python
from torch import nn
class modelForClassificationTraining(nn.Module):
    def __init__(self, model, *inputs, **kwargs):
        super().__init__()
        # 학습할 모델 불러오기
        self.model = modelWithPooling(model)
        # 모델 embed_size
        sentence_embedding_dimension = self.model.model.config.hidden_size
        # concat 해야하는 vector 개수(U,V, |U-V|)
        num_vectors_concatenated = 3
        # embed_size * 3 => 3 차원으로 축소시키는 classifier
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, 3)
    def forward(self, features, answer):
        """
        샴 네트워크는 하나의 모델로 두 개의 output을 산출하는 구조임.
        하나의 모델을 사용하지만 각각 출력하므로 Input 데이터 상호 간 영향을 줄 수 없게 됨.
        """
        # 개별 데이터 생성
        embeddings = [self.model(**input_data)["sentence_embedding"] for input_data in features]
        rep_a, rep_b = embeddings
        # U,V, |U-V| vector 병합
        vectors_concat = []
        vectors_concat.append(rep_a)
        vectors_concat.append(rep_b)
        vectors_concat.append(torch.abs(rep_a - rep_b))
        features = torch.cat(vectors_concat, 1)
        # 병합한 vector 차원 축소
        outputs = self.classifier(features)
        # Loss 계산
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, answer.view(-1))
        return {"loss": loss}
```



### ❖ Sbert 구조 : Numerical Data를 학습하는 경우 

Numerical Data는 문장과 문장 간 비교를 수치료 표현한 데이터를 말합니다.

> { \'sen1\': \'비행기가 이륙하고 있다.\',
    \'sen2\': \'비행기가 이륙하고 있다.\',
    \'score\': \'5.000\'}

Numerical 학습 구조는 코사인 유사도를 활용해 Embedding Vector를 비교합니다.
<p align = "center"><img src="https://yangoos57.github.io/static/9c9a98db74d4821476ca98bf435744f4/e17e5/img6.png" width="300" height="400"/></p>



#### ❖ Numerical Data 학습 구조


 
``` python
from torch import nn
class modelForRegressionTraining(nn.Module):
    def __init__(self, model, *inputs, **kwargs):
        super().__init__()
        # 학습을 수행할 모델 불러오기
        self.model = modelWithPooling(model)
    def forward(self, features, answer):
        # Sentence 1, Sentence 2에 대한 Embedding
        embeddings = [self.model(**input_data)["sentence_embedding"] for input_data in features]
        # Sentence 1, Sentence 2에 대한 Cosine Similarity 계산
        cos_score_transformation = nn.Identity()
        outputs = cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        # label score Normalization
        answer = answer / 5  # 0 ~ 5 => 0 ~ 1
        loss_fct = nn.MSELoss()
        loss = loss_fct(outputs, answer.view(-1))
        return {"loss": loss}

```


### Bi-Encoder 활용

학습이 완료되면 학습에 활용된 구조는 버리고 Sentence Bert만 추출하여 활용합니다. 이와 관련한 예제는 [Sbert 깃허브 페이지](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications)에 코드로 자세히 설명하고 있으니 응용 방법에 대해 궁금한 경우 해당 링크를 참고 바랍니다.


---

- Reference
    - https://yangoos57.github.io/blog/DeepLearning/paper/Sbert/Sbert/
    - https://hwiyong.tistory.com/392