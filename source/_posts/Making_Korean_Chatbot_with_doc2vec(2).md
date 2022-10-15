---
title: Making Korean Chatbot with doc2vec(2)
date: 2022-05-06
categories:
  - Python
  - Python
tags: 
  - Doc2vec
  - Chatbot
  - NLP
---

## 모델 다듬기

### FAQ데이터 늘리기

더 많은 학습 데이터로 모델을 학습한다. 

데이터 원본 링크: [https://www.data.go.kr/dataset/3068685/fileData.do](https://www.data.go.kr/dataset/3068685/fileData.do)

```python
import os
import warnings
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd

faqs = pd.read_csv(os.path.join('data','kor_elec_faq2.csv'), encoding='CP949')
faqs
faqs[['순번', '제목', '내용']]
```

pandas를 사용해 csv파일을 바로 읽어준다. utf-8 인코딩 문제로 에러가 나면 cp949로 넣어준다. 전체 필드에서 필요한 index와 질문(여기서는 제목), 답변(여기서는 내용)만 뽑아낸다. 총 351개의 질문과 답변 데이터가 생겼으니 전 게시물과 동일한 방법으로 모델학습을 시킨다. 

전 데이터는 pandas데이터가 아니었기 때문에 수정해 준다.

```python
# 리스트에서 각 문장부분 토큰화
token_faqs = []
for i in range(len(faqs)):
	token_faqs.append([tokenize_kkma_noun(faqs['제목'][i]), faqs['순번'][i]])

# Doc2Vec에서 사용하는 태그문서형으로 변경
tagged_faqs = [TaggedDocument(d, [c]) for d, c in token_faqs]
```

이렇게 하고 모델을 돌려도 성능이 좋지 않음. doc2vec모델의 경우 최소 만단위의 문장이 있어야 제대로 나온다. 

### 튜닝 시도해보기

이전까지 데이터 내에 있는 순번을 인덱스로 사용했는데 정상적으로 작동하지 않아 다시 만들어준다. 

```python
# 리스트에서 각 문장부분 토큰화
token_faqs = []
for i in range(len(faqs)):
    token_faqs.append([tokenize_kkma_noun(faqs['제목'][i]), i ])
    # token_faqs.append([tokenize_kkma_noun(faqs['제목'][i]), faqs['순번'][i]])

# Doc2Vec에서 사용하는 태그문서형으로 변경
tagged_faqs = [TaggedDocument(d, [int(c)]) for d, c in token_faqs]
# tagged_faqs = [TaggedDocument(d, [c]) for d, c in token_faqs]
```

문서 원본을 수정할 필요는 없고 TaggedDocument만들 때만 잘 넣어준면 된다. 기존에는 faqs[’순번’][i]를 태그 값으로 넣어주었는데 그냥  i를 넣는다. 이렇게 하면 좋은 점이 원본의 index와 태그값이 같아지기 때문에 나중에 원문질문을 복원할 때 faqs[’제목’][tag]로 출력이 가능하다. 

그리고 이제 가장 먼저 할 거는 전처리를 조금 수정하는 것이다. 형태소 분석을 할 때 필요 없는 데이터를 제외시키는 방법이다. 보통 문장에서는 명사와 동사가 중요하기 때문에 명사 동사 빼고는 다 날려본다. 

```python
#튜닝:형태소 필터링
kkma = Kkma()
filter_kkma = ['NNG',  #보통명사
             'NNP',  #고유명사
             'OL' ,  #외국어
             'VA','VV','VXV'
            ]

def tokenize_kkma(doc):
    jpype.attachThreadToJVM()
    token_doc = ['/'.join(word) for word in kkma.pos(doc)]
    return token_doc

def tokenize_kkma_noun(doc):
    jpype.attachThreadToJVM()
    token_doc = ['/'.join(word) for word in kkma.pos(doc) if word[1] in filter_kkma]
    return token_doc
```

이런식으로 filter_kkma리스트를 하나 만들어 형태소를 분석했을 때 나오는 형태소가 filter_kkma에 포함되어 있을 경우만 학습 대상에 추가한다. tokenize_kkma를 쓰면 전체 형태소 분석, tokenize_kkmk_noun을 쓰면 동사 명사만 추출한다. 

가장 결과가 좋게 나온 조합은 명사만 추출, for 문 50번에 epochs=100으로 한 결과값. 

```python
#튜닝:명사만 추출
kkma = Kkma()
filter_kkma = ['NNG',  #보통명사
             'NNP',  #고유명사
             'OL' ,  #외국어
            ]

def tokenize_kkma(doc):
    jpype.attachThreadToJVM()
    token_doc = ['/'.join(word) for word in kkma.pos(doc)]
    return token_doc

def tokenize_kkma_noun(doc):
    jpype.attachThreadToJVM()
    token_doc = ['/'.join(word) for word in kkma.pos(doc) if word[1] in filter_kkma]
    return token_doc
```

```python
# make model
    import multiprocessing
    import tensorflow as tf
    with tf.device('/device:GPU:0'):
        cores = multiprocessing.cpu_count()
        d2v_faqs = doc2vec.Doc2Vec(vector_size=20, #100
                                    alpha=0.025,
                                    min_alpha=0.025,
                                    hs=1,
                                    negative=0,
                                    dm=0,
                                    window=3,
                                    dbow_words = 1,
                                    min_count = 1,
                                    workers = cores,
                                    seed=0,
                                    epochs=100)
        d2v_faqs.build_vocab(tagged_faqs)

        # train document vectors
        for epoch in range(50):
            d2v_faqs.train(tagged_faqs,
                                    total_examples = d2v_faqs.corpus_count,
                                    epochs = d2v_faqs.epochs
                                    )
            d2v_faqs.alpha -= 0.0025 # decrease the learning rate
            d2v_faqs.min_alpha = d2v_faqs.alpha # fix the learning rate, no decay
```

```python
test_string = "변압기공동이용(모자거래)이란 무엇이며, 요금계산은 어떻게 합니까"
tokened_test_string = tokenize_kkma_noun(test_string)
tokened_test_string
```

```python
# 성능 측정
# raten = 5 #정확도 = 0.5128205128205128 % (180/351 )
raten = 1 #정확도 = 0.24216524216524216 % (85/351 ) 
found = 0
for i in range(len(faqs)):
    tstr = faqs['제목'][i]
    ttok = tokenize_kkma_noun(tstr)
    tvec = d2v_faqs.infer_vector(ttok)
    re = d2v_faqs.docvecs.most_similar([tvec], topn = raten)
    for j in range(raten):
        if i == re[j][0]: found = found + 1

print("정확도 = {} % ({}/{} )  ".format(found/len(faqs),found, len(faqs))
```

### 모델 저장, 불러오기

```python
# 모델 저장
d2v_faqs.save(os.path.join('data','/d2v_faqs_size100_min1_batch50_epoch100_nounonly_dm0.model'))
# 모델 load
d2v_faqs_1 = doc2vec.Doc2Vec.load(os.path.join('data','/d2v_faqs_size100_min1_batch50_epoch100_nounonly_dm0.model'))

#test
test_string = "건물을 새로 지을 때 임시전력은 어떻게 신청하나요"
tokened_test_string = tokenize_kkma_noun(test_string)
tokened_test_string

topn = 5
# 모델 추측
test_vector1 = d2v_faqs_1.infer_vector(tokened_test_string)
result1 = d2v_faqs_1.docvecs.most_similar([test_vector1], topn=topn)

for i in range(topn):
    print("모델 1 {}위. {}, {} {}".format(i+1, result1[i][1], result1[i][0],faqs['제목'][result1[i][0]] ))
```

---

- Local path :C:\Users\jmj30\Dropbox\카메라 업로드\Documentation\2022\2022 상반기\휴먼교육센터\mj_chatbot_prac\faq_chatbot
- Reference: [https://cholol.tistory.com/466](https://cholol.tistory.com/469?category=803480)