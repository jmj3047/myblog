---
title: Making Chatbot with doc2vec tutorial(1)
date: 2022-05-06
categories:
  - Python
tags: 
  - Doc2vec
  - Chatbot
  - NLP
---

## 모델 만들기

### 데이터 만들기

doc2vec을 이용해서 FAQ데이터들의 질문들을 벡터화하는 모델을 만들어 본다. word2vec이 단어를 벡터화 하는 것이라면 doc2vec은 단어가 아니라 문서를 기준으로 (여기서는 문장)벡터를 만드는 라이브러리이다. doc2vec을 사용하면 서로 다른 문서들이 같은 차원의 벡터값을 갖게 된다. 각 문서라 갖는 벡터값을 비교해 같으면 같을 수록 유사한 문서라는 것을 알 수 있다. 

따라서 doc2vec을 이용해 FAQ의 질문들을 벡터화 한다면 어떤 질문이 들어왔을 때 동일 모델로 질문을 벡터화 한다음, 저장돼 있는 질문들의 벡터와 비교해서 가장 유사한 질문을 찾을 수 있다. 가장 유사한 질문을 찾은 다음 그 질문의 답을 출력하면 FAQ챗봇을 만들 수 있다. 

**GPU사용 필수..!

```python
import os
import warnings
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument

faqs = [["1", "당해년도 납입액은 수정 가능 한가요?", "네, 당해년도 납입액은 12464 화면 등록전까지 수정 가능합니다."],
        ["2", "대리인통보 대상계좌 기준은 어떻게 되나요?", "모계좌 기준 가장 최근에 개설된 계좌의 관리점에서 조회 됩니다.  의원폐쇄된 자계좌는 조회대상 계좌에서 제외됩니다. 계좌주 계좌가 사절원 계좌가 아닌 경우만 조회됩니다"],
        ["3", "등록가능 단말기수는 어떻게 되나요?", "5대까지 등록 가능입니다."],
        ["4", "모바일계좌개설 가능한 시간은 어떻게 되나요?", "08:00 ~ 20:00(영업일만 가능"],
        ["5", "미국인일때 미국납세자등록번호 작성 방법은 어떻게 되나요?", "계좌주가 미국인일 때 계좌주의 미국납세자등록번호(사회보장번호(Social Security Number), 고용주식별번호(Employer Identification Number), 개인납세자번호(Individual Taxpayer Identification Number))를 기재합니다.."]
]
```

위와 같이 5개의 FAQ데이터를 임의로 만들었다. 이제 여기에 5개의 질문을 벡터화 할건데 사실 벡터화 할 때 데이터는 많을 수록 좋다. 적으면 서로 분간이 잘 안됨. 

### 형태소 분석

doc2vec으로 문장을 벡터화하기 전에 약간의 전처리 과정이 필요하다. 각 문장을 tokenize해야 한다. 토큰화 하는 과정이 영어랑 한국어랑 조금 다른데 한국어의 경우 형태소 분석(pos tagging)을 통해 형태소 단위로 나눈뒤 토큰으로 사용할 형태소를 결정하고 나눈다. 즉 각 문장을 형태소 단위의 배열로 만든다. 

한국어 형태소 분석기는 konlpy를 사용한다. 

```python
#형태소 분석
import jpype
from konlpy.tag import Kkma

kkma = Kkma()

def tokenize_kkma(doc):
    jpype.attachThreadToJVM() #자바를 사용하기 위한 소스 코드
    token_doc = ['/'.join(word) for word in kkma.pos(doc) ] #형태소 분석한 단어와 형태소 명을 '단어/형태소'형태로 출력하기 위한 코드
    return token_doc

tokenize_kkma(faqs[0][1])
```

Kkma를 import 하고 jpype도 import 한다. jpype는 파이썬에서 자바를 사용할 수 있게 해주는 패키지인데 기본적으로 kkma가 자바 베이스라서 꼭 필요하다. Kkma()로 형태소 분석기를 불러온다음 kkma.pos(doc)로 형태소 분석을 한다. 

```
출력 결과:

['당해/NNG',
 '년도/NNM',
 '납입/NNG',
 '액/XSN',
 '은/JX',
 '수정/NNG',
 '가능/NNG',
 '한/MDN',
 '가요/NNG',
 '?/SF']
```

형태소 분석을 하면 문장이 단어/형태소 형태의 배열로 출력된다. 1번 문장은 총 10개의 형태소로 나뉘었다. 형태소 분석기종류에 따라 결과가 조금씩 다를수 있다. 

### Doc2Vec 모델 만들기

Doc2Vec을 이용해 모델을 만들기 위해서는 토큰화 된 리스트와 태그 값이 필요하다. 여기서 태그는 문장 번호. [문장의 번호, 문장을 토큰화한 배열] 이렇게 두 개의 값을 가진 리스트를 사용해 doc2vec 모델을 만들 수 있다. 실제로 모델을 만드는데 사용하는 건 토큰 값이지만 비슷한 문장이 무엇인지 찾기 위한 인덱스로 태그 값을 사용하게 된다. 

```python
# 리스트에서 각 문장부분 토큰화
token_faqs = [(tokenize_kkma(row[1]), row[0]) for row in faqs]

# Doc2Vec에서 사용하는 태그문서형으로 변경
tagged_faqs = [TaggedDocument(d, [c]) for d, c in token_faqs]

tagged_faqs
```

```
[TaggedDocument(words=['당해/NNG', '년도/NNM', '납입/NNG', '액/XSN', '은/JX', '수정/NNG', '가능/NNG', '한/MDN', '가요/NNG', '?/SF'], tags=['1']),
 TaggedDocument(words=['대리인/NNG', '통보/NNG', '대상/NNG', '계좌/NNG', '기준/NNG', '은/JX', '어떻/VA', '게/ECD', '되/VV', '나요/EFQ', '?/SF'], tags=['2']),
 TaggedDocument(words=['등록/NNG', '가능/NNG', '단말/NNG', '기수/NNG', '는/JX', '어떻/VA', '게/ECD', '되/VV', '나요/EFQ', '?/SF'], tags=['3']),
 TaggedDocument(words=['모바일/NNG', '계좌/NNG', '개설/NNG', '가능/NNG', '하/XSV', 'ㄴ/ETD', '시간/NNG', '은/JX', '어떻/VA', '게/ECD', '되/VV', '나요/EFQ', '?/SF'], tags=['4']),
 TaggedDocument(words=['미국인/NNG', '일/NNG', '때/NNG', '미국/NNP', '납세자/NNG', '등록/NNG', '번호/NNG', '작성/NNG', '방법/NNG', '은/JX', '어떻/VA', '게/ECD', '되/VV', '나요/EFQ', '?/SF'], tags=['5'])]
```

TaggedDocument function을 사용하면 doc2vec에서 사용할 수 있는 태그 된 문서 형식으로 변경한다. 출력해 보면 words배열과 tags값을 갖는 Dic형태의 자료형이 되었음을 확인할 수 있다. 

```python
# make model
    import multiprocessing
    cores = multiprocessing.cpu_count()
    d2v_faqs = doc2vec.Doc2Vec(vector_size=50, 
                            alpha=0.025,
                            min_alpha=0.025,
                            hs=1,
                            negative=0,
                            dm=0,
                            dbow_words = 1,
                            min_count = 1,
                            workers = cores,
                            seed=0)
    d2v_faqs.build_vocab(tagged_faqs)

    # train document vectors
    for epoch in range(10):
        d2v_faqs.train(tagged_faqs,
                                   total_examples = d2v_faqs.corpus_count,
                                   epochs = d2v_faqs.epochs
                                   )
        d2v_faqs.alpha -= 0.0025 # decrease the learning rate
        d2v_faqs.min_alpha = d2v_faqs.alpha # fix the learning rate, no decay
```

모델을 만들고 학습 시킨다. 

doc2vec모델을 만들 때 파라미터는 여러가지가 들어가는데 여기서는 vector_size와 min_count정도를 수정했다. vector_size는 만들어지는 벡터 차원의 크기이고, min_count는 최소 몇 번 이상 나온 단어에 대해 학습할지 정하는 파라미터이다. 여기서는 일단 사이즈 50에 최소 횟수는 1회로 정했다. epoch는 10번으로 해서 train했다. 

### 유사 문장 찾기

이제 이 모델로 어떤 문장이 들어왔을 때 1~5번 중에 무엇과 비슷한지 알아보자. 먼저 어떤 문장이 들어오면 그 문장을 벡터화 하고 그 벡터가 어떤 문장과 비슷한지 태그 값을 찾아본다. 

```python
predict_vector = d2v_faqs.infer_vector(["당해년도 납입액은 수정 가능 한가요?"])
d2v_faqs.docvecs.most_similar([predict_vector], topn=2)
```

```
[('2', 0.21605531871318817), ('3', 0.10707802325487137)]
```

제대로 됐는지 확인하기 위해 1번 문장을 그대로 넣었지만 답은 2, 3번이 나왔다. 학습이 제대로 되지 않아서 이런 결과가 나왔다. 

테스트할 문장을 벡터화 할 때도 형태소 분석을 해줘야 한다. 왜냐하면 모델을 학습 할 때 문장들을 형태소로 분석해서 넣어줬기 때문이다. 

```python
test_string = "대리인통보 대상계좌 기준은 어떻게 되나요?"
tokened_test_string = tokenize_kkma(test_string)
tokened_test_string
```

```
['대리인/NNG',
 '통보/NNG',
 '대상/NNG',
 '계좌/NNG',
 '기준/NNG',
 '은/JX',
 '어떻/VA',
 '게/ECD',
 '되/VV',
 '나요/EFQ',
 '?/SF']
```

```python
test_vector = d2v_faqs.infer_vector(tokened_test_string)
d2v_faqs.docvecs.most_similar([test_vector], topn=2)
```

```
[('1', 0.1448383331298828), ('3', 0.0218462273478508)]
```

2번 문장으로 했을 때 결과입니다. 

doc2vec이라는 모델은 문서단위로 벡터화 하는 것이기 때문에 문서가 많아야 한다. 여기서는 문장이 많아야 한다. 문장이 많으면 많을 수록 문장 간의 거리를 계산해서 더 잘 구분해준다. 간단하게 생각하면 문장이 적으면 적중률이 높을 것 같지만 사실은 그 반대이다. 데이터가 많을수록 그 데이터간의 차이를 구분할 수 있기 때문에 더 잘 예측하게 된다. 

---

- Local path :C:\Users\jmj30\Dropbox\카메라 업로드\Documentation\2022\2022 상반기\휴먼교육센터\mj_chatbot_prac\faq_chatbot
- Reference: [https://cholol.tistory.com/466](https://cholol.tistory.com/466)