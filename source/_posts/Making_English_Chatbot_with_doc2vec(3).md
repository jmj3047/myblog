---
title: Making English Chatbot with doc2vec(3)
date: 2022-05-06
categories:
  - Python
  - Python
tags: 
  - Doc2vec
  - Chatbot
  - NLP
---

## 많은 데이터로 실험해보기

### 데이터 살펴보기

더 많은 학습 데이터로 모델을 학습한다. 

데이터 원본 링크: [https://www.kaggle.com/jiriroz/qa-jokes](https://www.kaggle.com/jiriroz/qa-jokes) 
총 3만 8천개의 문장

### 데이터 불러오기, 전처리

```python
import os
import warnings
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd

faqs = pd.read_csv(os.path.join('data','jokes.csv'))
faqs
```

한국어와 다르게 영어는 띄어쓰기로 단어가 잘 구분되기 때문에 형태소 분석은 생략한다. 형태소 분석을 하지 않아도 띄어쓰기로 split하면 단어 단위로 잘 짤리기 때문이다. 하지만 영어 단어를 원형으로 만들어 주는 lemmatization이나 the나 a같은 관사를 제거하는 stopword 제거는 해준다. 

```python
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')

# 토큰화
tokened_questions = [word_tokenize(question.lower()) for question in faqs['Question']]
tokened_questions
```

```
[['did',
  'you',
  'hear',
  'about',
  'the',
  'native',
  'american',
  'man',
  'that',
  'drank',
  '200',
  'cups',
  'of',
  'tea',
  '?'],
```

대문자를 모두 소문자로 바꿔주고 토큰화를 한 다음 띄어쓰기로 출력해주었다. 

```python
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
# lemmatization
lemmed_questions = [[lemmatizer.lemmatize(word) for word in doc] for doc in tokened_questions]
lemmed_questions
nltk.download('stopwords')
# stopword 제거 불용어 제거하기
stop_words = stopwords.words('english')
questions = [[w for w in doc if not w in stop_words] for doc in lemmed_questions]
questions
```

```
**[['hear', 'native', 'american', 'man', 'drank', '200', 'cup', 'tea', '?'],
 ["'s", 'best', 'anti', 'diarrheal', 'prescription', '?'],
 ['call', 'person', 'outside', 'door', 'ha', 'arm', 'leg', '?'],
 ['star', 'trek', 'character', 'member', 'magic', 'circle', '?'],
 ["'s", 'difference', 'bullet', 'human', '?'],
 ['wa', 'ethiopian', 'baby', 'cry', '?'],
 ["'s",
  'difference',
  'corn',
  'husker',
  'epilepsy',
  'hooker',
  'dysentery',
  '?'],
 ['2016', "'s", 'biggest', 'sellout', '?'],**
```

lemmatization하고 불용어를 제거하고 난 다음의 결과물. 이제 모든 전처리가 끝났으니 TaggedDocument로 변형시키고 나서 doc2vec에 넣어준다. 

```python
# 리스트에서 각 문장부분 토큰화

index_questions = []
for i in range(len(faqs)):
    index_questions.append([questions[i], i ])

# Doc2Vec에서 사용하는 태그문서형으로 변경
tagged_questions = [TaggedDocument(d, [int(c)]) for d, c in index_questions]
```

### doc2vec 모델화

doc2vec을 훈련하기 전에 모델에 변형을 주었다. for문도 빼고 파라미터도 변경해 주었다. 

```python
# make model
import multiprocessing
cores = multiprocessing.cpu_count()
d2v_faqs = doc2vec.Doc2Vec(vector_size=200,
                            # alpha=0.025,
                            # min_alpha=0.025,
                            hs=1,
                            negative=0,
                            dm=0,
                            dbow_words = 1,
                            min_count = 5,
                            workers = cores,
                            seed=0,
                            epochs=20)
d2v_faqs.build_vocab(tagged_questions)
d2v_faqs.train(tagged_questions,
               total_examples = d2v_faqs.corpus_count,
               epochs = d2v_faqs.epochs
                                )
    # # train document vectors
    # for epoch in range(50):
    #     d2v_faqs.train(tagged_faqs,
    #                                total_examples = d2v_faqs.corpus_count,
    #                                epochs = d2v_faqs.epochs
    #                                )
    #     d2v_faqs.alpha -= 0.0025 # decrease the learning rate
    #     d2v_faqs.min_alpha = d2v_faqs.alpha # fix the learning rate, no decay
```

```python
# 테스트하는 문장도 같은 전처리를 해준다.
test_string = "What's the best anti diarrheal prescription?"
tokened_test_string = word_tokenize(test_string)
lemmed_test_string = [lemmatizer.lemmatize(word) for word in tokened_test_string]
test_string = [w for w in lemmed_test_string if not w in stop_words]

test_string
```

```python
# 성능 측정
raten = 5
found = 0
for i in range(len(faqs)):
    tstr = faqs['Question'][i]
    tokened_test_string = word_tokenize(tstr)
    lemmed_test_string = [lemmatizer.lemmatize(word) for word in tokened_test_string]
    ttok = [w for w in lemmed_test_string if not w in stop_words]
    tvec = d2v_faqs.infer_vector(ttok)
    re = d2v_faqs.docvecs.most_similar([tvec], topn = raten)
    for j in range(raten):
        if i == re[j][0]: 
            found = found + 1
            break

print("정확도 = {} % ({}/{} )  ".format(found/len(faqs),found, len(faqs)))
```

```
정확도 = 0.8626303274190598 % (33012/38269 )
```

---

- Local path :C:\Users\jmj30\Dropbox\카메라 업로드\Documentation\2022\2022 상반기\휴먼교육센터\mj_chatbot_prac\faq_chatbot
- Reference: [https://cholol.tistory.com/466](https://cholol.tistory.com/473?category=803480)