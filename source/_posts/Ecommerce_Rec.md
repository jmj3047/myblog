---
title: 이커머스 추천모델링
date: 2023-04-03
categories:
  - Data Platform/Base
  - GCP
tags: 
  - Recommendation System
  - Deep Nueral Networks
  - Vertex AI
---
# 개요

- kaggle instacart 데이터로 추천 모델링 시스템을 만들기
- 빅쿼리, Vertex AI를 사용하여 모델을 만들고 예측하기
- 유저수를 10000명으로 줄여서 예측

# **사용한 데이터셋 및 cpu/gpu 성능**

- [Instacart 데이터셋](https://www.kaggle.com/c/instacart-market-basket-analysis/data)
- 세부 특성 확인 : [Instacart data dictionary](https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b)
- version 1: vCPU 4개, 15GB RAM, GPU 없음 → 텐서플로우가 돌아가지 않음
    - 비용: 월 202 달러, 시간당 0.27 달러
- version2: vCPU 8개, 52GB RAM, GPU NVIDIA Tesla P4 1개
    - 비용: 월 940 달러, 시간당 1.27 달러

# Data import (BQ → Jupyter)

- 이미 빅쿼리에 csv 파일을 적재함
- 주피터 커널 안에 ‘%%’ 를 사용하면 빅쿼리 문법을 사용할 수 있음

```python
%%bigquery aisles
SELECT * FROM `your-project-id.your-view.aisles`
ORDER BY 1
```

```python
print(aisles.isna().sum())
aisles
```

![ ](images/Ecommerce_Rec/Untitled.png)

```python
%%bigquery departments
SELECT * FROM `your-project-id.your-view.departments`
ORDER BY 1
```

```python
print(departments.isna().sum())
departments
```

- 이런 형식으로 order_product_prior, order_product_train, orders, products 데이터 프레임을 생성

# 데이터 전처리: 결측치 처리 및 샘플링

```python
# 결측치 처리('days_since_prior_order':첫 구매인경우 nan값이 입력되어 있음 => 0으로 대체)
#orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0)
orders = orders.dropna()
```

```python
print(orders.shape)
orders['eval_set'].value_counts()
```

![ ](images/Ecommerce_Rec/Untitled%201.png)

```python
orders[(orders['user_id']==10)] # => 주문데이터 분석: 유저당 prior주문기록, train주문기록이 있는 유저들이 있음.
```

![ ](images/Ecommerce_Rec/Untitled%202.png)

```python
orders[orders['user_id'].isin(orders['user_id'][orders['eval_set']=='test'])] # test data -> order_product 정보가 없음
```

![ ](images/Ecommerce_Rec/Untitled%203.png)

```python
orders['user_id'].value_counts()
```

![ ](images/Ecommerce_Rec/Untitled%204.png)

```python
orders_user_id_eval_set = []
for i in orders['user_id'][orders['eval_set']=='train'].unique():
    i = int(i)
    orders_user_id_eval_set.append(i)
```

```python
import random
random.seed(2021)

# 1만명의 유저만 샘플링 (첫시도)
user = random.sample(orders_user_id_eval_set, 10000)
#user = orders['user_id'][orders['eval_set']=='train'].unique().tolist() #샘플링 안하고 시도 (실패)
# orders에서 test dataset 관련 기록(prior포함) 제외
orders_2 = orders[orders['user_id'].isin(user)]
```

```python
print("유저 인원:",len(user), "// 주문건수 합:",len(orders_2))
```

![ ](images/Ecommerce_Rec/Untitled%205.png)

```python
# order_product_prior에서 test dataset 관련 기록 제외
order_product_prior = order_product_prior[order_product_prior['order_id'].isin(orders_2['order_id'])]
```

# 데이터 조합

```python
# 제품 정보 : product + aisles + department data
product_m = products.merge(aisles).merge(departments)
product_m
```

![ ](images/Ecommerce_Rec/Untitled%206.png)

```python
import pandas as pd
# product one-hot encoding data 
product_enc = pd.get_dummies(product_m, columns=['aisle'], prefix=[None])
product_encㄹ
```

![ ](images/Ecommerce_Rec/Untitled%207.png)

```python
order_product = pd.concat([order_product_prior,order_product_train]) 
order_product
```

![ ](images/Ecommerce_Rec/Untitled%208.png)

```python
# 주문 항목 정보 : product_m + order_product
order_detail = order_product.merge(product_m,how='left') #.sample(n=100000, random_state=2021)
```

```python
order_detail#.sort_values(by=order_detail['order_id'])
```

![ ](images/Ecommerce_Rec/Untitled%209.png)

```python
data = orders_2.merge(order_detail)#, on ='order_id')
```

```python
data
```

![ ](images/Ecommerce_Rec/Untitled%2010.png)

```python
data.isna().sum()
```

![ ](images/Ecommerce_Rec/Untitled%2011.png)

# 데이터 탐색적 분석(EDA)

## 주문 행태 분석

```python
# 요일별 주문 현황

data['order_dow'].value_counts()

data.hist('order_dow',grid=False, bins=7)
```

![ ](images/Ecommerce_Rec/Untitled%2012.png)

```python
# 시간대별 주문 현황

data['order_hour_of_day'].value_counts()

data.hist('order_hour_of_day',grid=False, bins=24)
```

![ ](images/Ecommerce_Rec/Untitled%2013.png)

```python
# 주문 횟수 현황 
data['order_number'].value_counts()

data.hist('order_number',grid=False, bins=24)
```

![ ](images/Ecommerce_Rec/Untitled%2014.png)

```python
# 재주문 여부 현황 
data['reordered'].value_counts()

data.hist('reordered',grid=False)
```

![ ](images/Ecommerce_Rec/Untitled%2015.png)

# 유투브 추천 알고리즘 적용

## 데이터 전처리

### 데이터 Encoding

```python
data['user_id'] = data['user_id'].astype(int)
data['product_id'] = data['product_id'].astype(int)
data['order_id'] = data['order_id'].astype(int)
data['days_since_prior_order'] = data['days_since_prior_order'].astype(int)

data = data.set_index(['user_id']).sort_index()
data = data.reset_index()
```

```python
# 유저 인덱스 인코딩
user_ids = data["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
#userencoded2user = {i: x for i, x in enumerate(user_ids)}

# 주문 인덱스 인코딩
order_ids = data["order_id"].unique().tolist()
order2order_encoded = {x: i for i, x in enumerate(order_ids)}
#order_encoded2order = {i: x for i, x in enumerate(order_ids)}

# 상품 인덱스 인코딩
product_ids = data["product_id"].unique().tolist()
product2product_encoded = {x: i for i, x in enumerate(product_ids)}
#product_encoded2product = {i: x for i, x in enumerate(product_ids)}

# 상품 이름 인코딩
pd_name_ids = data["product_name"].unique().tolist()
pd_name2pd_name_encoded = {x: i for i, x in enumerate(pd_name_ids)}
#pd_name_encoded2pd_name = {i: x for i, x in enumerate(pd_name_ids)}
```

```python
# 상품 대분류 인덱스 인코딩
department_ids = []
for i in data["department_id"].unique():
    i = int(i)
    department_ids.append(i)
department2department_encoded = {x: i for i, x in enumerate(department_ids)}
#department_encoded2department = {i: x for i, x in enumerate(department_ids)}

# 상품 소분류 인덱스 인코딩
aisle_ids = []
for i in data["aisle_id"].unique():
    i = int(i)
    aisle_ids.append(i)
aisle2aisle_encoded = {x: i for i, x in enumerate(aisle_ids)}
#aisle_encoded2aisle = {i: x for i, x in enumerate(aisle_ids)}

# 상품 대분류명 인덱스 인코딩
dept_name_ids = data["department"].unique().tolist()
dept_name2dept_name_encoded = {x: i for i, x in enumerate(dept_name_ids)}
#dept_name_encoded2dept_name = {i: x for i, x in enumerate(dept_name_ids)}

# 상품 소분류명 인덱스 인코딩
aisle_name_ids = data["aisle"].unique().tolist()
aisle_name2aisle_name_encoded = {x: i for i, x in enumerate(aisle_name_ids)}
#aisle_name_encoded2aisle_name = {i: x for i, x in enumerate(aisle_name_ids)}
```

```python
# 인코딩으로 바꾸기
data["user"] = data["user_id"].map(user2user_encoded)
data["product"] = data["product_id"].map(product2product_encoded)
data["order"] = data["order_id"].map(order2order_encoded)
data["pd_name"] = data["product_name"].map(pd_name2pd_name_encoded)

# data["department"] = data["department_id"].map(department2department_encoded)
# data["aisle"] = data["aisle"].map(aisle2aisle_encoded)
# data["dept_name"] = data["department"].map(dept_name2dept_name_encoded)
# data["aisle_name"] = data["aisle"].map(aisle_name2aisle_name_encoded)
```

### User 기준으로 데이터 조정(feature engineering)

- 구매자 기준으로 데이터 프레임 재생성
- feature engineering 추가 기능

```python
order_hist = data.groupby(['user'])['order_id'].unique().apply(list).reset_index()
product_hist = data.groupby(['user'])['product_id'].apply(list).reset_index()
order_dow_hist = data.groupby(['user'])['order_dow'].apply(list).reset_index() # unique().적용해보기
order_hour_of_day_hist = data.groupby(['user'])['order_hour_of_day'].apply(list).reset_index()
days_since_prior_order_hist = data.groupby(['user'])['days_since_prior_order'].apply(list).reset_index()
```

```python
data.groupby(['user'])['order_dow'].unique().apply(list
```

![ ](images/Ecommerce_Rec/Untitled%2016.png)

```python
order_product_hist = data.groupby(['order'])['product_id'].apply(list).reset_index()
```

```python
order_hist # 사용자의 주문목록
```

![ ](images/Ecommerce_Rec/Untitled%2017.png)

```python
product_hist # 사용자가 구매한 상품
```

![ ](images/Ecommerce_Rec/Untitled%2018.png)

```python
order_product_hist
```

![ ](images/Ecommerce_Rec/Untitled%2019.png)

```python
order_dow_hist
```

![ ](images/Ecommerce_Rec/Untitled%2020.png)

```python
order_hour_of_day_hist
```

![ ](images/Ecommerce_Rec/Untitled%2021.png)

```python
days_since_prior_order_hist
```

![ ](images/Ecommerce_Rec/Untitled%2022.png)

```python
# User dataset 생성 (학습에 사용할 데이터, prior order:[data['eval_set']=='prior'])
user_data = data[['user','user_id']].merge(order_hist, how='left').merge(product_hist, how='left').merge(order_dow_hist, how='left').merge(order_hour_of_day_hist, how = 'left').merge(days_since_prior_order_hist,how='left') #eval_set
user_data
```

![ ](images/Ecommerce_Rec/Untitled%2023.png)

```python
user_data = user_data.drop_duplicates('user') # 중복데이터 삭제
user_data.shape
```

`(10000, 7)`

```python
data_product_prior=data['product'][data['eval_set']=='prior']
```

### **predict_label 생성 및 데이터 분할**

```python
user_data['predict_labels'] = user_data['product_id'].apply(lambda x: int(random.uniform(0,data['product_id'].max())))
#user_data['predict_labels'] = user_data['product_id'].apply(lambda x: int(random.uniform(0,data['product'].max())))
# (random.uniform(0,data["product"][data['eval_set']=='prior'].max())) train 데이터의 product중 하나 (=> 알맞은 데이터가 들어가는지 코드 검증 필요)
```

```python
user_data
```

![ ](images/Ecommerce_Rec/Untitled%2024.png)

```python
train_data = user_data[(user_data.user>=30) &
                       (user_data.user<=39)]
test_data = user_data[(user_data.user>=40) &
                      (user_data.user<=59)]
```

## 후보모델(candidate generator model)

```python
data["product_id"].max()
```

`49688`

```python
data["product"].max()
```

`35406`

```python
# 하이퍼파라미터 정의

EMBEDDING_DIMS = 16
DENSE_UNITS = 64
DROPOUT_PCT = 0.1
ALPHA = 0.1
NUM_CLASSES = data["product_id"].max() + 2 
LEARNING_RATE = 0.1
```

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

```python
# custom layers

import tensorflow as tf
class MaskedEmbeddingsAggregatorLayer(tf.keras.layers.Layer):
    def __init__(self, agg_mode='sum', **kwargs):
        super(MaskedEmbeddingsAggregatorLayer, self).__init__(**kwargs)

        if agg_mode not in ['sum', 'mean']:
            raise NotImplementedError('mode {} not implemented!'.format(agg_mode))
        self.agg_mode = agg_mode
    
    @tf.function
    def call(self, inputs, mask=None):
        masked_embeddings = tf.ragged.boolean_mask(inputs, mask)
        if self.agg_mode == 'sum':
            aggregated =  tf.reduce_sum(masked_embeddings, axis=1)
        elif self.agg_mode == 'mean':
            aggregated = tf.reduce_mean(masked_embeddings, axis=1)
        return aggregated
    
    def get_config(self):
        # this is used when loading a saved model that uses a custom layer
        return {'agg_mode': self.agg_mode}
    
class L2NormLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2NormLayer, self).__init__(**kwargs)
    
    @tf.function
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.ragged.boolean_mask(inputs, mask).to_tensor()
        return tf.math.l2_normalize(inputs, axis=-1)

    def compute_mask(self, inputs, mask):
        return mask
```

```python
# modeling
import tensorflow as tf
import datetime
import os

input_user = tf.keras.Input(shape=(None, ), name='user') 
input_product_hist = tf.keras.layers.Input(shape=(None,), name='product_hist')
input_order_dow_hist = tf.keras.layers.Input(shape=(None,), name='order_dow_hist')
input_order_hour_of_day_hist = tf.keras.Input(shape=(None, ), name='order_hour_of_day_hist')
input_days_since_prior_order_hist = tf.keras.Input(shape=(None, ), name='days_since_prior_order_hist')

# layer 구성
features_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS, 
                                            mask_zero=True, trainable=True, name='features_embeddings')
labels_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS, 
                                            mask_zero=True, trainable=True, name='labels_embeddings')

avg_embeddings = MaskedEmbeddingsAggregatorLayer(agg_mode='mean', name='aggregate_embeddings')

dense_1 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_1')
dense_2 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_2')
dense_3 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_3')
l2_norm_1 = L2NormLayer(name='l2_norm_1')
dense_output = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax, name='dense_output')

# feature 투입
features_embeddings = features_embedding_layer(input_user)
l2_norm_features = l2_norm_1(features_embeddings)
avg_features = avg_embeddings(l2_norm_features)

labels_product_embeddings = labels_embedding_layer(input_product_hist)
l2_norm_product = l2_norm_1(labels_product_embeddings)
avg_product = avg_embeddings(l2_norm_product)

labels_order_dow_embeddings = labels_embedding_layer(input_order_dow_hist)
l2_norm_order_dow = l2_norm_1(labels_order_dow_embeddings)
avg_order_dow = avg_embeddings(l2_norm_order_dow)

labels_order_hour_embeddings = labels_embedding_layer(input_order_hour_of_day_hist)
l2_norm_order_hour = l2_norm_1(labels_order_hour_embeddings)
avg_order_hour = avg_embeddings(l2_norm_order_hour)

labels_since_prior_embeddings = labels_embedding_layer(input_days_since_prior_order_hist)
l2_norm_since_prior = l2_norm_1(labels_since_prior_embeddings)
avg_since_prior = avg_embeddings(l2_norm_since_prior)

print(avg_features)
print(avg_order_dow)
print(avg_order_hour)
print(avg_since_prior)

# 임베딩 벡터들 연결
concat_inputs = tf.keras.layers.Concatenate(axis=1)([avg_product,
                                                     avg_order_dow, 
                                                     avg_order_hour,
                                                     avg_since_prior
                                                     ])
# Dense Layers
dense_1_features = dense_1(concat_inputs)
dense_1_relu = tf.keras.layers.ReLU(name='dense_1_relu')(dense_1_features)
dense_1_batch_norm = tf.keras.layers.BatchNormalization(name='dense_1_batch_norm')(dense_1_relu)

dense_2_features = dense_2(dense_1_relu)
dense_2_relu = tf.keras.layers.ReLU(name='dense_2_relu')(dense_2_features)
dense_2_batch_norm = tf.keras.layers.BatchNormalization(name='dense_2_batch_norm')(dense_2_relu)

dense_3_features = dense_3(dense_2_relu)
dense_3_relu = tf.keras.layers.ReLU(name='dense_3_relu')(dense_3_features)
dense_3_batch_norm = tf.keras.layers.BatchNormalization(name='dense_3_batch_norm')(dense_3_relu)

outputs = dense_output(dense_3_batch_norm)

#Optimizer
optimiser = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

#--- prep model
model = tf.keras.models.Model(
    inputs=[input_product_hist,
            input_order_dow_hist,
            input_order_hour_of_day_hist,
            input_days_since_prior_order_hist
            ],
    outputs=[outputs]
)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['acc']) 

model.summary()
```

![ ](images/Ecommerce_Rec/Untitled%2025.png)

![ ](images/Ecommerce_Rec/Untitled%2026.png)

```python
train_data
```

![ ](images/Ecommerce_Rec/Untitled%2027.png)

```python
# 학습(training)
history = model.fit([tf.keras.preprocessing.sequence.pad_sequences(train_data['product_id']),
                     tf.keras.preprocessing.sequence.pad_sequences(train_data['order_dow']),
                     tf.keras.preprocessing.sequence.pad_sequences(train_data['order_hour_of_day']), #+ 1e-10, dtype=float
                     tf.keras.preprocessing.sequence.pad_sequences(train_data['days_since_prior_order'])
                    ],train_data['predict_labels'].values,
                  #batch_size=16,
                  steps_per_epoch=1, epochs=300)
```

![ ](images/Ecommerce_Rec/Untitled%2028.png)

```python
# 모델 저장
model.save("candidate_generation_v2.h5")
```

```python
# 모델 예측결과 추출
pred = model.predict([tf.keras.preprocessing.sequence.pad_sequences(test_data['product_id']),
           tf.keras.preprocessing.sequence.pad_sequences(test_data['order_dow']),
           tf.keras.preprocessing.sequence.pad_sequences(test_data['order_hour_of_day']), #+ 1e-10, dtype=float
           tf.keras.preprocessing.sequence.pad_sequences(test_data['days_since_prior_order'])
           ])

pred
```

![ ](images/Ecommerce_Rec/Untitled%2029.png)

```python
# candidate generation: 
###### 각 user당 top-7개의 추천 후보 데이터(predict_label)를 뽑아낸다.
import numpy as np
N = 7
k = np.sort((-pred).argsort()[:,:N])
print(k)
k = k.flatten()
k[k>data["product"].max()]=0
k = np.unique(k)
```

![ ](images/Ecommerce_Rec/Untitled%2030.png)

```python
k
```

![ ](images/Ecommerce_Rec/Untitled%2031.png)

## 순위 모델 1,2안(ranking model 1,2)

### 1안)재주문여부(reordered), 1:like, 0:dislike

```python
# load candidate_generation 
model = tf.keras.models.load_model(
    'candidate_generation_v2.h5',
    custom_objects={
        'L2NormLayer':L2NormLayer,
        'MaskedEmbeddingsAggregatorLayer':MaskedEmbeddingsAggregatorLayer
    }
)
```

```python
# 아이템의 속성(ex.aisle)을 불러오는 함수 

aisle_cols = aisles['aisle'].values.tolist()
# type(aisle_cols)
type(aisle_cols)
aisle_cols

aisles_encoded = {x: i for i, x in enumerate(aisle_cols)}

# movies-> product_enc
# genres-> aisles
# aisles -> ['a', 'b', 'c', ...]
def get_aisles(products, aisles):
	def get_all_aisles(ai):
		active = [str(aisles_encoded[aisle]) for aisle, a in zip(aisles, ai) if a==1]
		if len(active) == 0:
			return '0'
		return ','.join((active))
	
	products['all_aisles'] = [
                              get_all_aisles(ai) for ai in zip(*[products[aisle] for aisle in aisles]) # 문제없음.
                              ]

get_aisles(product_enc, aisle_cols)
```

```python
product_enc.head(1)
```

![ ](images/Ecommerce_Rec/Untitled%2032.png)

```python
data['reordered'].value_counts()
```

![ ](images/Ecommerce_Rec/Untitled%2033.png)

```python
ratings = data[['product_id', 'reordered']]
ratings
```

![ ](images/Ecommerce_Rec/Untitled%2034.png)

```python
# normalize
def normalize_col(df,col_name):
    df[col_name] = (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())
    return df
```

```python
#-- rating 데이터 형태로 정리하기 

#-- 레퍼런스 코드와 대조하여 정리. 
# 레퍼런스: https://github.com/HYEZ/Deep-Youtube-Recommendations/blob/master/neural_net.ipynb
# dataframe: movies->product_enc
# movie_id -> product_id
# title -> product_name -> pd_name
# title2title_encoded ->pd_name2pd_name_encoded
# genre -> aisle

product_data = product_enc.set_index(['product_id']).sort_index()

# using candidate result 'k'
product_data = product_data.loc[k+1]
product_data["pd_name_d"] = product_data["product_name"].map(pd_name2pd_name_encoded)
print(product_data.head())

# 레퍼런스와 대조하여 feature 정리: 
 # rating -> reordered (1: like / 0: dislike)
 # unix_timestamp-> order_hour_of_day / 굳이 필요없으면 사용하지 않기. 
ratings_cols = ['user_id', 'product_id', 'reordered', 'order_hour_of_day']
ratings = data[['product_id', 'reordered', 'user_id', 'order_hour_of_day' ]]

# 모델에 사용할 new_data
new_data = product_data.merge(ratings, on='product_id') # rating 추가
print(new_data.columns)

aisle_occurences = new_data[aisle_cols].sum().to_dict()
aisles_encoded = {x: i for i, x in enumerate(aisle_cols)}

get_aisles(product_data, aisle_cols)
```

![ ](images/Ecommerce_Rec/Untitled%2035.png)

![ ](images/Ecommerce_Rec/Untitled%2036.png)

```python
new_data.columns
```

![ ](images/Ecommerce_Rec/Untitled%2037.png)

```python
new_data = new_data[['product_id', 'user_id', 'reordered', 'order_hour_of_day', 'all_aisles', 'pd_name_d']]
new_data['product_type'] = np.where(new_data['reordered'] ==1, 'like', 'dislike') # 1이면 like

product_list = new_data.groupby(['user_id','product_type'])['product_id'].apply(list).reset_index()

aisle_list = new_data.groupby(['user_id'])['all_aisles'].unique().apply(list).reset_index()
aisle_list['all_aisles']=aisle_list['all_aisles'].apply(lambda x: list(set(','.join(x))) ) # 중복제거
aisle_list['all_aisles']=aisle_list['all_aisles'].apply(lambda x:[ x for x in x if x.isdigit() ])
    
# normalize
new_data = normalize_col(new_data, 'order_hour_of_day')
order_hour_of_day_list = new_data.groupby(['user_id'])['order_hour_of_day'].unique().apply(list).reset_index()

pd_name_list = new_data.groupby(['user_id'])['pd_name_d'].apply(list).reset_index()
print(product_list)
dataset = product_list.pivot(index='user_id', columns='product_type', values='product_id').reset_index()
dataset.fillna(new_data["product_id"].max()+1, inplace=True)

dataset['like'] =dataset['like'].apply(lambda x: x if type(x) is list else [])
dataset['dislike'] =dataset['dislike'].apply(lambda x: x if type(x) is list else [])

dataset = pd.merge(dataset, pd_name_list, how='left')
dataset = pd.merge(dataset, aisle_list, how='left')
dataset = pd.merge(dataset, order_hour_of_day_list, how='left')

dataset['predict_labels'] = dataset['like'].apply(lambda x: int(random.uniform(1,new_data["product_id"].max()))) 

dataset['like']=dataset['like'].apply(lambda x: [new_data["product_id"].max()+1] if x == [] else x)
dataset['dislike']=dataset['dislike'].apply(lambda x: [new_data["product_id"].max()+1] if x == [] else x)
# train_data=dataset[(dataset.user_id >= 1)&
#                                   (dataset.user_id <= 5)]
# test_data=dataset[(dataset.user_id >= 6)&
#                                   (dataset.user_id <= 9)]
train_data_r=dataset[(dataset.index >= 0)&
                                  (dataset.index <= 9)]
test_data_r=dataset[(dataset.index >= 20)&
                                  (dataset.index <= 24)]

print(dataset.index)
```

![ ](images/Ecommerce_Rec/Untitled%2038.png)

```python
train_data_r
```

![ ](images/Ecommerce_Rec/Untitled%2039.png)

```python
test_data_r
```

![ ](images/Ecommerce_Rec/Untitled%2040.png)

```python
# 하이퍼파라미터 정의
EMBEDDING_DIMS = 16
DENSE_UNITS = 64
DROPOUT_PCT = 0.1
ALPHA = 0.0
NUM_CLASSES=new_data["product_id"].max() + 3
LEARNING_RATE = 0.003
```

```python
# 모델
#---inputs
import tensorflow as tf
import datetime
import os
input_name = tf.keras.Input(shape=(None, ), name='product_name')
inp_item_liked = tf.keras.layers.Input(shape=(None,), name='like')
inp_item_disliked = tf.keras.layers.Input(shape=(None,), name='dislike')
input_aisle = tf.keras.Input(shape=(None, ), name='aisle')
input_order_hour = tf.keras.Input(shape=(None, ), name='order_hour')

#--- layers
features_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS, 
                                            mask_zero=True, trainable=True, name='features_embeddings')
labels_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS, 
                                            mask_zero=True, trainable=True, name='labels_embeddings')

avg_embeddings = MaskedEmbeddingsAggregatorLayer(agg_mode='mean', name='aggregate_embeddings')

dense_1 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_1')
dense_2 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_2')
dense_3 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_3')
l2_norm_1 = L2NormLayer(name='l2_norm_1')

dense_output = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax, name='dense_output')

#--- features
features_embeddings = features_embedding_layer(input_name)
l2_norm_features = l2_norm_1(features_embeddings)
avg_features = avg_embeddings(l2_norm_features)

labels_liked_embeddings = labels_embedding_layer(inp_item_liked)
l2_norm_liked = l2_norm_1(labels_liked_embeddings)
avg_liked = avg_embeddings(l2_norm_liked)

labels_disliked_embeddings = labels_embedding_layer(inp_item_disliked)
l2_norm_disliked = l2_norm_1(labels_disliked_embeddings)
avg_disliked = avg_embeddings(l2_norm_disliked)

labels_aisle_embeddings = labels_embedding_layer(input_aisle)
l2_norm_aisle = l2_norm_1(labels_aisle_embeddings)
avg_aisle = avg_embeddings(l2_norm_aisle)

labels_order_hour_embeddings = labels_embedding_layer(input_order_hour)
l2_norm_order_hour = l2_norm_1(labels_order_hour_embeddings)
avg_order_hour = avg_embeddings(l2_norm_order_hour)

# 임베딩 벡터들 연결
concat_inputs = tf.keras.layers.Concatenate(axis=1)([avg_features,
                                                     avg_liked,
                                                     avg_disliked,
                                                     avg_aisle,
                                                     avg_order_hour
                                                     ])
# Dense Layers
dense_1_features = dense_1(concat_inputs)
dense_1_relu = tf.keras.layers.ReLU(name='dense_1_relu')(dense_1_features)
dense_1_batch_norm = tf.keras.layers.BatchNormalization(name='dense_1_batch_norm')(dense_1_relu)

dense_2_features = dense_2(dense_1_relu)
dense_2_relu = tf.keras.layers.ReLU(name='dense_2_relu')(dense_2_features)
# dense_2_batch_norm = tf.keras.layers.BatchNormalization(name='dense_2_batch_norm')(dense_2_relu)

dense_3_features = dense_3(dense_2_relu)
dense_3_relu = tf.keras.layers.ReLU(name='dense_3_relu')(dense_3_features)
dense_3_batch_norm = tf.keras.layers.BatchNormalization(name='dense_3_batch_norm')(dense_3_relu)
outputs = dense_output(dense_3_batch_norm)

#Optimizer
optimiser = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

#--- prep model
model = tf.keras.models.Model(
    inputs=[input_name, 
            inp_item_liked, 
            inp_item_disliked,
            input_aisle,
            input_order_hour,
            ],
    outputs=[outputs]
)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['acc'])

model.summary()
```

`Model: "model_3"`

![ ](images/Ecommerce_Rec/Untitled%2041.png)

![ ](images/Ecommerce_Rec/Untitled%2042.png)

```python
#학습
train_data_r.columns
```

![ ](images/Ecommerce_Rec/Untitled%2043.png)

```python
ratings1 = model.fit([tf.keras.preprocessing.sequence.pad_sequences(train_data_r['pd_name_d'])+ 1e-10,
           tf.keras.preprocessing.sequence.pad_sequences(train_data_r['like'])+ 1e-10,
           tf.keras.preprocessing.sequence.pad_sequences(train_data_r['dislike'])+ 1e-10,
            tf.keras.preprocessing.sequence.pad_sequences(train_data_r['all_aisles'])+ 1e-10,
            tf.keras.preprocessing.sequence.pad_sequences(train_data_r['order_hour_of_day'], dtype=float) + 1e-10,
           ],train_data_r['predict_labels'].values,
           steps_per_epoch=1, epochs=300)
```

![ ](images/Ecommerce_Rec/Untitled%2044.png)

```python
# prediction 
pred_1 = model.predict([tf.keras.preprocessing.sequence.pad_sequences(test_data_r['pd_name_d'])+ 1e-10,
           tf.keras.preprocessing.sequence.pad_sequences(test_data_r['like'])+ 1e-10,
           tf.keras.preprocessing.sequence.pad_sequences(test_data_r['dislike'])+ 1e-10,
            tf.keras.preprocessing.sequence.pad_sequences(test_data_r['all_aisles'])+ 1e-10,
            tf.keras.preprocessing.sequence.pad_sequences(test_data_r['order_hour_of_day'], dtype=float) + 1e-10
           ])

pred_1
```

![ ](images/Ecommerce_Rec/Untitled%2045.png)

**ranking 결과**

```python
# ranking
###### 각 user당 top-7개의 추천 데이터를 뽑아낸다.
N = 7
# k = np.sort((-pred_1).argsort()[:,:N]) 
# np.argsort(): probability의 마이너스값을 작은 값부터 순서대로 데이터의 인덱스를 반환. ==> 즉, 양의값으로는 큰 값부터 반환한 셈.
# np.sort(): 인덱스 순으로 다시 정렬. 확률이 더 높은 것부터 보고싶으므로 레퍼런스에 있는 코드이지만 사용하지 않음. 
ranking_1 = (-pred_1).argsort()[:, :N]

ranking_1[ranking_1>new_data['product_id'].max()]=0 

print(ranking_1)

ranking_1_probability = np.sort(pred_1[:, :N])
print(ranking_1_probability)
```

![ ](images/Ecommerce_Rec/Untitled%2046.png)

### 2안) order_number: 18번째 이상 like, 18번째 미만 dislike

```python
# load candidate_generation 
model = tf.keras.models.load_model(
    'candidate_generation_v2.h5',
    custom_objects={
        'L2NormLayer':L2NormLayer,
        'MaskedEmbeddingsAggregatorLayer':MaskedEmbeddingsAggregatorLayer
    }
)
```

```python
# 아이템의 다른 속성(ex. aisle)을 불러오는 기능

aisle_cols = aisles['aisle'].values.tolist()
# type(aisle_cols)
type(aisle_cols)
aisle_cols

aisles_encoded = {x: i for i, x in enumerate(aisle_cols)}

# movies-> product_enc
# genres-> aisles
# aisles -> ['a', 'b', 'c', ...]
def get_aisles(products, aisles):
	def get_all_aisles(ai):
		active = [str(aisles_encoded[aisle]) for aisle, a in zip(aisles, ai) if a==1]
		if len(active) == 0:
			return '0'
		return ','.join((active))
	
	products['all_aisles'] = [
                              get_all_aisles(ai) for ai in zip(*[products[aisle] for aisle in aisles]) # 문제없음.
                              ]

get_aisles(product_enc, aisle_cols)
```

```python
product_enc.head(1)
```

![ ](images/Ecommerce_Rec/Untitled%2047.png)

```python
data.head()
```

![ ](images/Ecommerce_Rec/Untitled%2048.png)

```python
data[['order_number']].value_counts()
```

![ ](images/Ecommerce_Rec/Untitled%2049.png)

```python
# 주문데이터의 평균 주문회차 
data['order_number'].mean()
```

`18.063830654172047`

```python
data.shape
```

`(1570703, 19)`

```python
# order_number분포 확인 
import matplotlib.pyplot as plt
data['order_number'].value_counts()

data.hist('order_number',grid=False, bins = 30)
plt.minorticks_on()
plt.tick_params(axis='both', which ='both', direction='in', pad=8, top=True, right=True)
```

![ ](images/Ecommerce_Rec/Untitled%2050.png)

```python
data['order_number'].min(), data['order_number'].max(), data['order_number'].mean()
```

`(2, 100, 18.063830654172047)`

```python
# ratings_2 (order_number: 18이상 like / 18미만 dislike)
# 레퍼런스: https://www.delftstack.com/ko/howto/python-pandas/how-to-create-dataframe-column-based-on-given-condition-in-pandas/
# data['ratings_2'] = data['order_number'][(data['order_number']>=18)]

import numpy as np 

conditionlist = [ data['order_number'].to_numpy() >= 18, data['order_number'].to_numpy()<18]

choicelist = [1, 0]

data['ratings_2'] = np.select(conditionlist, choicelist)

print(data['ratings_2'].value_counts())
print(data.columns)
# 데이터 data['ratings_2'] = 1,0 값 확인
# data[['order_number', 'ratings_2']][(data['order_number']<18)].sort_values(by='order_number', ascending=False)
data[['order_number', 'ratings_2']][(data['order_number']>=18)].sort_values(by='order_number', ascending=False)
```

![ ](images/Ecommerce_Rec/Untitled%2051.png)

```python
# normalize
def normalize_col(df,col_name):
    df[col_name] = (df[col_name] - df[col_name].min()) / (df[col_name].max() - df[col_name].min())
    return df
```

```python
#-- rating 데이터 형태로 정리하기 

#-- 레퍼런스 코드와 대조하여 정리. 
# 레퍼런스: https://github.com/HYEZ/Deep-Youtube-Recommendations/blob/master/neural_net.ipynb
# dataframe: movies->product_enc
# movie_id -> product_id
# title -> product_name -> pd_name
# title2title_encoded ->pd_name2pd_name_encoded
# genre -> aisle

# 1안에서 사용한 product_data 그대로 사용. 아래는 에러 발생시 대안 코드. 
# product_data_2 = product_enc.set_index(['product_id']).sort_index()
# using candidate result 'k'
# product_data_2 = product_data_2.loc[k+1]
# product_data_2["pd_name_d"] = product_data_2["product_name"].map(pd_name2pd_name_encoded)
# print(product_data.head())

# 레퍼런스와 대조하여 feature 정리: 
 # rating -> reordered (1: like / 0: dislike)
 # unix_timestamp-> order_hour_of_day / 굳이 필요없으면 사용하지 않기. 
ratings_2_cols = ['user_id', 'product_id', 'ratings_2', 'order_hour_of_day']
ratings_2 = data[['product_id', 'ratings_2', 'user_id', 'order_hour_of_day' ]]

# 모델에 사용할 new_data
new_data_2 = product_data.merge(ratings_2, on='product_id') # rating 추가
print(new_data.columns)

aisle_occurences = new_data_2[aisle_cols].sum().to_dict()
aisles_encoded = {x: i for i, x in enumerate(aisle_cols)}

get_aisles(product_data, aisle_cols)
```

![ ](images/Ecommerce_Rec/Untitled%2052.png)

```python
print(product_data.columns)
print(new_data_2.columns)
```

![ ](images/Ecommerce_Rec/Untitled%2053.png)

```python
new_data_2 = new_data_2[['product_id', 'user_id', 'ratings_2', 'order_hour_of_day', 'all_aisles', 'pd_name_d']]
new_data_2['product_type'] = np.where(new_data_2['ratings_2'] ==1, 'like', 'dislike') # 1이면 like

product_list_2 = new_data_2.groupby(['user_id','product_type'])['product_id'].apply(list).reset_index()

aisle_list_2 = new_data_2.groupby(['user_id'])['all_aisles'].unique().apply(list).reset_index()
aisle_list_2['all_aisles']=aisle_list_2['all_aisles'].apply(lambda x: list(set(','.join(x))) ) # 중복제거
aisle_list_2['all_aisles']=aisle_list_2['all_aisles'].apply(lambda x:[ x for x in x if x.isdigit() ])
    
# normalize
new_data_2 = normalize_col(new_data_2, 'order_hour_of_day')
order_hour_of_day_list_2 = new_data_2.groupby(['user_id'])['order_hour_of_day'].unique().apply(list).reset_index()

pd_name_list_2 = new_data_2.groupby(['user_id'])['pd_name_d'].apply(list).reset_index()
print(product_list_2)
dataset_2 = product_list_2.pivot(index='user_id', columns='product_type', values='product_id').reset_index()
dataset_2.fillna(new_data_2["product_id"].max()+1, inplace=True)

dataset_2['like'] =dataset_2['like'].apply(lambda x: x if type(x) is list else [])
dataset_2['dislike'] =dataset_2['dislike'].apply(lambda x: x if type(x) is list else [])

dataset_2 = pd.merge(dataset_2, pd_name_list_2, how='left')
dataset_2 = pd.merge(dataset_2, aisle_list_2, how='left')
dataset_2 = pd.merge(dataset_2, order_hour_of_day_list_2, how='left')

dataset_2['predict_labels'] = dataset_2['like'].apply(lambda x: int(random.uniform(1,new_data_2["product_id"].max()))) 

dataset_2['like']=dataset_2['like'].apply(lambda x: [new_data_2["product_id"].max()+1] if x == [] else x)
dataset_2['dislike']=dataset_2['dislike'].apply(lambda x: [new_data_2["product_id"].max()+1] if x == [] else x)
# train_data=dataset[(dataset.user_id >= 1)&
#                                   (dataset.user_id <= 5)]
# test_data=dataset[(dataset.user_id >= 6)&
#                                   (dataset.user_id <= 9)]
train_data_r_2=dataset_2[(dataset_2.index >= 0)&
                                  (dataset_2.index <= 9)]
test_data_r_2=dataset_2[(dataset_2.index >= 20)&
                                  (dataset_2.index <= 24)]

print(dataset_2.index)
```

![ ](images/Ecommerce_Rec/Untitled%2054.png)

```python
train_data_r_2
```

![ ](images/Ecommerce_Rec/Untitled%2055.png)

```python
test_data_r_2
```

![ ](images/Ecommerce_Rec/Untitled%2056.png)

```python
# 하이퍼 파라미터 정의
EMBEDDING_DIMS = 16
DENSE_UNITS = 64
DROPOUT_PCT = 0.1
ALPHA = 0.0
NUM_CLASSES=new_data_2["product_id"].max() + 3
LEARNING_RATE = 0.003
```

```python
# 모델
#---inputs
import tensorflow as tf
import datetime
import os
input_name = tf.keras.Input(shape=(None, ), name='product_name')
inp_item_liked = tf.keras.layers.Input(shape=(None,), name='like')
inp_item_disliked = tf.keras.layers.Input(shape=(None,), name='dislike')
input_aisle = tf.keras.Input(shape=(None, ), name='aisle')
input_order_hour = tf.keras.Input(shape=(None, ), name='order_hour')

#--- layers
features_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS, 
                                            mask_zero=True, trainable=True, name='features_embeddings')
labels_embedding_layer = tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=EMBEDDING_DIMS, 
                                            mask_zero=True, trainable=True, name='labels_embeddings')

avg_embeddings = MaskedEmbeddingsAggregatorLayer(agg_mode='mean', name='aggregate_embeddings')

dense_1 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_1')
dense_2 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_2')
dense_3 = tf.keras.layers.Dense(units=DENSE_UNITS, name='dense_3')
l2_norm_1 = L2NormLayer(name='l2_norm_1')

dense_output = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax, name='dense_output')

#--- features
features_embeddings = features_embedding_layer(input_name)
l2_norm_features = l2_norm_1(features_embeddings)
avg_features = avg_embeddings(l2_norm_features)

labels_liked_embeddings = labels_embedding_layer(inp_item_liked)
l2_norm_liked = l2_norm_1(labels_liked_embeddings)
avg_liked = avg_embeddings(l2_norm_liked)

labels_disliked_embeddings = labels_embedding_layer(inp_item_disliked)
l2_norm_disliked = l2_norm_1(labels_disliked_embeddings)
avg_disliked = avg_embeddings(l2_norm_disliked)

labels_aisle_embeddings = labels_embedding_layer(input_aisle)
l2_norm_aisle = l2_norm_1(labels_aisle_embeddings)
avg_aisle = avg_embeddings(l2_norm_aisle)

labels_order_hour_embeddings = labels_embedding_layer(input_order_hour)
l2_norm_order_hour = l2_norm_1(labels_order_hour_embeddings)
avg_order_hour = avg_embeddings(l2_norm_order_hour)

# 임베딩 벡터들 연결
concat_inputs = tf.keras.layers.Concatenate(axis=1)([avg_features,
                                                     avg_liked,
                                                     avg_disliked,
                                                     avg_aisle,
                                                     avg_order_hour
                                                     ])
# Dense Layers
dense_1_features = dense_1(concat_inputs)
dense_1_relu = tf.keras.layers.ReLU(name='dense_1_relu')(dense_1_features)
dense_1_batch_norm = tf.keras.layers.BatchNormalization(name='dense_1_batch_norm')(dense_1_relu)

dense_2_features = dense_2(dense_1_relu)
dense_2_relu = tf.keras.layers.ReLU(name='dense_2_relu')(dense_2_features)
# dense_2_batch_norm = tf.keras.layers.BatchNormalization(name='dense_2_batch_norm')(dense_2_relu)

dense_3_features = dense_3(dense_2_relu)
dense_3_relu = tf.keras.layers.ReLU(name='dense_3_relu')(dense_3_features)
dense_3_batch_norm = tf.keras.layers.BatchNormalization(name='dense_3_batch_norm')(dense_3_relu)
outputs = dense_output(dense_3_batch_norm)

#Optimizer
optimiser = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

#--- prep model
model = tf.keras.models.Model(
    inputs=[input_name, 
            inp_item_liked, 
            inp_item_disliked,
            input_aisle,
            input_order_hour,
            ],
    outputs=[outputs]
)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['acc'])

model.summary()
```

`Model: "model_4"`

![ ](images/Ecommerce_Rec/Untitled%2057.png)

![ ](images/Ecommerce_Rec/Untitled%2058.png)

```python
# 학습
train_data_r_2.columns
```

![ ](images/Ecommerce_Rec/Untitled%2059.png)

```python
ranking_2 = model.fit([tf.keras.preprocessing.sequence.pad_sequences(train_data_r_2['pd_name_d'])+ 1e-10,
           tf.keras.preprocessing.sequence.pad_sequences(train_data_r_2['like'])+ 1e-10,
           tf.keras.preprocessing.sequence.pad_sequences(train_data_r_2['dislike'])+ 1e-10,
            tf.keras.preprocessing.sequence.pad_sequences(train_data_r_2['all_aisles'])+ 1e-10,
            tf.keras.preprocessing.sequence.pad_sequences(train_data_r_2['order_hour_of_day'], dtype=float) + 1e-10,
           ],train_data_r_2['predict_labels'].values,
           steps_per_epoch=1, epochs=300)
```

![ ](images/Ecommerce_Rec/Untitled%2060.png)

![ ](images/Ecommerce_Rec/Untitled%2061.png)

```python
# prediction 
pred_2 = model.predict([tf.keras.preprocessing.sequence.pad_sequences(test_data_r_2['pd_name_d'])+ 1e-10,
           tf.keras.preprocessing.sequence.pad_sequences(test_data_r_2['like'])+ 1e-10,
           tf.keras.preprocessing.sequence.pad_sequences(test_data_r_2['dislike'])+ 1e-10,
            tf.keras.preprocessing.sequence.pad_sequences(test_data_r_2['all_aisles'])+ 1e-10,
            tf.keras.preprocessing.sequence.pad_sequences(test_data_r_2['order_hour_of_day'], dtype=float) + 1e-10
           ])
pred_2
```

![ ](images/Ecommerce_Rec/Untitled%2062.png)

**ranking 결과**

```python
# ranking
###### 각 user당 top-7개의 추천 데이터를 뽑아낸다.
N = 7
# k = np.sort((-pred_1).argsort()[:,:N]) 
# np.argsort(): probability의 마이너스값을 작은 값부터 순서대로 데이터의 인덱스를 반환. ==> 즉, 양의값으로는 큰 값부터 반환한 셈.
# np.sort(): 인덱스 순으로 다시 정렬. 확률이 더 높은 것부터 보고싶으므로 레퍼런스에 있는 코드이지만 사용하지 않음. 
ranking_2 = (-pred_2).argsort()[:, :N]

ranking_2[ranking_2>new_data_2['product_id'].max()]=0 

print(ranking_2)

ranking_2_probability = np.sort(pred_2[:, :N])
print(ranking_2_probability)

# 추출된 인덱스 값들은 user별 구매목록라벨(여러개 상품들이 있는). 유사도 높은 다른 유저들의 구매목록이 추천된 셈.
```

![ ](images/Ecommerce_Rec/Untitled%2063.png)

## 순위모델 1,2안 평과 결과(ranking-eval(nDCG score))

```python
# 1안의 유저별 추천 '상품묶음' 라벨들. 
ranking_1
```

![ ](images/Ecommerce_Rec/Untitled%2064.png)

```python
# -- 라벨의 like_id값 비교용 
# pd_name_d
# train_data_r[['predict_labels','pd_name_d']]
train_df1 = train_data_r[['like', 'predict_labels']]
train_df1 = train_df1.rename(columns = {'like':'like_id', 'predict_labels': 'label'})
train_df1
```

![ ](images/Ecommerce_Rec/Untitled%2065.png)

```python
# predict_labels: 유저별 라벨 역할. 
# pd_name_d : user A가 구매한 상품명 인덱스 수치 묶음.
# 비슷한 유저의 상품 목록을 추천하는 것으로 생각. 

true_df1 = test_data_r[['user_id', 'like', 'predict_labels']]
# pred_data_1
true_df1 = true_df1.rename(columns={'like': 'true_like_id'
                                          ,'predict_labels': 'true_labels'})
true_df1 = true_df1[['user_id', 'true_like_id']]
true_df1
```

![ ](images/Ecommerce_Rec/Untitled%2066.png)

### rank_1안 최종 ndcg 값(함수 사용)

```python
# user 당 추천목록 데이터 추출하는 기능
  
def user_rec(n , rank, rec_df, train_df):
  # 전달인자(argument) 설명 :
   ## n: 몇번째 유저인지(the order of test-user starting from 0) 
   ## rank: 랭킹 결과 (ranking result of ranking model (array type))
   ## rec_df: 유저당 추천된 라벨과 라벨에 해당하는 상품 아이디 데이터프레임 (recommended label and product_id for each user)
   ## train_df: 랭킹 모델학습에 사용된 학습데이터에서 필요한 like, predict_labels만 추출한 데이터프레임(train_data's 'like and predict_labels'features used for ranking_model training)

  for label in rank[n-1]: # user2번째(array index=1) DF 만들기.
    rec_id = train_df['like_id'][(train_df['label']==label)].tolist()
    rec_df = rec_df.append(pd.DataFrame([[label, rec_id]], columns=['label', 'rec_id']), ignore_index=True)

  # rec_id = train_data['product_id'][(train_data['label']==label)].values
  
  return rec_df

# ---user1
# user1_rec_data = pd.DataFrame(columns=['label', 'rec_id', 'T/F'])
user1_rec_data = pd.DataFrame(columns=['label', 'rec_id'])

rec_df1 = user_rec(1, ranking_1, user1_rec_data, train_df1)
print('user1_rec_data')
print(rec_df1)

# ---user2
# user2_rec_data = pd.DataFrame(columns=['label', 'rec_id', 'T/F'])
user2_rec_data = pd.DataFrame(columns=['label', 'rec_id'])

rec_df2 = user_rec(2, ranking_1, user2_rec_data, train_df1)
print('user2_rec_data')
print(rec_df2)

# ---user3
user3_rec_data = pd.DataFrame(columns=['label', 'rec_id'])

rec_df3 = user_rec(3, ranking_1, user3_rec_data, train_df1)
print('user3_rec_data')
print(rec_df3)

# ---user4
user4_rec_data = pd.DataFrame(columns=['label', 'rec_id'])

rec_df4 = user_rec(4, ranking_1, user4_rec_data, train_df1)
print('user4_rec_data')
print(rec_df4)

# ---user5
user5_rec_data = pd.DataFrame(columns=['label', 'rec_id'])

rec_df5 = user_rec(5, ranking_1, user5_rec_data, train_df1)
print('user5_rec_data')
print(rec_df5)
```

![ ](images/Ecommerce_Rec/Untitled%2067.png)

```python
# 각 유저별 추천목록의 상품들 중 true-like한 것이 있는지 T/F(trud:1/false:0)으로 데이터프레임에 넣어주는 기능.

def true_like(n, rec_df, true_df):
  # 전달인자(argument)설명: 
   ## n: 몇번째 유저인지 (the order of test-user starting from 0)
   ## rec_df: 유저당 추천된 라벨과 라벨에 해당하는 상품 아이디 데이터프레임 (recommended label and product_id for each user)
   ## true_df: 랭킹모델의 예측에 사용된 test_data중에서 필요한 'user_id, true_like_id'특성만 추출한 데이터프레임(test_data's 'user_id and true_like_id'features used for ranking_model predicting)
  tf_list = []
  rec_id = rec_df['rec_id'].tolist() 
  # rec_id = rec_df['rec_id'].values 

  # true데이터인 test_data의 형태상, row에 있는 유저 n-1번째 row를 고정해야 함. 
  true = true_df['true_like_id'].tolist()[n-1]

  for rec in rec_id[:][:]:
    if len(rec) == 0:
      tf_list.append(0.0)

    elif len(rec)>=1:        
      tf = true[0] in rec[0]
      if tf == True:
        tf_list.append(1)
      else:
        tf_list.append(0)
      rec_df['T/F'] = pd.DataFrame(tf_list)

  print("true_like_id:", true)

  return rec_df

# ---user1의 추천목록에 대한 T/F(true-like or false) 결과 
rec_df1 = true_like(1, rec_df1, true_df1)
print(rec_df1)

# ---user2의 추천목록에 대한 T/F(true-like or false) 결과 
rec_df2 = true_like(2, rec_df2, true_df1)
print(rec_df2)

# ---user3의 추천목록에 대한 T/F(true-like or false) 결과 
rec_df3 = true_like(3, rec_df3, true_df1)
print(rec_df3)

# ---user4의 추천목록에 대한 T/F(true-like or false) 결과 
rec_df4 = true_like(4, rec_df4, true_df1)
print(rec_df4)

# ---user5의 추천목록에 대한 T/F(true-like or false) 결과 
rec_df5 = true_like(5, rec_df5, true_df1)
print(rec_df5)
```

![ ](images/Ecommerce_Rec/Untitled%2068.png)

```python
# 각 유저별 dcg, idcg, ndcg 계산하는 기능

import numpy as np
def get_ndcg(rec_df):
  # rec_df: 유저당 추천된 '라벨'과 '라벨에 해당하는 상품 아이디', 'T/F(true_like여부)'특성이 포함된 데이터프레임 (dataframe about recommended 'label' and 'product_id', 'T/F(true_like or not)' for each user)
  rec = rec_df['rec_id'].tolist()
  t = rec_df['T/F'].tolist()
  dcg = 0.0
  
  #dcg 계산
  for i, j in enumerate(t): # i, j: T/F안의 인덱스와 값들(1.0, 0.0..)을 돈다.
    if j == 1.0:
      dcg += (1.0/np.log2(i+1+1))
    else: 
      dcg += 0
  
  #idcg 계산
  idcg = sum((1.0/np.log2(i+1+1) for i in range(0, len(t)+1)))

  #ndcg 계산
  ndcg = dcg / idcg 
  
  # set 으로 결과 산출할 경우,
  # return dcg, idcg, ndcg
  
  #dataframe으로 결과 산출할 경우,
  ndcg_df = pd.DataFrame(columns=['dcg', 'idcg', 'ndcg'])
  ndcg_df = ndcg_df.append(pd.DataFrame([[dcg, idcg, ndcg]], columns=['dcg', 'idcg', 'ndcg']), ignore_index=True)

  return ndcg_df

# --- user1의 dcg, idcg, ndcg 
ndcg_df1 = get_ndcg(rec_df1)
print(ndcg_df1)

# --- user2의 dcg, idcg, ndcg 
ndcg_df2 = get_ndcg(rec_df2)
print(ndcg_df2)

# --- user3의 dcg, idcg, ndcg 
ndcg_df3 = get_ndcg(rec_df3)
print(ndcg_df3)

# --- user4의 dcg, idcg, ndcg 
ndcg_df4 = get_ndcg(rec_df4)
print(ndcg_df4)

# --- user5의 dcg, idcg, ndcg 
ndcg_df5 = get_ndcg(rec_df5)
print(ndcg_df5)
```

![ ](images/Ecommerce_Rec/Untitled%2069.png)

```python
# user별 dcg, idcg, ndcg 결과를 합쳐주는 함수 기능. 
import pandas as pd
def concat_result(df1, df2):
  df3 = pd.concat([df1, df2])
  
  return df3

result = concat_result(ndcg_df1, ndcg_df2)
result = concat_result(result, ndcg_df3)
result = concat_result(result, ndcg_df4)
result = concat_result(result, ndcg_df5)
result.index=['test-user1', 'test-user2', 'test-user3', 'test-user4', 'test-user5']
print('[ ranking_1안의 유저별 dcg, idcg, ndcg ] \n ', result)
print('-------------------\n')
print('[ ranking_1안의 dcg, idcg, ndcg평균 ] \n', result.mean())
```

![ ](images/Ecommerce_Rec/Untitled%2070.png)

### rank_2안(order_number 기준 like/dislike 에 따른 순위 목록)의 nDCG

```python
# 1안의 유저별 추천 '상품묶음' 라벨들. 
ranking_2
```

![ ](images/Ecommerce_Rec/Untitled%2071.png)

```python
# pd_name_d
# train_data_r[['predict_labels','pd_name_d']]
train_df2 = train_data_r_2[['like', 'predict_labels']].rename(columns={'like': 'like_id', 'predict_labels':'label'})
train_df2
```

![ ](images/Ecommerce_Rec/Untitled%2072.png)

```python
# predict_labels: 유저별 라벨 역할. 
# pd_name_d : user A가 구매한 상품명 인덱스 수치 묶음.
# 비슷한 유저의 상품 목록을 추천하는 것으로 생각. 

true_df2 = test_data_r_2[['user_id', 'like']].rename(columns={'like': 'true_like_id'})
true_df2
```

![ ](images/Ecommerce_Rec/Untitled%2073.png)

### rank_2안 최종 ndcg 값

```python
# user 당 추천목록 데이터 추출하는 기능
  
def user_rec(n , rank, rec_df, train_df):
  # 전달인자(argument) 설명 :
   ## n: 몇번째 유저인지(the order of test-user starting from 0) 
   ## rank: 랭킹 결과 (ranking result of ranking model (array type))
   ## rec_df: 유저당 추천된 라벨과 라벨에 해당하는 상품 아이디 데이터프레임 (recommended label and product_id for each user)
   ## train_df: 랭킹 모델학습에 사용된 학습데이터에서 필요한 like, predict_labels만 추출한 데이터프레임(train_data's 'like and predict_labels'features used for ranking_model training)

  for label in rank[n-1]: # user2번째(array index=1) DF 만들기.
    rec_id = train_df['like_id'][(train_df['label']==label)].tolist()
    rec_df = rec_df.append(pd.DataFrame([[label, rec_id]], columns=['label', 'rec_id']), ignore_index=True)

  # rec_id = train_data['product_id'][(train_data['label']==label)].values
  
  return rec_df

# ---user1
# user1_rec_data = pd.DataFrame(columns=['label', 'rec_id', 'T/F'])
user1_rec_data = pd.DataFrame(columns=['label', 'rec_id'])

rec_df1 = user_rec(1, ranking_2, user1_rec_data, train_df2)
print('user1_rec_data')
print(rec_df1)

# ---user2
# user2_rec_data = pd.DataFrame(columns=['label', 'rec_id', 'T/F'])
user2_rec_data = pd.DataFrame(columns=['label', 'rec_id'])

rec_df2 = user_rec(2, ranking_2, user2_rec_data, train_df2)
print('user2_rec_data')
print(rec_df2)

# ---user3
user3_rec_data = pd.DataFrame(columns=['label', 'rec_id'])

rec_df3 = user_rec(3, ranking_2, user3_rec_data, train_df2)
print('user3_rec_data')
print(rec_df3)

# ---user4
user4_rec_data = pd.DataFrame(columns=['label', 'rec_id'])

rec_df4 = user_rec(4, ranking_2, user4_rec_data, train_df2)
print('user4_rec_data')
print(rec_df4)

# ---user5
user5_rec_data = pd.DataFrame(columns=['label', 'rec_id'])

rec_df5 = user_rec(5, ranking_2, user5_rec_data, train_df2)
print('user5_rec_data')
print(rec_df5)
```

![ ](images/Ecommerce_Rec/Untitled%2074.png)

```python
# 각 유저별 추천목록의 상품들 중 true-like한 것이 있는지 T/F(trud:1/false:0)으로 데이터프레임에 넣어주는 기능.

def true_like(n, rec_df, true_df):
  # 전달인자(argument)설명: 
   ## n: 몇번째 유저인지 (the order of test-user starting from 0)
   ## rec_df: 유저당 추천된 라벨과 라벨에 해당하는 상품 아이디 데이터프레임 (recommended label and product_id for each user)
   ## true_df: 랭킹모델의 예측에 사용된 test_data중에서 필요한 'user_id, true_like_id'특성만 추출한 데이터프레임(test_data's 'user_id and true_like_id'features used for ranking_model predicting)
  tf_list = []
  rec_id = rec_df['rec_id'].tolist() 
  # rec_id = rec_df['rec_id'].values 

  # true데이터인 test_data의 형태상, row에 있는 유저 n-1번째 row를 고정해야 함. 
  true = true_df['true_like_id'].tolist()[n-1]

  for rec in rec_id[:][:]:
    if len(rec) == 0:
      tf_list.append(0.0)

    elif len(rec)>=1:        
      tf = true[0] in rec[0]
      if tf == True:
        tf_list.append(1)
      else:
        tf_list.append(0)
      rec_df['T/F'] = pd.DataFrame(tf_list)

  print("true_like_id:", true)

  return rec_df

# ---user1의 추천목록에 대한 T/F(true-like or false) 결과 
rec_df1 = true_like(1, rec_df1, true_df2)
print(rec_df1)

# ---user2의 추천목록에 대한 T/F(true-like or false) 결과 
rec_df2 = true_like(2, rec_df2, true_df2)
print(rec_df2)

# ---user3의 추천목록에 대한 T/F(true-like or false) 결과 
rec_df3 = true_like(3, rec_df3, true_df2)
print(rec_df3)

# ---user4의 추천목록에 대한 T/F(true-like or false) 결과 
rec_df4 = true_like(4, rec_df4, true_df2)
print(rec_df4)

# ---user5의 추천목록에 대한 T/F(true-like or false) 결과 
rec_df5 = true_like(5, rec_df5, true_df2)
print(rec_df5)
```

![ ](images/Ecommerce_Rec/Untitled%2075.png)

```python
# 각 유저별 dcg, idcg, ndcg 계산하는 기능

import numpy as np
def get_ndcg(rec_df):
  # rec_df: 유저당 추천된 '라벨'과 '라벨에 해당하는 상품 아이디', 'T/F(true_like여부)'특성이 포함된 데이터프레임 (dataframe about recommended 'label' and 'product_id', 'T/F(true_like or not)' for each user)
  rec = rec_df['rec_id'].tolist()
  t = rec_df['T/F'].tolist()
  dcg = 0.0
  
  #dcg 계산
  for i, j in enumerate(t): # i, j: T/F안의 인덱스와 값들(1.0, 0.0..)을 돈다.
    if j == 1.0:
      dcg += (1.0/np.log2(i+1+1))
    else: 
      dcg += 0
  
  #idcg 계산
  idcg = sum((1.0/np.log2(i+1+1) for i in range(0, len(t)+1)))

  #ndcg 계산
  ndcg = dcg / idcg 
  
  # set 으로 결과 산출할 경우,
  # return dcg, idcg, ndcg
  
  #dataframe으로 결과 산출할 경우,
  ndcg_df = pd.DataFrame(columns=['dcg', 'idcg', 'ndcg'])
  ndcg_df = ndcg_df.append(pd.DataFrame([[dcg, idcg, ndcg]], columns=['dcg', 'idcg', 'ndcg']), ignore_index=True)

  return ndcg_df

# --- user1의 dcg, idcg, ndcg 
ndcg_df1 = get_ndcg(rec_df1)
print(ndcg_df1)

# --- user2의 dcg, idcg, ndcg 
ndcg_df2 = get_ndcg(rec_df2)
print(ndcg_df2)

# --- user3의 dcg, idcg, ndcg 
ndcg_df3 = get_ndcg(rec_df3)
print(ndcg_df3)

# --- user4의 dcg, idcg, ndcg 
ndcg_df4 = get_ndcg(rec_df4)
print(ndcg_df4)

# --- user5의 dcg, idcg, ndcg 
ndcg_df5 = get_ndcg(rec_df5)
print(ndcg_df5)
```

![ ](images/Ecommerce_Rec/Untitled%2076.png)

```python
# user별 dcg, idcg, ndcg 결과를 합쳐주는 함수 기능. 
import pandas as pd
def concat_result(df1, df2):
  df3 = pd.concat([df1, df2])
  
  return df3

result = concat_result(ndcg_df1, ndcg_df2)
result = concat_result(result, ndcg_df3)
result = concat_result(result, ndcg_df4)
result = concat_result(result, ndcg_df5)
result.index=['test-user1', 'test-user2', 'test-user3', 'test-user4', 'test-user5']

print('[ ranking_2안의 유저별 dcg, idcg, ndcg ] \n ', result)
print('-------------------\n')
print('[ ranking_2안의 dcg, idcg, ndcg평균 ] \n', result.mean())
```

![ ](images/Ecommerce_Rec/Untitled%2077.png)

# 결론 및 향후 보완할 사항

(1) ranking_2안의 전체 평균 결과는 nDCG 스코어에서 0.1 높음.

- ranking_1안 ndcg 평균: 0.28
- ranking_2안 ndcg 평균: 0.38

(2) 유저별로 nDCG스코어가 ranking_1안이 높은 경우가 있고, ranking_2안이 높은 경우가 있으므로, 개인별 맞춤 제안을 하면 좋을 것, 특히 0이 나오는 경우도 있어서 이 경우는 신경 써서 제안을 해야 함.

- user2: 0.37 (ranking1_ndcg) < 0.57 (ranking2_ndcg)
- user1: 0.52 (ranking1_ndcg) < 0.00 (ranking2_ndcg)

```
[ ranking_1안의 유저별 dcg, idcg, ndcg ]
                   dcg      idcg      ndcg
test-user1  2.061606  3.953465  0.521468
test-user2  1.487137  3.953465  0.376160
test-user3  1.487137  3.953465  0.376160
test-user4  0.500000  3.953465  0.126471
test-user5  0.000000  3.953465  0.000000
-------------------

[ ranking_1안의 dcg, idcg, ndcg평균 ]
 dcg     1.107176
idcg    3.953465
ndcg    0.280052

[ ranking_2안의 유저별 dcg, idcg, ndcg ]
                   dcg      idcg      ndcg
test-user1  0.000000  3.953465  0.000000
test-user2  2.264010  3.953465  0.572665
test-user3  1.500000  3.953465  0.379414
test-user4  1.500000  3.953465  0.379414
test-user5  2.351116  3.953465  0.594698
-------------------

[ ranking_2안의 dcg, idcg, ndcg평균 ]
 dcg     1.523025
idcg    3.953465
ndcg    0.385238

```

(3) ranking 1,2 안의 차이점:

- (ranking_1안) reorder(재주문 여부)로 rating을 매긴 개념으로, 재주문을 한 경우(1)라면 **like**, 재주문을 안 한 경우(0)라면 **dislike**로 상품을 구분한 feature를 랭킹모델에 추가하여 추천목록을 생성한 방법.
- (ranking_2안) order_number(주문회차: 몇번째 구매인지)로 rating을 매긴 개념으로, 평균 주문회차인 18 이상인 경우(1) **like**, 18 미만(0)이면 **dislike**로 상품을 구분한 feature를 랭킹모델에 추가하여 추천목록을 생성한 방법.

---

- Reference
    - [https://jalynne-kim.medium.com/추천모델-이커머스-추천모델링-딥러닝모델-프로젝트-회고-d5017cb1335f](https://jalynne-kim.medium.com/%EC%B6%94%EC%B2%9C%EB%AA%A8%EB%8D%B8-%EC%9D%B4%EC%BB%A4%EB%A8%B8%EC%8A%A4-%EC%B6%94%EC%B2%9C%EB%AA%A8%EB%8D%B8%EB%A7%81-%EB%94%A5%EB%9F%AC%EB%8B%9D%EB%AA%A8%EB%8D%B8-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%ED%9A%8C%EA%B3%A0-d5017cb1335f)