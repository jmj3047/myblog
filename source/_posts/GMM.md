---
title: Gaussian Mixture Model
date: 2022-09-10
categories:
  - Data Analysis
  - Model 
tags: 
  - Gaussian Mixture Model
  - Clustering
  - ML Analysis
  - Python
  - English
---

## 1. What is GMM
- It is one of several models applying the Expectation Maximum (EM) algorithm.
- What is EM algorithm?
    - EM algorithm is basically an algorithm mainly used for Unsupervised learning.
    - It is also used in clustering.
    - The EM algorithm can be largely divided into two stages: E-step and M-step. In conclusion, it is a flow that finds the optimal parameter value by repeating E-step and M-step.
        - E-step: Calculate the likelihood value that is as close as possible to the likelihood of the initial value of a given arbitrary parameter.
        - M-step: Obtained a new parameter value that maximizes the likelihood calculated in the E-step
        - The above two steps are continuously repeated until the parameter value does not change significantly.
    - The purpose of the EM algorithm is to find parameters that maximize the likelihood.
    - Maximum Likelihood Estimation (MLE) is a Convex or Convex function whose objective function can be differentiated (which can obtain a global optimum) because the optimal parameter must be obtained directly by partial differentiation of the parameter variable.
    - However, EM is a process of finding a value close to the optimal parameter while repeating the E-M step. Similar to the ANN (artificial neural network) model, it does not guarantee to find the Global Optimum, and even if it does, it does not recognize that the point is the real Global Optimum. Therefore, it is very likely that the local minimum is found and the objective function is not composed of Convex or Concave functions.
    - The EM algorithm defines the function of the parameters to be obtained (ex. probability distribution) as a probability variable and optimizes it.
- It is also used in clustering models and is mainly used in speech recognition modeling.
- This is a [post](https://brunch.co.kr/@kakao-it/105) that Kakao, which develops voice recognition technology using GMM algorithms.
- Clustering is performed on the assumption that data are generated by datasets with multiple normal distributions.
- Several normal distribution curves are extracted and individual data are determined to which normal distribution belongs.
- This process is called parameter estimation in GMM, and two typical estimates are made and the EM method is applied for this parameter estimation.
    - Means and variance of individual normal distributions
    - Probability of which normal distribution each data corresponds

## 2. Code

```python
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

mpl.rc('font', family='NanumGothic') # 폰트 설정
mpl.rc('axes', unicode_minus=False) # 유니코드에서 음수 부호 설정

# 차트 스타일 설정
sns.set(font="NanumGothic", rc={"axes.unicode_minus":False}, style='darkgrid')
plt.rc("figure", figsize=(10,8))

warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris

iris = load_iris()
feature_names = ['sepal_length','sepal_width','petal_length','petal_width']

iris_df = pd.DataFrame(iris.data, columns = feature_names)
iris_df["target"] = iris.target

iris_df.head()
```

```python
from sklearn.mixture import GaussianMixture

# GMM: n_components = 모델의 총 수
gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

# target, gmm_cluster 비교
iris_df["gmm_cluster"] = gmm_cluster_labels

iris_df.groupby(["target","gmm_cluster"]).size()
```

```
결과
target  gmm_cluster
0       0              50
1       1              45
        2               5
2       2              50
dtype: int64
```

- GMM can be performed using `GaussianMixture()` in `sklearn.mixture()`
- n_components is the total number of models, and like n_clusters in K-Means, determines the number of clusters.
- Here, when the target of gmm_cluster was 1, only five were mapped differently, and all the rest were well mapped.
---

- [Korean reference1](https://techblog-history-younghunjo1.tistory.com/88)
- [Korean reference2](https://romg2.github.io/mlguide/19_%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%99%84%EB%B2%BD%EA%B0%80%EC%9D%B4%EB%93%9C-07.-%EA%B5%B0%EC%A7%91%ED%99%94-GMM/)