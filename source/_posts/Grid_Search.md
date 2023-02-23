---
title: Grid Search CV
date: 2022-09-16
categories:
  - Data Analysis
  - Basic 
tags: 
  - Grid Search CV
  - English
  - ML Analysis
  - Python
---


- Grid search finds the optimal parameters; each model has its own parameters, and it compares which combination yields the best score.
- This time, we will see a combination of two parameters and use decision tree.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

#훈련데이터 검증데이터 분류
iris = load_iris()

data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=999)

```

- Bring the iris data
- Data and target are classified into training data and validation data, and the ratio of validation data is set to 20%.
- Random_state = 999 means that you will continue to use random data that you have picked once. It does not matter if it is not 999 or any other number.

```python

#그리드서치
dtree = DecisionTreeClassifier()

grid_parameters = {"max_depth": [1, 2, 3],
                   "min_samples_split": [2, 3]
                   }

grid_dtree = GridSearchCV(dtree, param_grid=grid_parameters, cv=3, refit=True)

grid_dtree.fit(X_train, y_train)

```

- Create a decision tree model, save the value of the desired parameter in dictionary form.
- Cv is the number of sets of training data divided for cross-validation.
- Refit = True means finding the optimal parameter and then learning based on it.
- It is added another step before fitting

```python
#결과를 데이터 프레임으로 변환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df)

출력
mean_fit_time  std_fit_time  mean_score_time  std_score_time param_max_depth param_min_samples_split                                    params  split0_test_score  split1_test_score  split2_test_score  mean_test_score  std_test_score  rank_test_score
0       0.001321  4.587829e-04         0.000997    3.893359e-07               1                       2  {'max_depth': 1, 'min_samples_split': 2}              0.675              0.675              0.700         0.683333        0.011785                5
1       0.001676  4.799690e-04         0.001196    2.498432e-04               1                       3  {'max_depth': 1, 'min_samples_split': 3}              0.675              0.675              0.700         0.683333        0.011785                5
2       0.002052  8.137361e-05         0.000613    4.378794e-04               2                       2  {'max_depth': 2, 'min_samples_split': 2}              0.925              0.925              0.975         0.941667        0.023570                3
3       0.000998  8.104673e-07         0.000332    4.694597e-04               2                       3  {'max_depth': 2, 'min_samples_split': 3}              0.925              0.925              0.975         0.941667        0.023570                3
4       0.001271  2.113806e-04         0.000563    4.170978e-04               3                       2  {'max_depth': 3, 'min_samples_split': 2}              0.975              0.950              0.975         0.966667        0.011785                1
5       0.001044  4.221438e-05         0.000665    4.699172e-04               3                       3  {'max_depth': 3, 'min_samples_split': 3}              0.975              0.950              0.975         0.966667        0.011785                1

```

- To see the results fitted above, use cv_results_.
- Then there are a number of indicators.

```python
#최적의 파라미터 출력
scores_df = scores_df[["params", "mean_test_score", "rank_test_score", "split0_test_score", "split1_test_score", "split2_test_score"]]
print(scores_df)

# 최고의 파라미터 저장해줌
print(f"최적의 파라미터: {grid_dtree.best_params_}")
print(f"최고 정확도: {grid_dtree.best_score_}")

출력
                                    params  mean_test_score  rank_test_score  split0_test_score  split1_test_score  split2_test_score
0  {'max_depth': 1, 'min_samples_split': 2}         0.683333                5              0.675              0.675              0.700
1  {'max_depth': 1, 'min_samples_split': 3}         0.683333                5              0.675              0.675              0.700
2  {'max_depth': 2, 'min_samples_split': 2}         0.941667                3              0.925              0.925              0.975
3  {'max_depth': 2, 'min_samples_split': 3}         0.941667                3              0.925              0.925              0.975
4  {'max_depth': 3, 'min_samples_split': 2}         0.966667                1              0.975              0.950              0.975
5  {'max_depth': 3, 'min_samples_split': 3}         0.966667                1              0.975              0.950              0.975
최적의 파라미터: {'max_depth': 3, 'min_samples_split': 2}
최고 정확도: 0.9666666666666667
```

- Print out the optimal parameters which is max_depth:3, min_samples_split:2 and the score is 0.96
- In this case, the function that gives the best parameter and the highest score is best_params_, best_score_.

---
- [Korean reference](https://wpaud16.tistory.com/65)