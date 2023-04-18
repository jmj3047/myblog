---
title: Supervised Machine Learning, Regression and Classification
date: 2023-03-26
categories:
  - Data Analysis
  - Model
tags: 
  - ML Analysis
  - Coursera
  - AI Studies
  - English
---

## Course
- [Lecture 1](https://www.coursera.org/learn/machine-learning) in [Machine Learning Course](https://www.coursera.org/specializations/machine-learning-introduction)


## Regression with multiple input variables
**1. vectorization**
- Many CPUs have "vector" or "SIMD" instruction sets which apply the same operation simultaneously to two, four, or more pieces of data. Modern x86 chips have the SSE instructions, many PPC chips have the "Altivec" instructions, and even some ARM chips have a vector instruction set, called NEON.
- "Vectorization" (simplified) is the process of rewriting a loop so that instead of processing a single element of an array N times, it processes (say) 4 elements of the array simultaneously N/4 times.
- I chose 4 because it's what modern hardware is most likely to directly support for 32-bit floats or ints.

---

- **The difference between vectorization and loop unrolling:**
 Consider the following very simple loop that adds the elements of two arrays and stores the results to a third array.
    
    ```python
    for (int i=0; i<16; ++i)
        C[i] = A[i] + B[i];
    ```
    
- Unrolling this loop would transform it into something like this:
    
    ```python
    for (int i=0; i<16; i+=4) {
        C[i]   = A[i]   + B[i];
        C[i+1] = A[i+1] + B[i+1];
        C[i+2] = A[i+2] + B[i+2];
        C[i+3] = A[i+3] + B[i+3];
    }
    ```
    
- Vectorizing it, on the other hand, produces something like this:
    
    ```python
    for (int i=0; i<16; i+=4)
        addFourThingsAtOnceAndStoreResult(&C[i], &A[i], &B[i]);
    ```
    
- Where "addFourThingsAtOnceAndStoreResult" is a placeholder for whatever intrinsic(s) your compiler uses to specify vector instructions.

**2. feature scaling**
- Feature scaling is a method used to normalize the range of independent variables or features of data. It is generally performed during the data preprocessing step. It is important because it can help machine learning algorithms work properly and converge faster. There are several methods for feature scaling, including rescaling (min-max normalization), mean normalization, and standardization (Z-score normalization)
    1. **Rescaling (min-max normalization)**: This method rescales the range of features to a new range, such as [0, 1] or [-1, 1]. The general formula for a min-max of [0, 1] is given as: **`x' = (x - min) / (max - min)`** where x is an original value and x’ is the normalized value.
    2. **Mean normalization**: This method involves subtracting the mean of a feature from each data point and then dividing by the range (max - min) or standard deviation of that feature.
    3. **Standardization (Z-score normalization)**: This method involves subtracting the mean of a feature from each data point and then dividing by the standard deviation of that feature. This results in each feature having a mean of 0 and a standard deviation of 1.

**3. feature engineering(audio data)**
- Feature engineering in audio data involves using domain knowledge to extract relevant features from raw audio data.
- This can include techniques such as converting audio files into spectrograms, which produce a high-dimensional space of data that can be further reduced by applying a convolutional neural network model.
    1. **Spectrogram generation**: This involves converting audio files into spectrograms, which produce a high-dimensional space of data that can be further reduced by applying a convolutional neural network model.
    2. **Time series analysis**: This involves analyzing audio data as a set of time series and extracting relevant features from it.
    3. **Sound engineering**: This involves using domain knowledge from the field of sound engineering to extract relevant features from audio data.

**4. polynomial regression(다항 회귀)**
- Polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial in x.
- Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E(y|x).
- Although polynomial regression fits a nonlinear model to the data, as a statistical estimation problem it is linear, in the sense that the regression function E(y|x) is linear in the unknown parameters that are estimated from the data.
- example
    
    - Suppose we have a dataset with the following x and y values: `x = [1, 2, 3, 4, 5] y = [2, 6, 12, 20, 30]`
    
    - If we plot these points on a graph, we can see that the relationship between x and y is not linear. However, we can use polynomial regression to fit a curve to these points.
    
    - First, we need to decide on the degree of the polynomial. In this case, let’s use a second-degree polynomial (a quadratic equation). This means our model will have the form: `y = a * x^2 + b * x + c`, where a, b, and c are coefficients that we need to estimate from the data.
    
    - We can use the **`polyfit`** function from the **`numpy`** library to estimate these coefficients:
    
    ```python
    import numpy as np
    
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 6, 12, 20, 30])
    
    coefficients = np.polyfit(x, y, deg=2)
    ```
    
    - This will return an array of coefficients **`[a, b, c]`**. In this case, the coefficients are **`[1.0, -1.7763568394002505e-15, 1.0]`**.
    
    - Now that we have the coefficients of our polynomial model, we can use it to make predictions for new x values. For example:
    
    ```python
    x_new = 6
    y_new = coefficients[0] * x_new**2 + coefficients[1] * x_new + coefficients[2]
    ```
    
    - This will give us a predicted y value of **`42`**for **`x=6`**
            
## Classification 
1. Classification and Logistic Regression
2. Cost function 
3. Gradient descent 
4. Overfitting

---

- Reference
    - [https://stackoverflow.com/questions/1422149/what-is-vectorization](https://stackoverflow.com/questions/1422149/what-is-vectorization)
    - [https://en.wikipedia.org/wiki/Feature_scaling](https://en.wikipedia.org/wiki/Feature_scaling)
    - [https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html)
    - [https://towardsdatascience.com/what-is-feature-scaling-why-is-it-important-in-machine-learning-2854ae877048](https://towardsdatascience.com/what-is-feature-scaling-why-is-it-important-in-machine-learning-2854ae877048)
    - [https://bkshin.tistory.com/entry/머신러닝-8-Feature-Scaling-Feature-Selection](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-8-Feature-Scaling-Feature-Selection)
    - [https://link.springer.com/chapter/10.1007/978-1-4842-8925-9_9](https://link.springer.com/chapter/10.1007/978-1-4842-8925-9_9)
    - [https://en.wikipedia.org/wiki/Polynomial_regression](https://en.wikipedia.org/wiki/Polynomial_regression)