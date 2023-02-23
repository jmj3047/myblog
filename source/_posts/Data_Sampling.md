---
title: Data Sampling
date: 2022-09-07
categories:
  - Data Analysis
  - Basic
tags: 
  - Probabilistic sampling
  - Nonprobability sampling
  - English
---

## 1. Reason why you need
- The more input data you have on machine learning, the slower the processing.
- Therefore, in order to speed up the processing speed of machine learning, acceleration of learning speed of data would be helpful, which can be done with optimization of machine learning with representative data.
- Then let's see how we can reduce the data so that we can only use the data we need.

## 2. What is data sampling
- The process of organizing the data and making it the best input data.
- For example, it can speed up the processing of machine learning by using sales of 'month' rather than using sales of 'day' units to analyze last year's profits of a pizza house.
- This is the work of making optiml data.
- There are probabilistic sampling, nonprobability sampling in data sampling method.
- The probabilistic sampling method is a sampling method based on statistics, and the nonprobability sampling is a sampling method in which the subjectivity of a person is involved.
- Depending on each sampling method, you should select and use the sampling method that matches the data and the situation because there are advantages and disadvantages.

## 3. Probabilistic sampling
> Probabilistic sampling is a random sampling method that can be divided into simple random sampling, two-step sampling, stratified sampling, cluster/collective sampling, and system sampling.
- Simple Random Sampling: A method of randomly extracting samples from the entire data.
- Two-Step Sampling: A sampling method that separates the entire n data into m subpopulations. Selects m subpopulations and provides simple random sampling of N data from m subpopulations. It is an accurate sampling method rather than simple random sampling.
- Stratified Sampling: By separating the population from several layers, it is a method of randomly extracting data from each layer by n. For example, it is a method of dividing the layers of Korean cities and province and extracting n data from each layer.
- cluster/collective sampling: If the population is composed of multiple clusters, it is a method of selecting one or several clusters and using the entire data of the selected clusters. For example, it is a way to use all of the data by setting the Korea as it’s city and province.
- system sampling: It is a method of extracting data one by one at a regular interval by numbering all data from 1 to n. This method is mainly used to sample representative values of the time series data

## 4. Nonprobability sampling
> This is a method of subjectively extracting the probability of being selected as a sample in advance. The advantage and disadvantage of this sampling is that the subjective intention of the sampling person is involved. The implicit population extracted by nonprobability sampling is a good sampling if it matches the ideal population, the most suitable population for the subject. Nonprobability sampling methods include convenience sampling, purpose sampling, and quota sampling methods.
- Convenience Sampling: A method of sampling by selecting a point or location where data is good for collecting. The sample surveyed by this sampling method has a disadvantage that it is less representative than the population. It’s not possible to go through the statistical inference process. Statistical inference is that we generalize the sample analysis results to speculation about populations.
- Purpose Sampling: This is how you select the object you think is the most suitable object for your purpose; you will sample the data that is subjectively appropriate for your purpose.The downside is that it has also low representative for the population.
- Quota Sampling: Divide the population into segments, assigning each segment a quartar that represents the number of samples. Within segments, the characteristics related to the topic must be similar, and the populations must be distributed differently between segments. This is methodologically similar to layer-by-layer sampling. But the difference is that the sample is not selected by probability but by subjective judgment.

## 5. Conclusion
- Comparing probabilistic sampling with nonprobability sampling, probabilistic sampling may look good when judged by just words. However, stochastic sampling is advantageous for data that can be analyzed based on statistics, and non-probability sampling is advantageous for data such as language and music. It is better to choose and use probabilistic and nonprobable sampling as appropriate.
---
- [Korean Reference](https://muzukphysics.tistory.com/entry/ML-5-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%83%98%ED%94%8C%EB%A7%81-%EB%B0%A9%EB%B2%95%EA%B3%BC-%ED%95%84%EC%9A%94%EC%84%B1-%ED%99%95%EB%A5%A0%EC%A0%81-%EB%B9%84%ED%99%95%EB%A5%A0%EC%A0%81-%EC%83%98%ED%94%8C%EB%A7%81)