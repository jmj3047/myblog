---
title: Pyspark Tutorial(3)
date: 2022-04-21
categories:
  - Python 
  - Pyspark
tags: pyspark
---

# Machine Learning

## 01.regression.py

```python
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

print("Starting Session")

#세션 할당
spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

#데이터 불러오기
#StructType 과정 생략 가능
data = spark.read.option("header", "true").option("inferSchema", "true").csv("data/realestate.csv")

#print(data.show())

#데이터 프레임을 행렬로 변환
assembler = VectorAssembler().setInputCols(["HouseAge", "DistanceToMRT","NumberConvenienceStores"]).setOutputCol("features") #데이터 컬럼 값 아무거나 넣어도 됨

#타겟 데이터 설정
df = assembler.transform(data).select("PriceofUnitArea","features")

#데이터 분리
trainTest = df.randomSplit([0.5,0.5])
trainingDF = trainTest[0]
testDF = trainTest[1]

#Decision Tree 클래스 정의
dtr = DecisionTreeRegressor().setFeaturesCol("features").setLabelCol("PriceofUnitArea")

#모델 학습
model = dtr.fit(trainingDF)
#print(model)

#모델 예측
fullPredictions = model.transform(testDF).cache()

#예측값과 label확인
predictions = fullPredictions.select("prediction").rdd.map(lambda x: x[0])

#실제데이터
labels = fullPredictions.select("PriceofUnitArea").rdd.map(lambda x: x[0])

#예측값과 label을 zip으로 묶어줌
preds_label = predictions.zip(labels).collect()

for prediction in preds_label:
    print(prediction)

#세션 종료
spark.stop()
```

---

## 02.logistic_regression.py

```python
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression #Important


# 세션 할당
spark = SparkSession.builder.appName("AppName").getOrCreate()

# load Data
training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
print("Data loaded")

# model
# Scikit-Learn 문법과 비슷
mlr = LogisticRegression() # Important
mlr_model = mlr.fit(training) # Important

# 로지스텍 회귀, 선형 모델 .. 계수와 상수를 뽑아낼 수 있음
print("Coefficients :" + str(mlr_model.coefficients))
print("Intercept :" + str(mlr_model.intercept))


spark.stop()

```

---

## 03.pipeline.py

```python
from tokenize import Token
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

from pyspark.sql import SparkSession

# 세션 할당 
spark = SparkSession.builder.appName("MLPipeline").getOrCreate()

# 가상의 데이터 만들기
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

# Feature Engineering
# 1. Prparation
# step01. 텍스트를 단어로 분리
tokenizer = Tokenizer(inputCol='text', outputCol='words')

# step02. 변환된 텍스트를 숫자로 변환
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")

# step03. 모델을 가져옴
lr = LogisticRegression(maxIter=5, regParam=0.01)

# 2. Starting pipepline
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# 3. Model Training
model = pipeline.fit(training)

# 4. Prepare test documents, which are unlabeled (id, text) tuples.
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

# 5. Prediction
prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    row_id, text, prob, prediction = row #튜플 형태로 반환
    print(
        # 문자열 포맷팅
        "(%d, %s) -------> probability=%s, prediction=%f" % (row_id, text, str(prob), prediction)
    )

# training.show()

# 세션 종료
spark.stop()
```

---

## 04.randomforest.py

```python
from cProfile import label
from pyspark.sql import SparkSession

# 머신러닝 라이브러리
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 데이터 불러오기 
spark = SparkSession.builder.appName("RandomForest").getOrCreate()

data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
print(type(data))


# Feature Engineering
# label column 
labelIndexer = StringIndexer(inputCol='label', outputCol='indexedLabel').fit(data)

# 범주형 데이터 체크, 인덱스화
featureIndexer = VectorIndexer(inputCol='features', 
                               outputCol='IndexedFeatures', maxCategories=4).fit(data)

# 데이터 분리
(trainingData, testData) = data.randomSplit([0.7, 0.3])


# 모델 
rf = RandomForestClassifier(labelCol='indexedLabel', # 종속변수
                            featuresCol='IndexedFeatures', # 독립변수
                            numTrees=10)

# outputCol='indexedLabel' --> original label로 변환
labelConvereter = IndexToString(inputCol='prediction', 
                                outputCol='predictedLabel', labels=labelIndexer.labels)

# 파이프라인 구축
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConvereter])

# 모델 학습
model = pipeline.fit(trainingData)

# 모델 예측
predictions = model.transform(testData)

# 행에 표시할 것 추출  
predictions.select("predictedLabel", 'label', 'features').show(5)


# 모형 평가
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print("Test Error = %f " % (1.0 - accuracy))


spark.stop()
```