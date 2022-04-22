---
title: Pyspark Tutorial(1)
date: 2022-04-21
categories:
  - Python 
  - Pyspark
tags: pyspark
---


Reference: [https://spark.apache.org/docs/latest/quick-start.html](https://spark.apache.org/docs/latest/quick-start.html)


# Get Started

## 01.basic.py

```python
# -*- coding: utf-8 -*-

import pyspark
print(pyspark.__version__)

from pyspark.sql import SparkSession

#스파크 세션 초기화 :spark session이 하나 만들어진것
spark = SparkSession.builder.master('local[1]').appName('SampleTutorial').getOrCreate()
rdd = spark.sparkContext.parallelize([1,2,3,4,5])

print("rdd Count", rdd.count())

spark.stop()
```

---

## 02.rating.py

```python
#SparkContext
#RDD

from pyspark import SparkConf, SparkContext
import collections

print("Hello")

def main():
    # MasterNode = local
    # MapReduce
    
    conf = SparkConf().setMaster("local").setAppName('RatingHistogram')
    sc = SparkContext(conf = conf)
    
    lines = sc.textFile("ml-100k/u.logs")
    #print(lines)
    ratings = lines.map(lambda x: x.split()[2])
    #print("ratings:",ratings) #rdd라는 객체가 만들어진것
    
    result = ratings.countByValue()
    #print("result:",result)
    
    #정렬하기
    sortedResults = collections.OrderedDict(sorted(result.items()))
    for key, value in sortedResults.items():
        print("%s %i" % (key, value))
    
if __name__ == "__main__":
    main()
    
##spark를 쓰는 이유:로그 데이터를 가져와서 규칙을 대입해서 정렬한다음에 정형데이터로 치환하기위해
#실제로 의미있는 로그라면 분석도 의미가 있다
#분석과 로그데이터를 처리할 수 있는 환경을 지원해줌
#과거에는 두개가 따로 있었음
```

---

## 03.dataloading.py

```python
#Spark SQL 적용

#Spark Session
from pyspark.sql import SparkSession

# #스파크 세션 생성
# my_spark = SparkSession.builder.getOrCreate()
# print(my_spark)

# #테이블을 확인하는 코드
# print(my_spark.catalog.listDatabases())

# #show database
# my_spark.sql("show databases").show()

# #check current DB
# my_spark.catalog.currentDatabase()
# my_spark.stop()

#loading csv file
spark = SparkSession.builder.master('local[1]').appName("DBTutorial").getOrCreate()
flights = spark.read.option('header','true').csv('data/flight_small.csv')
#flights.show(4)

#spark.catalog.currentDatabase()
#flights 테이블을 default DB에 추가함
flights.createOrReplaceTempView('flights')

#print(spark.catalog.listTables('default'))

#spark.sql('show tables from default').show()

#쿼리 통해서 데이터 저장
query = "FROM flights SELECT * LIMIT 10"
query2 = "SELECT * FROM flights LIMIT 10"

# 스파크 세션 할당
flights10 = spark.sql(query2)
#flights10.show()

#spark 데이터 프레임을 pandas data frame으로 변환
import pandas as pd
pd_flights10 = flights10.toPandas()
print(pd_flights10.head())
```

---

## 04.struct_type.py

- Reference: [https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.types.StructType.html](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.types.StructType.html)

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func #alias
from pyspark.sql.types import StructType, StructField, IntegerType, LongType

print("Hello")

#세션 할당
spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

#스키마 작성(u.logs 데이터)
schema = StructType(
    [
        StructField("userID",IntegerType(), True)
        , StructField("movieID",IntegerType(), True)
        , StructField("rating", IntegerType(), True)
        , StructField("timestamp", LongType(), True)
    ]
)

print("Schema is done")

#데이터 불러오기
movies_df = spark.read.option("sep","\t").schema(schema).csv("ml-100k/u.logs")

#내림차순으로 인기 있는 영화 정렬
#movieID로 그룹바이, count() 진행, orderby
topMovieIds = movies_df.groupby("movieID").count().orderBy(func.desc('count'))

print(topMovieIds.show(10))

#세션 종료
spark.stop()
```

---

## 05.advance_structtype.py

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as func #alias
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
import codecs

print("Starting Session")

def loadMovieNames(): #u.item에서 영화 이름 가져옴
    movieNames = {}
    with codecs.open("ml-100k/u.item","r", encoding="ISO-8859-1", errors ="ignore") as f:
        for line in f:
            fields = line.split("|")
            movieNames[int(fields[0])] = fields[1] #데이터 추가하는 딕셔너리 문법
    
    return movieNames
            
#세션 할당
spark = SparkSession.builder.appName("PopularMovies").getOrCreate()

#파이썬 딕셔너리 객체를 spark 객체로 변환
nameDict = spark.sparkContext.broadcast(loadMovieNames())

#스키마 작성(u.logs 데이터)
schema = StructType(
    [
        StructField("userID",IntegerType(), True)
        , StructField("movieID",IntegerType(), True)
        , StructField("rating", IntegerType(), True)
        , StructField("timestamp", LongType(), True)
    ]
)

print("Schema is done")

#데이터 불러오기
movies_df = spark.read.option("sep","\t").schema(schema).csv("ml-100k/u.logs")

#내림차순으로 인기 있는 영화 정렬할 필요 없음
#movieID로 그룹바이, count() 진행, orderby
#topMovieIds = movies_df.groupby("movieID").count().orderBy(func.desc('count'))
topMovieIds = movies_df.groupby("movieID").count()

# 딕셔너리 
# key-value
# 키값을 알면 value자동으로 가져옴(movietitle)
def lookupName(movieID):
    return nameDict.value[movieID]

# 사용자 정의 함수 사용할 때 쓰는 spark 문법
lookupNameUDF = func.udf(lookupName)

# MovieTitle을 기존 topMovieIds 데이터에 추가
#컬럼 추가
moviesWithNames = topMovieIds.withColumn("movietitle",lookupNameUDF(func.col("movieID")))

#정렬
final_df = moviesWithNames.orderBy(func.desc("count"))

print(final_df.show(10))

#세션 종료
spark.stop()
```