---
title: Pyspark Tutorial(2)
date: 2022-04-21
categories:
  - Python 
  - Pyspark
tags: pyspark
---

# Data cleansing

## 01.pipeline.py

```python
from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.sql import functions as F

#Create Spark Session
spark = SparkSession.builder.master("local[1]").appName("MLSampleTutorial").getOrCreate()

#Load Data
df = spark.read.csv("data/AA_DFW_2015_Departures_Short.csv.gz", header = True)

print("file loaded")

print(df.show())

#remove duration = 0
df = df.filter(df[3] > 0) #Actual elapsed time (Minutes) 여기 컬럼 값이 0보다 작은건 보여주지 않음
# df.show()

#ADD ID column
df = df.withColumn('id',F.monotonically_increasing_id()) #id값을 자동으로 넣어줌

df.write.csv("data/output.csv", mode = 'overwrite')

spark.stop()
```

---

## 02.total_spent.py

```python
# #라이브러리 불러오기
# from pyspark import SparkConf, SparkContext
# #사용자 정의 함수

# #main함수
# def main():
#     conf = SparkConf.setMaster("local").setAppName('SpentbyCustomer')
#     sc = SparkContext(conf= conf)
#      # 파이썬 코드
# # 실행코드 작성
# if __name__ == "__main__":
#    main()
########## 이게 spark 기본 세팅 #########

#라이브러리 불러오기
from pyspark import SparkConf, SparkContext

#사용자 정의 함수
def extractCusPrice(line):
    fields = line.split(",")
    return(int(fields[0]), float(fields[2]))

#main함수
def main():
    
    #스파크 설정
    conf = SparkConf().setMaster("local").setAppName('SpentbyCustomer')
    sc = SparkContext(conf= conf)

    #데이터 불러오기
    input = sc.textFile('data/customer-orders.csv')
    #print('is data?')
    mappedInput = input.map(extractCusPrice) #튜플 형태로 나옴
    totalByCustomer = mappedInput.reduceByKey(lambda x, y : x + y)

    #정렬
    flipped = totalByCustomer.map(lambda x: (x[1], x[0]))
    totalByCustomerSorted = flipped.sortByKey()
    
    results = totalByCustomerSorted.collect()
    for result in results:
        print(result)
        
    #파이썬 코드
    
# 실행코드 작성
if __name__ == "__main__":
    main()
```

---

## 03.friends_by_age.py

```python
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("FriendsByAge")
sc = SparkContext(conf = conf)

def parseLine(line):
    fields = line.split(',')
    age = int(fields[2])
    numFriends = int(fields[3])
    return (age, numFriends)

lines = sc.textFile("data/fakefriends.csv")
rdd = lines.map(parseLine)
totalsByAge = rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
averagesByAge = totalsByAge.mapValues(lambda x: x[0] / x[1])
results = averagesByAge.collect()
for result in results:
    print(result)
```

---

## 04.min_temp.py

```python
#온도를 측정하는 프로그램 만들기
from dataclasses import field
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster('local').setAppName('MinTemperatures') #마스터 노드에다가 올린다
sc = SparkContext(conf = conf)

print("Start")

def parseLine(line):
    
    fields = line.split(",") #쉼표로 다 끊어줌 -> 리스트로 반환됨
    stationID = fields[0]
    entryType = fields[2]
    temperature = float(fields[3]) * 0.1 * (9.0/5.0) + 32.0
    
    return (stationID, entryType, temperature)

lines = sc.textFile('data/1800.csv')
#print(lines)

parseLines = lines.map(parseLine)
#print(parseLine)

minTemps = parseLines.filter(lambda x: "TMIN" in x[1])
stationTemps = minTemps.map(lambda x: (x[0],x[2]))
minTemps = stationTemps.map(lambda x, y: min(x, y))
results = minTemps.collect()

#print(results)

for result in results:
    print(result[0]+ "\t{:.2f}F".format(result[1]))
```