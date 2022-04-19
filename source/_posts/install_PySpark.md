---
title: PySpark 설치 방법
date: 2022-04-19
categories:
  - Python 
tags: pyspark
---

## 사전 준비
- 스파크를 설치하는 과정
- 사전에 파이썬3 설치
- 파이썬이 처음이라면 Anaconda 설치

## 자바 설치

- 설치파일: **[Java SE 8 Archive Downloads (JDK 8u211 and later)](https://www.oracle.com/java/technologies/javase/javase8u211-later-archive-downloads.html)**
- 오라클 로그인 필요
- 다운로드 파일을 관리자로 실행 → Next 버튼 클릭 → 파일에서 경로 수정 (이 때, `Program Files` 공백이 있는데, 이러한 공백은 환경 설치 시 문제가 될 수 있음)
    
    ![](images/install_PySpark/Untitled.png)
    
- 경로 수정
    
    ![](images/install_PySpark/Untitled%201.png)
    
- 자바 런타임 환경의 폴더도 동일하게 변경해준다. (변경 클릭 후 수정)
    
    ![](images/install_PySpark/Untitled%202.png)
    
- C드라이브 바로 다음 경로에 `jre` 폴더를 생성하고 저장

## Spark 설치

- 설치 주소:  **[https://spark.apache.org/downloads.html](https://spark.apache.org/downloads.html)**
    
    ![](images/install_PySpark/Untitled%203.png)
    
- 설치 파일 다운로드
    - `Download Spark: [spark-3.2.0-bin-hadoop3.2.tgz](https://www.apache.org/dyn/closer.lua/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz)` 를 클릭 한 후, 아래 화면에서 `HTTP 하단` 페이지를 클릭하면 다운로드를 받을 수 있다.
        - 설치 URL: **[https://www.apache.org/dyn/closer.lua/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz](https://www.apache.org/dyn/closer.lua/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz)** (2022년 1월 기준)
        
        ![](images/install_PySpark/Untitled%204.png)
        
- WinRAR 프로그램 다운로드
    - 이 때, `.tgz` 압축파일을 풀기 위해서는 `WinRAR` 을 설치한다.
    - 설치 파일:  **[https://www.rarlab.com/download.htm](https://www.rarlab.com/download.htm)**
    - 각 개인 컴퓨터에 맞는 것을 설치
    
    ![](images/install_PySpark/Untitled%205.png)
    
- Spark 폴더 생성 및 파일 이동
    - 파일 이동을 하도록 한다.
        - spark-3.2.0-bin-hadoop3.2 폴더 내 모든 파일을 복사한다.
    - 그 후, C드라이브 하단에 spark 폴더를 생성한 후, 모두 옮긴다.
    
    ![](images/install_PySpark/Untitled%206.png)
    
- log4j.properties 파일 수정
    
    • `conf` - `[log4j.properties](http://log4j.properties)` 파일을 연다.
    
    ![](images/install_PySpark/Untitled%207.png)
    
    - 해당 파일을 메모장으로 연 후, 아래에서 `INFO` → `ERROR` 로 변경한다.
        - 작업 실행 시, 출력하는 모든 logs 값들을 없앨 수 있다.
        
        ```jsx
        # Set everything to be logged to the console
        # log4j.rootCategory=INFO, console
        log4j.rootCategory=ERROR, console
        log4j.appender.console=org.apache.log4j.ConsoleAppender
        log4j.appender.console.target=System.err
        log4j.appender.console.layout=org.apache.log4j.PatternLayout
        log4j.appender.console.layout.ConversionPattern=%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n
        ```
        

## winutils 설치

- 이번에는 스파크가 윈도우 로컬 컴퓨터가 Hadoop으로 착각하게 만들 프로그램이 필요하다.
    - 설치파일: **[https://github.com/cdarlint/winutils](https://github.com/cdarlint/winutils)**
        - 여기에서 각 설치 버전에 맞는 winutils를 다운로드 받는다.
    - 필자는 3.2.0 버전을 다운로드 받았다.
- C드라이브에서 winutils-bin 폴더를 차례로 생성한 후, 다운로드 받은 파일을 저장한다.
- 이 파일이 Spark 실행 시, 오류 없이 실행될 수 있도록 파일 사용 권한을 얻도록 한다.
    - 이 때에는 CMD 관리자 권한으로 파일을 열어서 실행한다.
    - 만약, ChangeFileModeByMask error (3) 에러 발생 시, C드라이브 하단에, `tmp\hive` 폴더를 차례대로 생성을 한다.

```jsx
C:\Windows\system32>cd c:\winutils\bin
c:\winutils\bin> winutils.exe chmod 777 \tmp\hive
```

## 환경변수 설정

- 시스템 환경변수를 설정한다.
    - 각 사용자 계정에 `사용자 변수 - 새로 만들기 버튼`을 클릭한다.
    
    ![](images/install_PySpark/Untitled%208.png)
    
- SPARK_HOME 환경변수를 설정한다.
    
    ![](images/install_PySpark/Untitled%209.png)
    
- JAVA_HOME 환경변수를 설정한다.
    
    ![](images/install_PySpark/Untitled%2010.png)
    
- HADOOP_HOME 환경변수를 설정한다.
    
    ![](images/install_PySpark/Untitled%2011.png)
    
- 이번에는 `PATH` 변수를 편집한다. 아래코드를 추가한다.
    
    ![](images/install_PySpark/Untitled%2012.png)
    
- 아래 코드를 추가한다.
    - %SPARK_HOME%\bin
    - %JAVA_HOME%\bin
    
    ![](images/install_PySpark/Untitled%2013.png)
    

## 스파크 테스트

- CMD 파일을 열고 `c:\spark` 폴더로 경로를 설정 한다.
    - pyspark를 입력했을 때 아래처럼 로고가 나오면 성공
    
    ![](images/install_PySpark/Untitled%2014.png)
    
- 아래 코드가 실행되는지 확인한다.
    
    ```jsx
    >>> rd = sc.textFile("README.md")
    >>> rd.count()
    109
    >>>
    ```