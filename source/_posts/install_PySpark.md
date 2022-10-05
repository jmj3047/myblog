---
title: How to install PySpark
date: 2022-04-19
categories:
  - Python
  - Pyspark
tags: pyspark
---

## Preparation
- installing spark
- need python3
- if you are first using python, install anaconda

## Installing JAVA

- Installing file: **[Java SE 8 Archive Downloads (JDK 8u211 and later)](https://www.oracle.com/java/technologies/javase/javase8u211-later-archive-downloads.html)**
- Need to login Oracle
- Run the download file as admin → Click Next button → Changing the path on file (Space between words like `Program Files` can be problem during installation)
    
    ![](images/install_PySpark/Untitled.png)
    
- Changing Path
    
    ![](images/install_PySpark/Untitled%201.png)
    
- Same changes to folders in the JAVA runtime environment folder (Click 'Change' and modify)
    
    ![](images/install_PySpark/Untitled%202.png)
    
- Create and save `jre` folder in the path right after the C dirve

## Installing Spark

- Installing site:  **[https://spark.apache.org/downloads.html](https://spark.apache.org/downloads.html)**
    
    ![](images/install_PySpark/Untitled%203.png)
    
- Download installation file
    - After clicking `Download Spark: [spark-3.2.0-bin-hadoop3.2.tgz](https://www.apache.org/dyn/closer.lua/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz)`, you can download it by clicking the `HTTP 하단` page like picture below
        - Installation URL: **[https://www.apache.org/dyn/closer.lua/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz](https://www.apache.org/dyn/closer.lua/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz)** (2022.01)
        
        ![](images/install_PySpark/Untitled%204.png)
        
- Download WinRAR Program
    - You need to install `WinRAR`, to unzip `.tgz` file.
    - Installation file:  **[https://www.rarlab.com/download.htm](https://www.rarlab.com/download.htm)**
    - Install what fits your computer
    
    ![](images/install_PySpark/Untitled%205.png)
    
- Create Spark folder and move files
    - Moving files
        - Copy all the file in spark-3.2.0-bin-hadoop3.2 folder
    - After that, create spark folder below C drive and move all of them to it.
    
    ![](images/install_PySpark/Untitled%206.png)
    
- Modify log4j.properties file
    
    • Open the file`conf` - `[log4j.properties](http://log4j.properties)` 
    
    ![](images/install_PySpark/Untitled%207.png)
    
    - Open the log file as notebook and change `INFO` → `ERROR` just like example below.
        - During the process, all the output values can be removed.
        
        ```jsx
        # Set everything to be logged to the console
        # log4j.rootCategory=INFO, console
        log4j.rootCategory=ERROR, console
        log4j.appender.console=org.apache.log4j.ConsoleAppender
        log4j.appender.console.target=System.err
        log4j.appender.console.layout=org.apache.log4j.PatternLayout
        log4j.appender.console.layout.ConversionPattern=%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n
        ```
        

## Installing winutils

- This time, we need program that makes local computer mistakes Sparks for Hadoop.
    - Installing file: **[https://github.com/cdarlint/winutils](https://github.com/cdarlint/winutils)**
        - Download winutils programs that fit installation version.
    - I downloaded version 3.2.0
- Create winutils/bin folder on C drive and save the downloaded file.
- Ensure this file is authorized to be used so that it can be executed without errors whne running Spark
    - This time, open CMD as admin and run the file
    - If ChangeFileModeByMask error (3) occurs, create `tmp\hive` folder below C drive.

```jsx
C:\Windows\system32>cd c:\winutils\bin
c:\winutils\bin> winutils.exe chmod 777 \tmp\hive
```

## Setting environment variables

- Set the system environment variable
    - Click the `사용자 변수 - 새로 만들기` button on each user account
    
    ![](images/install_PySpark/Untitled%208.png)
    
- Set SPARK_HOME variable
    
    ![](images/install_PySpark/Untitled%209.png)
    
- Set JAVA_HOME variable
    
    ![](images/install_PySpark/Untitled%2010.png)
    
- Set HADOOP_HOME variable
    
    ![](images/install_PySpark/Untitled%2011.png)
    
- Edit `PATH` variable. Add the code below.
    
    ![](images/install_PySpark/Untitled%2012.png)
    
- Add code below
    - %SPARK_HOME%\bin
    - %JAVA_HOME%\bin
    
    ![](images/install_PySpark/Untitled%2013.png)
    

## Testing Spark

- Open CMD file, set the path as `c:\spark` folder
    - if the logo appears when input 'spark', success
    
    ![](images/install_PySpark/Untitled%2014.png)
    
- Check whether the code below works
    
    ```jsx
    >>> rd = sc.textFile("README.md")
    >>> rd.count()
    109
    >>>
    ```