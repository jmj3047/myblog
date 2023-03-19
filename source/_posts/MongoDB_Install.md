---
title: MongoDB Install & Basic Command
date: 2022-04-25
categories:
  - Data Platform/Base
  - MongoDB
tags: 
  - MongoDB
---


- Link: [www.mongodb.com/try/download/enterprise](https://www.mongodb.com/try/download/enterprise)
- Download proper version of Mongodb

![](images/MongoDB_Install/Untitled.png)

Install이 완료된 후에는 MongoDB 환경변수 설정을 위해 시스템 환경 변수 편집을 진행하여 줍니다.

환경변수 편집을 위해 환경변수 >시스템 변수 Path 설정을 선택하여 줍니다.

![](images/MongoDB_Install/Untitled%201.png)

설치된 MongoDB의 bin폴더 경로를 입력하여 줍니다.(C:\Program Files\MongoDB\Server\5.0\bin)

![](images/MongoDB_Install/Untitled%202.png)

저장 후 cmd창에서 mongdb --version을 통해 정상 설치를 확인하여 줍니다.

cmd 창에 mongodb 실행

```powershell
>mongo
```

명령어 두줄로 잘 실행되는지 간단히 확인

```powershell
> db.world.insert({ "speech" : "Hello World!" });
> cur = db.world.find();x=cur.next();print(x["speech"]);
```

## Basic Command

사용 가능한 모든 데이터베이스 표시 :

```sql
show dbs;
```

액세스 할 특정 데이터베이스를 선택 (Ex: `mydb` . 이미 존재하지 않으면 `mydb` 가 생성됩니다 :

```sql
use mydb;
```

데이터베이스에 모든 콜렉션을 표시. 먼저 콜렉션을 선택하십시오 (위 참조).

```sql
show collections;
```

데이터베이스와 함께 사용할 수있는 모든 기능 표시 :

```sql
db.mydb.help();
```

현재 선택한 데이터베이스를 확인

```sql
> db
mydb
```

`db.dropDatabase()` 명령은 기존 데이터베이스를 삭제하는 데 사용됩니다.

```sql
db.dropDatabase()
```

---

Reference
- [https://khj93.tistory.com/entry/MongoDB-Window에-MongoDB-설치하기](https://khj93.tistory.com/entry/MongoDB-Window%EC%97%90-MongoDB-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0)
- [https://learntutorials.net/ko/mongodb/topic/691/mongodb-시작하기](https://learntutorials.net/ko/mongodb/topic/691/mongodb-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0)