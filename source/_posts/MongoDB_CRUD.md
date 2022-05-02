---
title: MongoDB CRUD
date: 2022-04-25
categories:
  - Data Base
tags: 
  - MongoDB

---

<!-- {% githubCard user:jmj3047 %} -->

- Initial setting

```sql
>db
test
>use video
switched to db video # if video doesnt exist, created
>db
video
>db.movies # created movies collection
video.movies
```

- Create: insertOne 함수

```sql
>movie = {"title" : "Star Wars: Episode IV - A New Hope", "director" : "George Lucas", "year" : 1977}
{
        "title" : "Star Wars: Episode IV - A New Hope",
        "director" : "George Lucas",
        "year" : 1977
}

>db.movies.insertOne(movie) #영화가 데이터 베이스에 저장됨
{
        "acknowledged" : true,
        "insertedId" : ObjectId("6266057e4619361339f0c881")
}
#Find 함수로 호출
>db.movies.find().pretty()
{
        "_id" : ObjectId("6266057e4619361339f0c881"),
        "title" : "Star Wars: Episode IV - A New Hope",
        "director" : "George Lucas",
        "year" : 1977
}
```

- Read: find, findOne 함수

```sql
> db.movies.findOne(movie)  
{
        "_id" : ObjectId("6266057e4619361339f0c881"),
        "title" : "Star Wars: Episode IV - A New Hope",
        "director" : "George Lucas",
        "year" : 1977
}
```

- Update : updateOne 함수

```sql
> db.movies.updateOne({title : "Star Wars: Episode IV - A New Hope"}, {$set : {reviews: []}})    
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }

> db.movies.find().pretty()
{
        "_id" : ObjectId("6266057e4619361339f0c881"),
        "title" : "Star Wars: Episode IV - A New Hope",
        "director" : "George Lucas",
        "year" : 1977,
        "reviews" : [ ]
}
```

- Delete: deleteOne, deleteMany 함수

```sql
>db.movies.deleteOne({title : "Star Wars: Episode IV - A New Hope"})
{ "acknowledged" : true, "deletedCount" : 1 }
```

---
Reference
- 몽고DB 완벽 가이드: 실전 예제로 배우는 NoSQL 데이터베이스 기초부터 활용까지

