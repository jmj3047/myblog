---
title: MongoDB Update Operator
date: 2022-04-28
categories:
  - Data Base
tags: 
  - MongoDB
---

- ```$set```: 필드값을 설정하고 필드가 존재하지 않으면 새 필드가 생성됨. 스키마를 갱신하거나 사용자 정의 키를 추가 할때 편리함.
- ```$unset```: 키와 값을 모두 제거함

```sql
> db.users.insertOne({"name":"joe"})
> db.users.updateOne({"_id" : ObjectId("6269e32b7b15b7097fbad433")},{"$set":{"favorite book" : "War and Peace"}})
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }

> db.users.findOne()
{
        "_id" : ObjectId("6269e32b7b15b7097fbad433"),
        "name" : "joe",
        "favorite book" : "War and Peace"
}
> db.users.updateOne({"name" : "joe"},{"$unset":{"favorite book":1}})
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }
> db.users.find()
{ "_id" : ObjectId("6269e32b7b15b7097fbad433"), "name" : "joe" }
```


- ```$inc```: ```$set```과 비슷하지만, 숫자를 증감하기 위해 사용. int, long, double, decimal 타입 값에만 사용 가능

```sql

>db.games.insertOne({"game" :"pinball","user" :"joe"})
{
        "acknowledged" : true,
        "insertedId" : ObjectId("6269e4be7b15b7097fbad436")
}

> db.games.find()
{ "_id" : ObjectId("6269e4be7b15b7097fbad436"), "game" : "pinball", "user" : "joe" }

> db.games.updateOne({"game" :"pinball","user" :"joe"},{"$inc":{"score":50}})
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }

> db.games.find()
{ "_id" : ObjectId("6269e4be7b15b7097fbad436"), "game" : "pinball", "user" : "joe", "score" : 50 }

> db.games.updateOne({"game" :"pinball","user" :"joe"},{"$inc":{"score":10000}})
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }

> db.games.find()
{ "_id" : ObjectId("6269e4be7b15b7097fbad436"), "game" : "pinball", "user" : "joe", "score" : 10050 }
>
```

- ```$push```: 배열이 이미 존재하지만 배열 끝에 요소를 추가하고, 존재하지 않으면 새로운 배열을 생성함.
- ```$each```: ```$push```에 ```$each```제한자를 사용하면 작업 한 번으로 값을 여러개 추가할 수 있음.

```sql
> db.blog.posts.findOne() 
{ 
	"_id" : ObjectId("4b2d75476cc613d5ee930164"), "title" : "A blog post", "content" : "..."
} 

> db.blog.posts.updateOne({"title" : "A blog post"}, 
	{"$push" : {"comments" : 
		{"name" : "joe", "email" : "joe@example.com", 
		"content" : "nice post."}}}) 
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 } 

> db.blog.posts.findOne() 
{
	 "_id" : ObjectId("4b2d75476cc613d5ee930164"), 
	"title" : "A blog post", "content" : "...", 
	"comments" : [
		 {"name" : "joe", 
		"email" : "joe@example.com", 
		"content" : "nice post."
		} 
	] 
}
```

- ```$ne```: 배열이 존재하지 않을 때 해당 값을 추가하면서 배열을 집합처럼 처리할 때 사용.
- ```$addToSet```: 다른주소를 추가할 때 중복을 피할 수 있음
- 고유한 값을 여러개 추가하려면 ```$addToSet```/```$each```조합을 활용해야 함. ```$ne```/```$push```조합으로는 할 수 없음.


```sql
> db.users.updateOne({"_id" : ObjectId("4b2d75476cc613d5ee930164")}, 
	{"$addToSet" : {"emails" : {"$each" :
		["joe@php.net", "joe@example.com", "joe@python.org"]}}}) 
{ "acknowledged" : true, "matchedCount" : 1, "modifiedCount" : 1 }
 
> db.users.findOne({"_id" : ObjectId("4b2d75476cc613d5ee930164")}) 
{ 
	"_id" : ObjectId("4b2d75476cc613d5ee930164"), 
	"username" : "joe", 
	"emails" : [ "joe@example.com", "joe@gmail.com", "joe@yahoo.com", "joe@hotmail.com" "joe@php.net" "joe@python.org" 
	] 
}
```

---
Reference
- 몽고DB 완벽 가이드: 실전 예제로 배우는 NoSQL 데이터베이스 기초부터 활용까지

