---
title: (Error solved) Accessing non-existent property 'lineno' of module exports inside circular dependency
date: 2023-05-07
categories:
  - Setting 
tags: 
  - Hexo
  - English
---

### (node:14126) Warning, Accessing non-existent property 'lineno' of module exports inside circular dependency

어느날 부터인가 블로그 업로드 전에 `hexo sever` 를 입력하면 밑에 node 관련된 warning이 뜨기 시작했다. 
- For some reason, when I type `hexo sever` before uploading a blog, a warning about nodes appears below.

```sql
MacBookPro myblog % hexo server
INFO  Validating config
INFO  Start processing
INFO  Hexo is running at http://localhost:4000/ . Press Ctrl+C to stop.
(node:14126) Warning: Accessing non-existent property 'lineno' of module exports inside circular dependency
(Use `node --trace-warnings ...` to show where the warning was created)
(node:14126) Warning: Accessing non-existent property 'column' of module exports inside circular dependency
(node:14126) Warning: Accessing non-existent property 'filename' of module exports inside circular dependency
(node:14126) Warning: Accessing non-existent property 'lineno' of module exports inside circular dependency
(node:14126) Warning: Accessing non-existent property 'column' of module exports inside circular dependency
(node:14126) Warning: Accessing non-existent property 'filename' of module exports inside circular dependency
```

게시물이 업로드 되는데에는 문제가 없었지만, 찝찝해서 해결하려고 찾아본 결과: 
- I didn't have any issues with the post uploading, but when I went to troubleshoot, I found that this code: 

```sql
rm -rf node_modules package-lock.json && npm install && npm run
```

위 명령어를 입력했더니 해결 되었다. 대충 찾아봤더니 npm과 node의 버전 문제 아니면 package-lock.json의 문제 둘중에 하나 인 거 같았다. 
- I entered the above command and it worked. After a quick search, I realized it was either a version issue with npm and node or a problem with package-lock.json.

---

- Reference: [https://github.com/nodejs/help/issues/2347](https://github.com/nodejs/help/issues/2347)