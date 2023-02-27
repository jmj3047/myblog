---
title: Mac bash 파일로 hexo, git 명령어 자동화
date: 2023-02-27
categories:
  - Setting
  - Git
tags: 
  - Git
  - Hexo 
  - Automation
  - Bash
---
### Mac bash 파일로 hexo, git 명령어 자동화

- 필자가 블로그글을 작성하는데 hexo, git 명령어 자동화의 필요성을 느껴 이 글을 작성함
- mac에서 자동화 하는 경우 윈도우와 달리 batch 파일이 아니라 bash 파일로 실행해야 한다.
- 우선 메모장에 자동화를 원하는 코드를 작성한다.
- bash 파일 작성시에는 `#!/bin/bash` 를 꼭 작성해주어야 실행이 가능하다.

![](images/Bash_Automation/Untitled.png)

- 실행파일 이름을 정하고(필자는 `submit` 으로 설정하였음) 저장한 뒤 확장자를 없애주어야 한다.
- 저장한 `submit` 파일을 블로그 로컬 폴더 안에 넣는다

![](images/Bash_Automation/Untitled%201.png)

- 터미널에 `chmod +x submit` 를 입력해서 권한을 부여해준다
- 실행시키려면 `sh submit` 을 입력해주면 잘 돌아가는 것을 볼 수 있다.

![](images/Bash_Automation/Untitled%202.png)