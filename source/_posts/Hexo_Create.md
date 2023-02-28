---
title: Hexo Blog 생성 및 재연결
date: 2022-10-06
categories:
  - Setting 
tags: 
  - Hexo
---

## Hexo Blog 생성

- 간단하게 `Hexo` 블로그를 만들어 본다.

### I. 필수 파일 설치

- 1단계: **[nodejs.org](https://nodejs.org/en/)** 다운로드
    - 설치가 완료 되었다면 간단하게 확인해본다.
        
        ```bash
        $ node -v
        ```
        
- 2단계: **[git-scm.com](https://git-scm.com/)** 다운로드
    - 설치가 완료 되었다면 간단하게 확인해본다.
        
        ```bash
        $ git --version
        ```
        
- 3단계: hexo 설치
    - hexo는 npm을 통해서 설치가 가능하다.
        
        ```bash
        $ npm install -g hexo-cli
        ```
        

### **II. 깃허브 설정**

- 두개의 깃허브 `Repo`를 생성한다.
    - 포스트 버전관리 (name: myblog)
    - 포스트 배포용 관리 (name: rain0430.github.io)
    - `rain0430` 대신에 각자의 `username`을 입력하면 된다.
- 이 때, `myblog repo`를 `git clone`을 통해 적당한 경로로 내려 받는다.

`$ git clone your_git_repo_address.git`

### **III. 블로그 만들기**

- (옵션) 적당한 곳에 경로를 지정한 다음 다음과 같이 폴더를 만든다.
    
    ```bash
    $ mkdir makeBlog # 만약 Powershell 이라면 mkdir 대신에 md를 쓴다. 
    $ cd makeBlog
    ```
    
- 임의의 블로그 파일명을 만든다.
    
    ```bash
    $ hexo init myblog
    $ cd myblog
    $ npm install
    $ npm install hexo-server --save
    $ npm install hexo-deployer-git --save
    ```
    

+ ERROR Deployer not found: git
+ hexo-deployer-git을 설치 하지 않으면 deploy시 위와 같은 ERROR가 발생합니다.

- `_config.yml` 파일 설정
    - 싸이트 정보 수정
        
        ```yaml
        title: 제목을 지어주세요
        subtitle: 부제목을 지어주세요
        description: description을 지어주세요
        author: YourName
        ```
        
    - 블로그 URL 정보 설정
        
        ```yaml
        url: https://rain0430.github.io
        root: /
        permalink: :year/:month/:day/:title/
        permalink_defaults:
        ```
        
    - 깃허브 연동
        
        ```yaml
        # Deployment
        deploy:
          type: git
          repo: https://github.com/rain0430/rain0430.github.io.git
          branch: main
        ```
        

### **IV. 깃허브에 배포하기**

- 배포 전, 터미널에서 `localhost:4000` 접속을 통해 화면이 뜨는지 확인해본다.
    
    ```bash
    $ hexo generate
    $ hexo server
    INFO  Start processing
    INFO  Hexo is running at http://localhost:4000 . Press Ctrl+C to stop.
    ```
    
- 화면 확인이 된 이후에는 깃허브에 배포한다.
- 사전에, `gitignore` 파일에서 아래와 같이 설정을 진행한다.
    
    ```
    .DS_Store
    Thumbs.db
    db.json
    *.log
    node_modules/
    public/
    .deploy*/
    ```
    
- 최종적으로 배포를 진행한다.
    
    ```bash
    $ hexo deploy
    ```
    
- 배포가 완료가 되면 브라우저에서 `USERNAME.github.io`로 접속해 정상적으로 배포가 되었는지 확인한다.

## Hexo Blog 재연결

- 기존 블로그 폴더 파일 압축해서 백업한 후 진행해야 한다. → theme 같은 경우 받아오는거부터 다시 해야 하기 때문
- 재연결보다는 재생성이라고 말하는게 더 적합하다.
- 다른 로컬에서 블로그를 재연결해서 사용할 경우 아래와 같이 순차적으로 진행하면 된다.
    
    ```bash
    $ hexo init your_blog_repo # 여기는 각자 소스 레포 확인
    $ cd myblog
    $ git init 
    $ git remote add origin https://github.com/your_name/your_blog_repo.git # 각자 소스 레포 주소
    ```
    
- 아래 명령어에서 에러가 발생이 있다.
    
    ```bash
    $ git pull --set-upstream origin main # 에러 발생
    ```
    
- 그런 경우, 아래 명령어를 추가한다. 기존의 디렉토리와 파일을 모두 삭제한다는 뜻이다.
    
    ```bash
    $ git clean -d -f
    ```
    
- 그리고 에러가 발생했던 명령어를 다시 실행한다.
- 이 때에는 이제 정상적으로 실행되는 것을 확인할 수 있다.
    
    ```bash
    $ git pull --set-upstream origin main # 에러 발생 안함 / 소스 확인
    ```
    
- 이제 정상적으로 환경 세팅은 된 것이다. 순차적으로 아래와 같이 진행하도록 한다.
    - 이 때, theme 폴더에 본인의 테마 소스코드가 잘 있는지 확인을 하도록 한다.
        
        ```bash
        $ npm install 
        $ hexo clean
        $ hexo generate
        $ hexo server
        ```
        

---

- reference
    - [생성](https://dschloe.github.io/settings/hexo_blog/)
    - [재연결](https://dschloe.github.io/settings/hexo_blog_reconnected/)