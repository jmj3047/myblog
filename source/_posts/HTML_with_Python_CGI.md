---
title: HTML with Python_CGI
date: 2022-05-04
categories:
  - Python
  - HTML
tags: 
  - Web Server
  - CGI
---


** code: [https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/cgi_webpython.py](https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/cgi_webpython.py)

### Making Website with CGI

1. Constructing Web Server: Download and Install Apche 
    - Official: Installing Apache in window([https://httpd.apache.org/docs/2.4/platform/windows.html](https://httpd.apache.org/docs/2.4/platform/windows.html))
    - Beginner version: Install Bitnami Wamp Stack
        
        ![](images/HTML_with_Python_CGI/Untitled.png)
        
    - push 1 or 2
        
        ![](images/HTML_with_Python_CGI/Untitled%201.png)
        
    - click this button below
        
        ![](images/HTML_with_Python_CGI/Untitled%202.png)
        
        ##### it takes some time to install it #####
        

---

1. Constructing Web Server: Bitnami Wamp Stack
    - Start Bitnami Wamp Stack
        
        ![](images/HTML_with_Python_CGI/Untitled%203.png)
        
    - Click Go to Application: If you can see the site like the picture below, success!
        
        ![](images/HTML_with_Python_CGI/Untitled%204.png)
        
    - Start or Stop the server: Click Manage Server
        - If program below is shut down, go to the folder where ‘bitnami wamp stack’ installed and click ‘manager-windows.exe’
        
        ![](images/HTML_with_Python_CGI/Untitled%203.png)
        
        ![](images/HTML_with_Python_CGI/Untitled%205.png)
        

---

1. Using Python in Web(HTML): Setting Apache
    - Install python and apache
    - find folder where apach installed(‘D:\wamp\apache2\conf’) > ‘conf’ folder > httpd.conf
    - open httpd.conf file and search:
        
        ```
        LoadModule cgid_module modules/mod_cgid.so
        ```
        
        and if ‘#’ exists in front of code, delete it
        
    - Find <Directory code> tag in httpd.conf and add some lines
        
        ```html
        <Directory "/Applications/mampstack-8.0.5-0/apache2/htdocs">
              #
              # Possible values for the Options directive are "None", "All",
              # or any combination of:
              #   Indexes Includes FollowSymLinks SymLinksifOwnerMatch ExecCGI 			MultiViews
              #
              # Note that "MultiViews" must be named *explicitly* --- "Options All"
              # doesn't give it to you.
              #
              # The Options directive is both complicated and important.  Please see
              # http://httpd.apache.org/docs/2.4/mod/core.html#options
              # for more information.
              #
              Options Indexes FollowSymLinks
        
              #
              # AllowOverride controls what directives may be placed in .htaccess files.
              # It can be "All", "None", or any combination of the keywords:
              #   AllowOverride FileInfo AuthConfig Limit
              #
              AllowOverride None
        
              #
              # Controls who can get stuff from this server.
              #
              Require all granted
        
        		********** add this code ********** 
              <Files *.py>
                  Options ExecCGI
                  AddHandler cgi-script .py
              </Files>
            ***********************************
          
        	</Directory>
        ```
        
        - htdocs 디렉토리 내 확장자가 py인 모든 파일은 CGI기능을 활성시키고 CGI로 실행하라는 의미
    - Restart Apache Web Server in manager-osx

---

1. Python file setting 
    - [index.py](http://index.py/) 가 있는 htdocs 디렉토리에서 index.py 실행 후 아래같이 입력(다른 파이썬 파일을 만들어도 상관 없음)
    
    ```python
    #!/usr/local/bin/python3 >>> python.exe 경로 환경변수에 저장해줬다면 !Python만 해도 됨  
      print("Content-Type: text/html")
      print()
    ```
    
    - index.html 의 코드 넣기( 다른 html 파일 이어도 됨): index.py가 실행되었을 때 index.html 의 코드가 출력되게 해주는 코드
    
    ```python
    #!/usr/local/bin/python3
    print("Content-Type: text/html")
    print()
    print('''<!doctype html>  # ---> 줄바꿈을 위해 docsting (''' ''') 사용
    <html>
    <head>
      <title>WEB1 - Welcome</title>
      <meta charset="utf-8">
    </head>
    <body>
      <h1><a href="index.html">WEB</a></h1>
      <ol>
        <li><a href="qs-1.html">HTML</a></li>
        <li><a href="qs-2.html">CSS</a></li>
        <li><a href="qs-3.html">JavaScript</a></li>
      </ol>
      <h2>WEB</h2>
      <p>The World Wide Web (abbreviated WWW or the Web) is an information space where documents and other web resources are identified by Uniform Resource Locators (URLs), interlinked by hypertext links, and can be accessed via the Internet.[1] English scientist Tim Berners-Lee invented the World Wide Web in 1989. He wrote the first web browser computer program in 1990 while employed at CERN in Switzerland.[2][3] The Web browser was released outside of CERN in 1991, first to other research institutions starting in January 1991 and to the general public on the Internet in August 1991.
      </p>
    </body>
    </html>
    ''')
    ```
    
    - 웹 브라우저 주소창에 localhost:8080/index.py 입력하고 접속
    - index.html 파일의 내용이 잘 출력된다면 구현 성공
    - Internal Server Error 가 확인된다면 에디터에서 apache2/logs 디렉토리 내 error_log 파일에 있는 에러 코드 확인 및 구글링
    

### EXAMPLE

```python
#!C:\Python310\python.exe --->파이썬 경로

# 한글이 꺠지지 않으려면 꼭 넣어야 함
# -*- coding:utf-8 -*-
import sys
import codecs
sys.stdout =codecs.getwriter("utf-8")(sys.stdout.detach())

import cgi
# cgitb는 CGI 프로그래밍시 디버깅을 위한 모듈로 cgitb.enable()할 경우 런타임 에러를 웹브라우저로 전송함
# cgitb.enable() 하지 않은 상태로 실행 중 오류가 발생한 경우 웹서버는 클라이언트에게 HTTP응답 코드 500을 전송함
import cgitb
cgitb.enable()
# HTTP 규격에서 헤더 전송 이후에는 반드시 줄 바꿈을 하게되어 있음으로 마지막에 \r\n을 전송
# 마지막에 \r\n을 전송하지 않으면 브라우저 측에서 오류가 발생
print("Content-type: text/html;charset=utf-8\r\n")
print("""
      <!doctype html>
      <html>
      <head>
      <meta charset='utf-8'>
      <h1>안녕?</h1>
      <h2>Thank you so much</h2>
      <h3>This page is made by Python</h3>     
      """)
a = 3+4+5
b = a/3
print('b는 :', b)
print("</head>")
print("</html>")
```

#### Result
![](images/HTML_with_Python_CGI/Untitled%206.png)

---

- Reference
    - [https://daekiry.tistory.com/4?category=928946](https://daekiry.tistory.com/4?category=928946)
    - [https://daekiry.tistory.com/5?category=928946](https://daekiry.tistory.com/5?category=928946)
    - [https://daekiry.tistory.com/6](https://daekiry.tistory.com/6)
    - [https://velog.io/@ssoulll/python-웹-페이지를-CGI로-구현](https://velog.io/@ssoulll/python-%EC%9B%B9-%ED%8E%98%EC%9D%B4%EC%A7%80%EB%A5%BC-CGI%EB%A1%9C-%EA%B5%AC%ED%98%84)