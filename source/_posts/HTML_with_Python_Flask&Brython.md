---
title: HTML with Python_Flask & Brython
date: 2022-05-04
categories:
  - Python
  - HTML
tags: 
  - Web Server
  - Flask
  - Brython
---

## Flask

- 파이썬 기반 마이크로 웹 개발 프레임워크
- 웹 개발의 핵심기능만 간경하게 유지
- 필요한 기능은 다른 라이브러리나 프레임워크로 손쉽게 확장
- 신속하게 최소한의 노력으로 웹 애플리케이션 개발 가능

### Installation

- start virtualenv
- pip install flask
    - Error
        
        → note: could not find a version that satisfies the requirement flask
        
        → 네트워크 문제로 외부 라이브러리 저장소에 접근하지 못할 경우 나오는 문제,
        
        → 직접 [https://github.com/mitsuhiko/flask](https://github.com/mitsuhiko/flask) 위치로 가서 소스 받아 설치 해야 함.
        

### Strat Flask

**code: [https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/flask_prac/0.flask_hello.py](https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/flask_prac/0.flask_hello.py)

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'
	
if __name__ == '__main__':
		app.debug =True
    app.run()
```

- 소스를 실행하고, terminal 에서 'python flask_test.py' 입력 후 http://127.0.0.1:5000 으로 접근
    
    ![ ](images/HTML_with_Python_Flask&Brython/Untitled.png)
    

### Process for starting Flask Application

1. 특정 URL 호출(request) : http://127.0.0.1:5000/ 또는 http://localhost:5000
2. 특정 URL 매핑 검색 : @app.route('/')
3. 특정 URL에 매칭된 함수(def 함수) 실행 : def hello_world()
4. 비즈니스 로직 실행 : result
5. 결과 응답으로 전송(response): return result
6. HTML 로 화면에 출력
7. 쿠키(Cookie), 세션(Session), 로깅(logging) 등 제공

### Routing

**code: [https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/flask_prac/1.flask_login.py](https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/flask_prac/1.flask_login.py)

- URL을 통해 처리할 핸들러를 찾는 것
- 플라스크는 복잡한 URI를 함수로 연결하는 방법을 제공
- URI 를 연결하는 route() 데코레이터 함수 제공
- / 접속 시 root_world() 가 호출 됨
- /hello 접속 시 hello_world() 가 호출 됨

```python
from flask import Flask, redirect, url_for
app = Flask(__name__)

@app.route('/') #127.0.0.1:5000에 가면 함수 실행
def root_world():
    result = 'root world'
    return result

@app.route('/hello') #127.0.0.1:5000/hello 를 가면 실행
def hello_world():
    result = 'hello world'
    return result

if __name__ == '__main__':
		app.debug =True
    app.run()
```

![](images/HTML_with_Python_Flask&Brython/Untitled%201.png)

- app.debug는 개발의 편의를 위해 존재
    - True값을 경우 코드를 변경하면 자동으로 서버가 재 실행 됨
    - 또한, 웹상에서 파이썬 코드를 수행할 수 있게 되므로, **운영환경**에서 **사용**을 **유의**해야 함.
    - 현재 접근은 개발 소스가 존재하는 로컬에서만 접근 가능
        - 외부에서도 접근을 가능하게 하려면 app.run(host='0.0.0.0')로 서버 실행 부를 변경해야 함

```python
from flask import Flask, redirect, url_for
app = Flask(__name__)

@app.route('/users/<user_id>') 
#동적 변수를 사용하여 URI 접속
# <동적변수>를 뷰함수의 인자로 사용
# <동적 변수> 다음에 /를 넣으면 안됨
def user_id(userid):
    result = 'user_id = ' + userid
    return result

@app.route('/admin')
def hello_admin():
    return 'Hello Admin'

@app.route('/guest/<guest>')
def hello_guest(guest):
    return 'Hello %s as Guest' % guest

@app.route('/user/<name>')
def hello_user(name):
    if name == 'admin':
        return redirect(url_for('hello_admin'))
    else:
        return redirect(url_for('hello_guest', guest=name))

# url_for(): 함수를 호출하는 URI를 반환
# redirect(): 다른 route 경로 이동(다른 페이지 이동)

if __name__ == '__main__':
    app.debug = True
    app.run()
```

![](images/HTML_with_Python_Flask&Brython/Untitled%202.png)

### **Flask GET 방식으로 값 전송 및 처리**

**code: [https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/flask_prac/2.flask_app.py](https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/flask_prac/2.flask_app.py)

[https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/flask_prac/templates/login/login_form_get.html](https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/flask_prac/templates/login/login_form_get.html)

- mkdir templates 폴더 생성
- login_form_get.html 파일 작성
- get 방식 지정 : method="get"
    
    ```python
    from flask import Flask, request, session, render_template
    app = Flask(__name__)
    
    @app.route('/login_form_get') 
    def login_form_get():
        return render_template('login/login_form_get.html')
    
    @app.route('/login_get_proc', methods=['GET']) 
    def login_get_proc():
        user_id = request.args.get('user_id')
        user_pwd = request.args.get('user_pwd')
        
        if len(user_id) == 0 or len(user_pwd) == 0:
            return 'no {} or {}'.format(user_id, user_pwd)
        
        return 'welcome {}'.format(user_id)
    
    if __name__ == '__main__':
        app.debug = True
        app.run()
    ```
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset = "UTF-8">
            <title>login_form_get.html</title>
        </head>
        <body>
            <h1>
                <form action="/login_get_proc" method="get">
                    ID: <input type="text", name="user_id"><br>
                    PW: <input type="password", name="user_pwd"><br>
                    <input type="submit", value="Click">
                </form>
            </h1>
        </body>
    </html>
    ```
    
    ![](images/HTML_with_Python_Flask&Brython/Untitled%203.png)
    

---

## Brython

**code:[https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/flask_prac/templates/brython_test.html](https://github.com/jmj3047/mj_chatbot_prac/blob/c5bec233b833b24345deeffe7391621415dc1dcb/flask_prac/templates/brython_test.html)

- python을 HTML 코드에 삽입해서 사용
    
    ```html
    <html>
    
        <head>
            <script type="text/javascript" src="/path/to/brython.js"></script>
        </head>
    
        <body onload="brython()">
    
            <script type="text/python">
            from browser import document, alert
    
            def echo(event):
                alert(document["zone"].value)
    
            document["mybutton"].bind("click", echo)
            </script>
    
            <input id="zone"><button id="mybutton">click !</button>
    
        </body>
    
    </html>
    ```
    

---

- Reference:
    - [https://essim92.tistory.com/8](https://essim92.tistory.com/8)
    - [https://code-examples.net/ko/q/dc0356](https://code-examples.net/ko/q/dc0356)
    - [https://github.com/brython-dev/brython](https://github.com/brython-dev/brython) #Test Brython online(DEMO)