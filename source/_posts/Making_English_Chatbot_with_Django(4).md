---
title: Making English Chatbot with Django(4)
date: 2022-05-06
categories:
  - Python
  - Django
tags: 
  - Doc2vec
  - Chatbot
  - NLP
---

## 실제 서비스 구현해보기

**code: [https://github.com/jmj3047/faq_chatbot_example.git](https://github.com/jmj3047/faq_chatbot_example.git)

vs code로 django 설정하기: [https://integer-ji.tistory.com/81](https://integer-ji.tistory.com/81)

### 채팅창 만들기

![](images/Making_English_Chatbot_with_Django(4)/Untitled.png)

html/css를 사용해 간단한 채팅화면을 만들었다.

```html
<!-- //templates/addresses/chat_test.html -->
<!DOCTYPE html>
<html lang="en">
<script type="text/javascript" src="/static/jquery-3.2.1.min.js"></script>
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inconsolata:wght@400;600&display=swap" rel="stylesheet">
</head>
<style>
* {font-family: 'Inconsolata', monospace;}
.chat_wrap {display:none;width: 350px;height: 500px;position: fixed;bottom: 30px;right: 95px;background: #a9bdce;}
.chat_content {font-size:16pt; position:relative; height: 600px;width: 500px;overflow-y:scroll;padding:10px 15px;background: cornflowerblue}
.chat_input {border:solid 0.5px lightgray; padding:2px 5px;}
.chat_header {padding: 10px 15px; width: 500px; border-bottom: 1px solid #95a6b4;}
.chat_header .close_btn {border: none;background: lightgray;float: right;}
.send_btn {border: none; background: #ffeb33;height: 100%; color: #0a0a0a;}
.msg_box:after {content: '';display: block;clear:both;}
.msg_box > span {padding: 3px 5px;word-break: break-all;display: block;max-width: 300px;margin-bottom: 10px;border-radius: 4px}
.msg_box.send > span {background:#ffeb33;float: right;}
.msg_box.receive > span {background:#fff;float: left;}
</style>
<body>
<div class="chat_header">
    <span style="font-size:20pt;">EDITH</span>
    <button type="button" id="close_chat_btn" class="close_btn">X</button>
</div>
<div id="divbox" class="chat_content"></div>
<form id="form" style="display: inline">
    <input type="text" placeholder="write message.." name="input1" class="chat_input" id="input1" size="74" style="margin:-3px; display: inline; width: 468px; height: 32px; font-size: 16pt;" />
    <input type="button" value="SEND" id="btn_submit" class="send_btn" style="margin:-5px; display: inline; width: 53px; height: 38px; font-size: 14pt;"  />
</form>
<script>
    $('#btn_submit').click(function () {
        send();
    });
    $('#form').on('submit', function(e){
       e.preventDefault();
       send();
    });
    $('#close_chat_btn').on('click', function(){
        $('#chat_wrap').hide().empty();
    });
    function send(){
        $('#divbox').append('<div class="msg_box send"><span>'+$('#input1').val()+'<span></div>');
        $("#divbox").scrollTop($("#divbox")[0].scrollHeight);
        console.log("serial"+$('form').serialize())
        $.ajax({
            url:  'http://127.0.0.1:8000/chat_service/', //챗봇 api url
            type: 'post',
            dataType: 'json',
            data: $('form').serialize(),
            success: function(data) {
                <!--$('#reponse').html(data.reponse);-->
                $('#divbox').append('<div class="msg_box receive"><span>'+ data.response +'<span></div>');
                $("#divbox").scrollTop($("#divbox")[0].scrollHeight);
            }
        });
        $('#input1').val('');
    }
</script>
</body>
</html>
```

간단하게 설명하면 Django로 restfulAPI를 구현하기 위한 소스 위에 챗봇을 붙이기 위한 화면과 모델이 들어가 있는 버전이다. 위에 소스는 화면 역할을 하는 chat_test.html 파일이다. 

jquery 라이브러리를 사용했기 때문에 jquery를 import 해야 한다. jquery file이 static 폴더에 있어야 한다.

jquery는 소스 하단부에 있는 script를 위해 필요하다. 채팅에서 전송 버튼을 누르거나 엔터를 누르면 send()라는 함수가 실행되고 이 함수는 ajax로 질문에 대한 답변을 받아오는 API를 호출한다. 여기서는 [localhost/chat_service를](http://localhost/chat_service를) 호출한다. 

### 채팅을 위한 API

화면이 만들어졌으면 이제 질문을 받아 답변을 생성하는 API를 만든다. 아직 FAQ데이터를 학습한 모델은 넣지 않았으니 인풋이 들어오면 더미데이터(dummy)를 리턴하는 API를 만든다. 이런 API 동작들은 view.py에서 구현할 수 있다. 

```python
#faq_chatbot_example/addresses/views.py

@csrf_exempt
def chat_service(request):
    if request.method == 'POST':
        input1 = request.POST['input1']
        output = dict()
        output['response'] = "이건 응답"
        return HttpResponse(json.dumps(output), status=200)
    else:
        return render(request, 'addresses/chat_test.html')
```

Django 프로젝트 안에 addresses 앱에 있는 views.py를 보면 chat_service 함수를 만들었다. POST형식으로 콜이 오면 response에 아웃풋 메세지를 담아서 json형태로 리턴한다. views.py에 함수를 만들고 url로 연결하기 위해서 urls.py에 chat_service를 입력한다. 

```python
###django  3.8.3 버전 맞춰줘야 함
#/faq_chatbot_example/restfulapiserver/urls.py
# from django.conf.urls import url, include
from addresses import views
from django.urls import path, re_path, include
from django.contrib import admin

urlpatterns = [
						...
    path('chat_service/', views.chat_service),
						...
]
```

urls.py에서 ~/chat_service를 views.chat_service에 연결시킨다.  이제 ~/chat_service로 콜하면 views.chat_service가 실행된다. 아까 위에서 만든 채팅페이지에 전송버튼을 누르면 ajax를 이용해 chat_service를 호출했다. 정상적으로 되는지 테스트 해본다.

### FAQ 모델 넣기

addresses 앱 안에 새로운 py 모델을 만들어서 넣기

```python
#/faq_chatbot_example/addresses/faq_chatbot.py
from gensim.models import doc2vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# 파일로부터 모델을 읽는다. 없으면 생성한다.
try:
    d2v_faqs = Doc2Vec.load('d2v_faqs_size200_min5_epoch20_jokes.model')
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    faqs = pd.read_csv('jokes.csv')
except:
    faqs = pd.read_csv('jokes.csv')
    nltk.download('punkt')
    # 토근화
    tokened_questions = [word_tokenize(question.lower()) for question in faqs['Question']]
    lemmatizer = WordNetLemmatizer()
    nltk.download('wordnet')
    # lemmatization
    lemmed_questions = [[lemmatizer.lemmatize(word) for word in doc] for doc in tokened_questions]
    nltk.download('stopwords')
    # stopword 제거 불용어 제거하기
    stop_words = stopwords.words('english')
    questions = [[w for w in doc if not w in stop_words] for doc in lemmed_questions]
    # 리스트에서 각 문장부분 토큰화
    index_questions = []
    for i in range(len(faqs)):
        index_questions.append([questions[i], i ])

    # Doc2Vec에서 사용하는 태그문서형으로 변경
    tagged_questions = [TaggedDocument(d, [int(c)]) for d, c in index_questions]
    # make model
    import multiprocessing
    cores = multiprocessing.cpu_count()
    d2v_faqs = doc2vec.Doc2Vec(
                                    vector_size=200,
                                    hs=1,
                                    negative=0,
                                    dm=0,
                                    dbow_words=1,
                                    min_count=5,
                                    workers=cores,
                                    seed=0,
                                    epochs=20
                                    )
    d2v_faqs.build_vocab(tagged_questions)
    d2v_faqs.train(tagged_questions,
                   total_examples=d2v_faqs.corpus_count,
                   epochs=d2v_faqs.epochs)

    d2v_faqs.save('d2v_faqs_size200_min5_epoch20_jokes.model')

# FAQ 답변
def faq_answer(input):
    # 테스트하는 문장도 같은 전처리를 해준다.
    tokened_test_string = word_tokenize(input)
    lemmed_test_string = [lemmatizer.lemmatize(word) for word in tokened_test_string]
    test_string = [w for w in lemmed_test_string if not w in stop_words]

    topn = 5
    test_vector = d2v_faqs.infer_vector(test_string)
    result = d2v_faqs.docvecs.most_similar([test_vector], topn=topn)
    print(result)

    for i in range(topn):
        print("{}위. {}, {} {} {}".format(i + 1, result[i][1], result[i][0], faqs['Question'][result[i][0]], faqs['Answer'][result[i][0]]))

    return faqs['Answer'][result[0][0]]

faq_answer("What do you call a person who is outside a door and has no arms nor legs?")
```

위 소스에서 상단에 있는 모델을 만드는 코드는 API 서버를 실행하는 시점에서 호출된다. 무조건 호출하는 건 아니고 views.py에서 import를 써 넣으면 최초 1번은 실행되게 된다. 채팅 웹페이지로부터 faq_chatbot.py에 있는 faq_answer를 호출하는 것 까지 flow를 그려보면 chat_test.html→view.py(chat_service)→faq_chatbot.py(faq_answer) 순서이다. 따라서 views.py에서 faq_answer함수를 호출하기 위해 import를 하게 되는데 django는 최초 실행시 views.py를 한번 읽기 때문에 faq_chatbot.py에 적어놓은 소스가 한번 실행되게 된다. 

매번서버를 실행할 때마다 모델을 새로 만들게 되면 서버 기동 속도가 느려지고 비효율적이기 때문에 모델을 만들고 나서 패일로 저장하고, 만들어진 파일이 없다면 모델을 생성하도록 try/except를 사용했다. 

추가적으로 프로젝트상 소스가 실행되기 때문에 파일경로는 root이다. jokes.csv가 있어야 할 곳과 모델이 생성되는 곳의 경로는 프로젝트의 root폴더이다. 

자 이제 질문의 답을 찾아주는 함수가 만들어 졌으니 아까 더미 데이터로 리턴해주던 views.py의 함수를 바꿔보자. 

```python
#faq_chatbot_example/addresses/views.py

@csrf_exempt
def chat_service(request):
    if request.method == 'POST':
        input1 = request.POST['input1']
        response = faq_answer(input1)
        output = dict()
        output['response'] = response
        return HttpResponse(json.dumps(output), status=200)
    else:
        return render(request, 'addresses/chat_test1.html')
```

이전에는 response에 무조건 더미 응답을 보냈는데 이제는 faq_answer함수를 사용해 해당 질문에 알맞은 정답을 가져온다. faq_answer함수를 사용하기 위해 제일 상단에 from .faq_chatbot import faq_answer를 선언해야 한다. 

### 실행 결과(html 파일 수정)

![](images/Making_English_Chatbot_with_Django(4)/KakaoTalk_20220504_115038979.png)

![](images/Making_English_Chatbot_with_Django(4)/KakaoTalk_20220504_135613912.png)

---

**code: ‣[https://github.com/jmj3047/faq_chatbot_example.git](https://github.com/jmj3047/faq_chatbot_example.git)

- Reference: [https://cholol.tistory.com/478](https://cholol.tistory.com/478)