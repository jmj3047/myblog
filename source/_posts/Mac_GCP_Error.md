---
title: Mac VScode GCP 인증 관련 오류
date: 2023-03-09
categories:
  - Data Base
  - GCP
tags: 
  - GCP
  - Auth
  - Error
---

- gcp내에 있는 예제들을 실행 시킬때면 주피터 노트북으로 gcp를 사용할때 사용자를 인증해야 하는 이슈가 생김
    
    ```
    Error google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials. Please set GOOGLE_APPLICATION_CREDENTIALS or explicitly create credentials and re-run the application.
    ```
    
- mac에서 방법을 찾다가 오류를 해결함
- 이 [노트북](https://github.com/GoogleCloudPlatform/analytics-componentized-patterns/blob/master/retail/clustering/bqml/bqml_scaled_clustering.ipynb)의 아래 코드(!gcloud~)를 실행하다가 오류가 남
    
    ![ ](images/Mac_GCP_Error/Untitled.png)
    
- 이 [게시물](https://jmj3047.github.io/2023/03/03/Kaggle_GCP/)에서 **‘GCP에 데이터 세트 만들고 서비스 계정 생성하기’** 항목의 서비스 계정을 생성하고 json파일을 다운 받아 주었다.
- 그리고 코드를 수정하여 이렇게 작성했더니 주피터 노트북에서도 bq명령어나 gcloud명령어가 잘 돌아갔다..!
    
    ![ ](images/Mac_GCP_Error/Untitled%201.png)
    

- 문제는 m1 때문인거 같은데 이거 때문에 vscode까지 지우고 다시 깔았다. 아마 구글이 사용자 인증하는 서비스를 macOS에서는 제공하지 않아서 그런거 같은데.. 물리적으로 json 파일을 저장하지 않고도 credential 사용해서 바로 연결 할수 있는 방법을 찾았다면 추후에 포스팅 하겠다.