---
title: Gen App Builder
date: 2023-04-18
categories:
  - Data Platform/Base
  - GCP
tags: 
  - Gen App Builder
  - AI tool
---

## 개요

- 구글에서 새롭게 발표한 gen app builder에 대해서 알아보자
- 어떤 기능이 있고 official doc이 발표된 게 많이 없지만 있는걸 최대한 활용해서 설명한다(유투브에 나온 데모 설명 포함)

## 웹사이트 설명

- 오피셜 문서는 아니지만 사이트에 간단하게 설명이 잘 되어 있어서 첨부.
- Gen App Builder의 도움으로 [기계 학습](https://www.analyticsvidhya.com/machine-learning/) 경험이 없는 개발자도 엔터프라이즈급 생성 AI 앱을 만들 수 있습니다. 이 강력한 도구는 [코드 없는](https://www.analyticsvidhya.com/blog/2023/03/low-code-no-code-development/) 접근 방식을 제공하여 개발자가 몇 분 또는 몇 시간 내에 고품질 환경을 구축할 수 있도록 합니다. 주요 기능은 다음과 같습니다.
1. **코딩 필요 없음:** Gen App Builder를 사용하면 사용자는 코드를 작성하지 않고도 맞춤형 비즈니스 애플리케이션을 만들 수 있습니다. 이 플랫폼은 시각적 인터페이스와 드래그 앤 드롭 기능을 제공하여 사용자가 [애플리케이션을 쉽게 설계하고 구축](https://www.analyticsvidhya.com/blog/2021/12/data-science-app-with-python/) 할 수 있도록 합니다 .
2. **다른 Google 클라우드 서비스와의 통합:** Gen App Builder는 ****[Cloud SQL](https://www.analyticsvidhya.com/blog/2022/07/managing-sql-database-on-google-cloud/) 및 Cloud Storage 와 같은 다른 Google Cloud 서비스와 통합되어 데이터를 쉽게 저장하고 관리할 수 있습니다.
3. **사용자 지정 가능한 템플릿:** 이 플랫폼은 사용자가 애플리케이션을 빠르고 쉽게 만들 수 있도록 사용자 지정 가능한 템플릿을 제공합니다. 템플릿은 고객 관계 관리, 인사 및 프로젝트 관리를 비롯한 다양한 사용 사례에 사용할 수 있습니다.
4. **실시간 협업:** 이 App Builder를 사용하면 사용자가 애플리케이션에서 실시간으로 협업할 수 있습니다. 이를 통해 팀이 쉽게 협력하고 피드백을 공유할 수 있습니다.
5. **확장성:** Gen App Builder로 구축된 애플리케이션은 확장성이 뛰어나고 많은 양의 데이터와 사용자를 처리할 수 있습니다.
6. **보안:** Google Cloud의 보안 기능은 App Builder에 내장되어 있어 ****[보안 위협](https://www.analyticsvidhya.com/blog/2021/01/security-threats-to-machine-learning-systems/)으로부터 애플리케이션과 데이터를 보호합니다 .

## 구글 오피셜 문서

- 구글 오피셜 문서에 있는 설명을 간략하게 추린 것이다.

#### ****생성형 앱 빌더를 사용한 새로운 생성형 AI 기반 검색 및 대화형 환경 빌드****

- 생성형 앱 빌더를 사용하여 머신러닝 기술을 활용한 엔터프라이즈급 생성형 AI 애플리케이션 개발
- 빠르게 고품질 환경을 구축하고 애플리케이션 및 웹사이트에 통합
- 기반 모델과 정보 검색을 결합해 관련성 높은 맞춤 정보 제공
- 텍스트, 이미지, 기타 미디어로 응답할 수 있는 멀티모달 앱 구축
- 서드 파티 앱 및 서비스와 연결하고 트랜잭션 지원 기능 제공

#### ****차세대 대화형 AI 경험 및 어시스턴트****

- 생성형 앱 빌더를 사용해 원활한 대화형 방식으로 기업 데이터 세트를 활용하고 사용자 경험 개선
- AI 기반 앱으로 모든 소스의 정보 종합 및 구체적인 실행 가능한 답변 제공
- 고객 서비스 분야에서 수익, 고객 만족도, 충성도 향상 도모
- 멀티모달 기능과 대화형 UI 트랜잭션을 사용하여 제품 구매 과정 돕기
- 다양한 산업 및 사용 사례에 적용 가능, 예: 소비재, 공공 서비스, 금융, 인트라넷 등

#### ****Google 품질 수준의 검색과 기반 모델의 결합****

- 조직 전체 데이터에서 올바른 정보 찾기 중요, 기존 도구로는 어려움
- 생성형 앱 빌더로 Google 품질 검색 기능과 생성형 AI 결합하여 관련성 높은 맞춤 정보 찾기 지원
- 코딩 경험 없이 몇 분~시간 만에 대화형 검색 환경 구축 가능
- 멀티모달 검색 환경을 통해 텍스트, 이미지, 동영상 검색 지원
- 자연스럽게 인용 제공 및 맞춤 결과 지원으로 사용자 만족도 향상
- 데이터 사용률 증가, 프로세스 효율성 개선, 직원과 고객 만족도 높이기
- 다양한 소스의 복잡한 데이터와 상호작용 능력으로 고객 서비스 개선 가능
- 대화형 처리 기능과 결합하여 고객 참여도, 직원 생산성 개선 가능
- 개발자와 기업이 새로운 경험과 수익 기회를 실현할 것으로 기대

## 유투브 데모 설명

- 상황: 투자 전략을 수립하기 위해 반도체 시장을 평가해야 하는 애널리스트, 엔지니어들이 회사 홈페이지에 gen app builder를 사용하여서 검색 엔진을 만들어줌
- 세계적으로 반도체 부족 현상이 일어남 → 어떤 산업에서 가장 영향을 많이 미쳤는지를 검색 엔진으로 물어봄
- 결과: internal, external이 표시되어서 회사가 제공한 데이터냐 아니냐에 따라서 결과가 나옴, 각각 결과마다 ai 엔진이 자동적으로 요약해서 보여줌
    
    ![ ](images/Gen_App_Builder/Untitled.png)
    
- 셋중에 하나를 더 알고 싶다면 클릭한다. 그럼 글들을 자동적으로 정리해서 목차를 만들어줌
    
    ![ ](images/Gen_App_Builder/Untitled%201.png)
    
- 기업에서 구독하고 있는 금융 저널들에 중요한 정보가 담겨 있음. 그래서 앱 설정해서 이 정보들을 포함해 달라고 설정 가능, 그리고 찾은 정보들을 전부 합치고 정리해서 보여달라고도 설정 가능
    
    ![ ](images/Gen_App_Builder/Untitled%202.png)
    
- 이렇게 정보를 추가 한 후 interest rate에 대해서 follow-up 질문을 함
- 검색한 정보와 함께 검색 overall summary를 같이 보여줌
    
    ![ ](images/Gen_App_Builder/Untitled%203.png)
    
- 두 질문으로 산업군의 이해가 높아짐
- 이 모든것을 회사 url, 데이터를 cloudstorage에 올려서 (구글은 이 정보를 회사 허가 하에만 사용) search로 할건지 chat으로 할건지를 선택한 다음 만들면 끝이남 → 서치 엔진을을 쉽게 만들어줌
    
    ![ ](images/Gen_App_Builder/Untitled%204.png)
    
- 만들고 나서 커스터마이징도 가능 → 하단 네비게이션 바에 있는 것들을 우리가 조정해서 사용할 수 있음
    
    ![ ](images/Gen_App_Builder/Untitled%205.png)
    
- deploy 코드를 따로 배포 → 회사 html에 삽입하면 서치 엔진이 됨, api로도 제공
    
    ![ ](images/Gen_App_Builder/Untitled%206.png)
    

---

- Reference
    - [https://www.analyticsvidhya.com/blog/2023/04/gen-app-builder-google-clouds-latest-generative-ai-tools/](https://www.analyticsvidhya.com/blog/2023/04/gen-app-builder-google-clouds-latest-generative-ai-tools/)
    - [https://cloud.google.com/blog/ko/products/ai-machine-learning/create-generative-apps-in-minutes-with-gen-app-builder?_ga=2.206347863.-517135162.1681093488&hl=ko](https://cloud.google.com/blog/ko/products/ai-machine-learning/create-generative-apps-in-minutes-with-gen-app-builder?_ga=2.206347863.-517135162.1681093488&hl=ko)
    - [https://www.youtube.com/watch?v=kOmG83wGfTs](https://www.youtube.com/watch?v=kOmG83wGfTs)