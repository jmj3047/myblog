---
title: A Comprehensive Survey on EEG-Based Emotion Recognition, A Graph-Based Perspective
date: 2026-03-13
categories:
  - Paper
  - EEG
tags: 
  - EEG Emotion Recognition
  - Graph
---

Journal/Conference : arXiv preprint arXiv:2408.06027.
Year(published year): 2024
Author: Liu, C., Zhou, X., Wu, Y., Ding, Y., Zhai, L., Wang, K., Jia, Z. and Liu, Y.
Subject: EEG, Emotion Recognition, Graph

## Abstract

본 논문은 기능적 뇌 신호(Functional Brain Signals) 분야에서 DG와 DA를 다룬 최초의 체계적인 리뷰 논문으로, 관련 방법론, 수행 과제(Tasks), 그리고 데이터셋을 분류하고 논의합니다. 또한, 이 분야의 향후 연구 방향에 대해서도 제언합니다

## 통합 프레임워크 및 분류 체계 (Unified Framework & Taxonomy)

이 논문은 그래프 기반 모델 구축 시 고려해야 할 핵심 질문 4가지를 바탕으로 분류 체계를 제안합니다.

핵심 질문: (1) 어떤 EEG 특징을 선택할 것인가? (2) 뇌 영역 간 연결성(인접 행렬)을 어떻게 계산할 것인가? (3) 노드 정보를 어떻게 집계할 것인가? (4) 어떤 네트워크 수준의 구조 조작을 채택할 것인가?

계층 구조: 위 질문들에 대응하여 특징 선택 → 에지 계산 → 노드 집계 → 그래프 조작의 4단계 계층 구조를 가집니다.

![ ](./images/EEG_GNN_Survey/image.png)

## 특징 선택 (Feature Selection - 노드 설정)

그래프의 각 노드는 EEG 채널(전극)을 의미하며, 해당 노드에 할당될 특징을 정의합니다.

- 시간적 노드 (Temporal Nodes):
    - 생신호(Raw Signal): 필터링만 거친 원본 EEG 신호를 직접 입력으로 사용합니다.
    - 장점: 이벤트 관련 전위(ERP)와 같은 진폭 및 파형의 일시적 변화를 반영하기에 유리하며, ECG, GSR 등 다른 생체 신호와 동기화하여 분석할 때 확장성이 높습니다.
    - 단점
        - 시간적 노드는 원본 신호나 시간 관련 특징을 직접 사용하기 때문에 다음과 같은 한계가 있습니다.
        - 일시적 노이즈에 취약함: 눈 깜빡임(eye movements)과 같은 일시적인 잡음(transient noise)에 더 민감하게 반응하여 신호가 오염될 가능성이 높습니다.
        - 상대적으로 불안정한 표현: 주파수 노드에 비해 감정 상태를 나타내는 신호의 안정성이 떨어질 수 있습니다
- 주파수 노드 (Frequency Nodes):
    - 핵심 개념: 특정 주파수 대역(delta, theta, alpha, beta, gamma)과 감정 간의 직접적인 상관관계를 활용합니다.
    - 대표 특징: 미분 엔트로피(DE)가 가장 널리 쓰이며, 이 외에도 전력 스펙트럼 밀도(PSD), 미분 비대칭(DASM), 합리적 비대칭(RASM) 등이 사용됩니다.
    - 장점: 눈 깜빡임과 같은 일시적 노이즈에 강하며 감정 상태의 안정적인 표현을 제공합니다.
    - 단점:
        - 일시적 변화 포착의 어려움: 진폭이나 파형의 급격한 변화와 같은 이벤트 관련 전위(ERP) 특성을 반영하는 데 한계가 있습니다.
        - 이는 감정 활동 중에 나타나는 뇌파의 중요한 순간적 변화를 놓칠 수 있음을 의미합니다.
        - 데이터셋 의존성: 현재 주파수 노드가 많이 쓰이는 이유는 기술적 우월성 때문이라기보다, SEED와 같은 대중적인 데이터셋이 미분 엔트로피(DE) 특징을 주로 제공하는 환경적 요인이 큽니다. 즉, 특정 데이터 형식에 최적화되어 있어 범용적인 설계라고 보기 어려울 수 있습니다.