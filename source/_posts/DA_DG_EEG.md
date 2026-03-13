---
title: Domain Adaptation and Generalization of Functional Medical Data, A Systematic Survey of Brain Data
date: 2026-03-13
categories:
  - Paper
  - EEG
tags: 
  - EEG Emotion Recognition
  - Domain Adaptation
  - Domain Generalization
  - Functional medical data
---

Journal/Conference : ACM Computing Surveys, 56(10), pp.1-39.
Year(published year): 2024
Author: Sarafraz, G., Behnamnia, A., Hosseinzadeh, M., Balapour, A., Meghrazi, A. and Rabiee, H.R.
Subject: EEG Emotrion Recognition, Domain adaptation, domain generalization, functional medical data

## Abstract

머신러닝 알고리즘의 뛰어난 성능에도 불구하고, 훈련 데이터와 테스트 데이터의 분포가 다를 경우 성능이 크게 저하되는 문제가 있습니다. 의료 데이터 연구에서는 인간의 건강과 직결된다는 점, 고가의 장비 활용, 세밀한 실험 설정 등으로 인해 이 문제가 더욱 심각합니다. 따라서 데이터 분포 변화(distribution shifts) 상황에서도 성능을 유지하는 도메인 일반화(DG)와 도메인 적응(DA) 기술을 확보하는 것은 의료 데이터 분석에서 필수적인 단계입니다.

본 논문은 기능적 뇌 신호(Functional Brain Signals) 분야에서 DG와 DA를 다룬 최초의 체계적인 리뷰 논문으로, 관련 방법론, 수행 과제(Tasks), 그리고 데이터셋을 분류하고 논의합니다. 또한, 이 분야의 향후 연구 방향에 대해서도 제언합니다

## Introduction

문제 제기: 학습 데이터와 테스트 데이터의 분포가 다를 때 머신러닝 성능이 저하되는 '도메인 시프트(Domain Shift)' 문제를 지적합니다.

의료 데이터의 특수성: 뇌 데이터는 장비의 차이, 피험자의 상태 등에 민감하여 일반화가 어렵지만, 건강과 직결되기에 높은 정확도가 필수적임을 강조합니다.

본 연구의 목적: EEG 및 fMRI 데이터를 중심으로 최신 DA/DG 기법들을 체계적으로 분류하고 비교하는 최초의 리뷰임을 밝힙니다.

## Background

- Domain Adaptation: 타겟 도메인 데이터를 어느정도 갖고 있는 상태에서 실험하는 것
- Domain Generalization: 타겟 도메인을 전혀 볼수 없는 상태에서 여러개의 소스 도메인으로부터 도베인에 불변하는 특징을 학습하여 처음보는 새로운 도메인에서도 잘 작동하는 모델을 만드는 것

## DA and DG for Medical Data Analysis

- Cross-subject (피험자 간) 과제: 의료 데이터 분석에서 가장 흔한 과제로, 사람마다 다른 뇌 구조나 생리적 특성으로 발생하는 데이터 변동성을 제거하는 것이 목표입니다.
- Cross-dataset (데이터셋 간) 과제: 서로 다른 연구 기관이나 병원에서 수집된 데이터셋 사이의 도메인 시프트를 해결하려는 시도입니다.
- Cross-session (세션 간) 과제: 동일한 피험자라도 측정 시간이나 날짜(Cross-day)에 따라 신호가 달라지는 '내적 변동성(Intra-subject variability)'을 다룹니다.
- Cross-device (장비 간) 과제: 뇌 신호를 측정할 때 사용된 서로 다른 하드웨어 장비로 인해 발생하는 데이터의 차이를 고려합니다.

의료 현장의 특수성에 따라 두 기술은 서로 다른 응용 가치를 가집니다.

- 일반화(DG)의 필수성: 의료 모델이 실용적이려면 본 적 없는 새로운 환자(Unseen data)에게도 잘 작동해야 합니다. 매번 새로운 환자가 올 때마다 모델을 다시 학습시키는 것은 시간과 비용 측면에서 매우 비효율적이기 때문입니다.
- 적응(DA)의 역할: 특정 기관의 장비로 측정된 데이터나 특정 환자군에 맞춰 모델을 최적화해야 할 때는 DA가 핵심적인 역할을 합니다.
- 결정적 차이 (타겟 데이터 접근성):
    - DA: 학습 과정에서 타겟 데이터의 구조나 특성을 활용하여 소스 지식을 타겟에 맞게 변형합니다.
    - DG: 소스 데이터의 지식만을 확장하여 어떤 구조를 가진 도메인이 들어와도 작동할 수 있는 강건한 모델을 제안합니다.
- 혼합적 접근: 최근 연구들은 기본적으로 적응(DA)을 수행하면서도 일반화(DG) 아이디어를 결합하여 모델의 신뢰도를 높이는 경향을 보입니다.

## Methods

### DA Approaches

#### Alignment

4.1.1 정렬 (Alignment)
소스와 타겟 간의 입력이나 특징 분포를 일치시켜 성능 저하를 막는 가장 보편적인 방법입니다.

적대적 특징 정렬 (Adversarial Feature Alignment): 도메인 분류기가 소스와 타겟을 구분하지 못하도록 특징 추출기를 속이는 방식으로 공통 특징을 학습합니다. DANN(Domain Adversarial Neural Network)과 GRL(Gradient Reversal Layer)이 대표적입니다.

도메인 정렬 (Domain Alignment): MMD(Maximum Mean Discrepancy)나 TCA(Transfer Component Analysis) 같은 통계적 지표를 활용해 소스와 타겟의 분포 차이를 수학적으로 최소화합니다.

인스턴스 정렬 (Instance Alignment): 샘플 단위로 최적 운송(Optimal Transport) 이론을 적용하거나 특징 유사도에 따라 가중치를 부여해 소스와 타겟을 정렬합니다.

분류기 정렬 (Classifier Alignment): 여러 분류기의 예측값이 일관되도록 하거나, 분류 경계선 근처의 타겟 샘플을 정확히 분류하도록 예측 불일치성을 최소화합니다.

4.1.2 데이터 조작 (Data Manipulation)
데이터 전처리 단계에서 도메인 차이를 줄이는 기법입니다.

정규화: z-score 정규화 등을 통해 데이터의 스케일 차이를 줄여 조건부/한계 분포 시프트를 감소시킵니다.

피험자 클러스터링: 타겟과 유사한 감정 반응을 보이는 소스 피험자들만 선택적으로 학습에 활용하여 적응 효율을 높입니다.

4.1.3 특징 분리 (Feature Disentanglement)
입력을 도메인 고유 정보와 도메인 불변(과제 관련) 정보로 쪼개는 방식입니다.

뇌 신호는 여러 자극이 뒤섞인 복합체이므로, 과제와 관련된 반응만 골라내는 이 방식은 뇌 데이터 분석에 매우 직관적이고 효과적입니다.

4.1.4 의사 라벨 학습 (Pseudo-Label Training)
타겟 데이터에 라벨이 없을 때, 소스 모델의 예측값을 임시 라벨로 사용합니다.

반복적인 자기 학습(Self-training)을 통해 모델이 타겟 도메인의 특징을 스스로 반영하며 점진적으로 개선되도록 합니다.

#### DG Approaches

4.2.1 표현 학습 (Representation Learning)
추출된 특징 자체가 특정 도메인에 편향되지 않도록(Domain-invariant) 만드는 것이 핵심입니다.
적대적 학습: 여러 소스 도메인을 구분하지 못하게 함으로써 전역적인 특징을 학습합니다.
특징 가중치 부여: 뇌의 영역별 중요도에 따라 가중치를 부여해 더 일반화 가능한 특징을 강조합니다.
다중 소스 도메인 정렬: 여러 소스 간의 차이를 줄여 공통 공간(Shared space)을 찾습니다.

4.2.2 학습 시나리오 (Learning Scenarios)
알고리즘 설계 자체에서 일반화 능력을 끌어냅니다.
앙상블 학습 (Ensemble Learning): 여러 네트워크의 예측을 다수결 등으로 결합하여 안정성을 확보합니다.
메타 학습 (Meta-Learning): '학습하는 법을 학습'하여 소수 샘플만으로도 새로운 피험자에게 빠르게 적응할 수 있게 합니다.
자기지도 학습 (Self-supervised Learning): 대조 학습(Contrastive learning) 등을 통해 라벨 없이도 데이터의 내재적 패턴을 학습하여 전이 성능을 높입니다.

4.2.3 데이터 조작 (Data Manipulation)
데이터 증강 (Data Augmentation): 노이즈 추가, 신호 섞기, GAN 기반 생성 등을 통해 데이터 공간을 확장하여 모델의 경험치를 높입니다.
특징 선택 (Feature Selection): 그래프 풀링이나 뇌 지도(Atlas)를 활용해 불필요한 노이즈 영역을 제거하고 핵심 부위만 남깁니다.

4.2.4 아키텍처 내장 (Architecture Embedded)
일반화에 유리한 신경망 구조를 직접 설계합니다.
CNN 기반: 해석 가능한 CNN이나 분리형 채널 Attention을 사용해 시공간적 정보를 효율적으로 추출합니다.
그래프 구조 (GNN): 뇌의 연결성을 그래프로 모델링하여 피험자 간 변동성을 보상합니다.
특수 층: 적응형 배치 정규화(Adaptive BN) 등을 넣어 피험자 간 편향을 제거합니다.