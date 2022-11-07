---
title: Multi-Task Learning for Voice Trigger Detectionn
date: 2022-11-02
categories:
  - Paper
  - Multi-Task Learning
tags: 
  - Multi-Task Learning
  - Voice Trigger Detection
  - English
---

Journal/Conference : ICASSP IEEE
Year(published year): 2020
Author: Siddharth Sigtia, Pascal Clark, Rob Haynes, Hywel Richards, John Bridle
Subject: Multi-Task Learning

# Multi-Task Learning for Voice Trigger Detection

> Summary
- We start by training a general acoustic model that produces phonetic transcriptions given a large labelled training dataset.
- Next, we collect a much smaller dataset of examples that are challenging for the baseline system.
- We then use multi-task learning to train a model to simultaneously produce accurate phonetic transcriptions on the larger dataset and discriminate between true and easily confusable examples using the smaller dataset.

## Introduction

Significant challenge is that unlike automatic speech recognition (ASR) systems, collecting training examples for a specific keyword or phrase in a variety of conditions is a difficult problem

![](images/MTL_for_VTD/Untitled.png)

In the literature, the problem of detecting a speech trigger phrase is interchangeably referred to as voice trigger detection [3], keyword spotting [4], wake-up word detection [5] or hotword detection [6]. In the rest of this paper, we refer to this problem as voice trigger detection.

In the multi-stage approach (Figure 1), the first stage comprises a low-power DNN-HMM system that is always on [3].

In this design, it is the second stage that determines the final accuracy of the system and the models used in this stage are the subject of this paper.

Our main contribution is to propose a multi-task learning strategy where a single model is trained to optimise 2 objectives  simultaneously. The first objective is to assign the highest score to the correct sequence of phonetic labels given a speech recording.

This objective is optimised on a large labelled training dataset which is also used for training the main speech recogniser and is therefore easy to obtain. The second objective is to discriminate between utterances that contain the trigger phrase and those that are phonetically similar and easily confusable.

## Baseline

The baseline model architecture comprises an acoustic model (AM) with four bidirectional LSTM layers with 256 units each, followed by an output affine transformation + softmax layer over context independent (CI) phonemes, word and sentence boundaries, resulting in 53 output symbols (Figure 2).

![](images/MTL_for_VTD/Untitled%201.png)

Firstly, the fact that the second-pass model is used for re-scoring and not in a continuous streaming setting allows us to use bidirectional LSTM layers. Secondly, using context-independent phones as targets allows us to share training data with the main ASR.

This is particularly important since in many cases it is not possible to obtain a large number of training utterances with the trigger phrase, for example when developing a trigger detector for a new language. Furthermore, having CI phones as targets results in a flexible model that can be used for detecting any keyword.

Given an audio segment x from the first pass, we are interested in calculating the probability of the phone sequence in the trigger phrase, P(TriggerPhrasePhoneSeq|x).

## Multi-Task Learning

The question we really want to answer is, “given an audio segment from the first pass, does it contain the trigger phrase or not?”

We would like the second-pass model to be a binary classifier which determines the presence or absence of the trigger phrase.

However the issue with this design is that collecting a large number of training examples that result
in false detections by the baseline system is a difficult problem (c.f. Section 4). Furthermore, the second pass models have millions of parameters, so they can easily overfit a small training set resulting in poor generalisation. Therefore, we are faced with the choice between a more general phonetic AM that can be trained on a large, readily available dataset but is optimised for the wrong criterion or a trigger phrase specific detector that is trained on the correct criterion but with a significantly smaller training dataset.

One solution to this problem is to use multi-task learning (MTL) [19]

Note that predicting the sequence of phonetic labels in an utterance and deciding whether an utterance contains a specific trigger phrase or not, are related tasks.

We train a single network with a stack of shared/tied biLSTM layers with two seperate output layers (one for each task) and train the network jointly on both sets of training data (Figure 2).

We hypothesise that the joint network is able to learn useful features from both tasks: a) the network can be trained to predict phone labels on a large labelled dataset of general speech which covers a wide distribution of complex acoustic conditions, b) the same network can also learn to discriminate between examples of true triggers and confusable examples on a relatively smaller dataset. An alternative view of this process is that the phonetic transcription task with a significantly larger training set acts as a regulariser for the trigger phrase discrimination task with a much smaller dataset.

The objective function for the phrase specific/discriminative output layer is defined as follows: the softmax output layer contains two output units, one for the trigger phrase and the other one for the blank symbol used by the CTC loss function [8, 16]

## Evaluation

There were 100 subjects, approximately balanced between male and female adults. Distances from the device were controlled, ranging from 8 to 15 feet away. There are over 13K utterances overall, evenly divided between four acoustic conditions: (a) quiet room, (b) external noise from a TV or kitchen appliance in the room, (c) music playback from the recording device at medium volume, and (d) music playback from the recording device at loud volume.

These examples are used to measure the proportion of false rejections (FRs).

In addition to these recordings, this test set also consists of almost 2,000 hours of continuous audio recordings from TV, radio, and podcasts. This allows the measurement of the false-alarm (FA) rate in terms of FAs per hour of active external audio.

The second test set is an unstructured data collection at home by our employees, designed to be more representative of realistic, spontaneous usage of the smart speaker. 

With this data, it is possible to measure nearly unbiased false-reject and false-alarm rates for realistic in-home scenarios similar to customer usage.

![](images/MTL_for_VTD/Untitled%202.png)

We use detectionerror trade-off (DET) curves to compare the accuracy between models. Each curve displays the FA rate and the proportion of FRs associated with sweeping the trigger threshold for a
particular model.

In practice, we compare the shapes of the DET curves for different models in the vicinity of viable operating points. We compare five models: the baseline phonetic CTC model trained on the ASR dataset (blue), the baseline phrase specific model trained on the much smaller training set with randomly initialised weights (red), the same phrase specific model but with weights initialised with the learned weights from the baseline phonetic CTC model (yellow), the phonetic (purple) and phrase specific (green) branches of the proposed MTL model.

Note that the phrase specific model  with weight initialisation from the baseline phonetic model
(yellow) is effectively trained using both datasets.

In both test sets, the MTL phonetic (purple) and phrase-specific (green) models outperform the baseline phonetic CTC (blue), reducing the FR rate by almost half at many points along the curve.

The non-MTL phrase specific models (red and yellow) yield significantly worse accuracies
in comparison, which is unsurprising given that the training dataset is two orders of magnitude smaller compared to the phonetic baseline (blue).

Comparing the structured data evaluation (left) and the take-home data evaluation (right), it is
also striking how the error rates are generally much higher for the latter.

## Conclusions

We trained the model to simultaneously produce phonetic transcriptions on a large ASR dataset and to discriminate between difficult examples on a much smaller trigger phrase specific training
set. We evaluate the proposed model on two challenging test sets and find the proposed method is able to almost halve errors and does not require any extra model parameters.

---
- Link: **[Multi-Task Learning for Voice Trigger Detection](https://arxiv.org/abs/2001.09519)**