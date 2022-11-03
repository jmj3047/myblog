---
title: Multi-Task Learning for Speaker Verification and Voice Trigger Detection
date: 2022-10-31
categories:
  - Paper
  - Multi-Task Learning
tags: 
  - Multi-Task Learning
  - Speaker Verification
  - Voice Trigger Detection
  - English
---

Journal/Conference : ICASSP IEEE
Year(published year): 2020
Author: Siddharth Sigtia, Erik Marchi, Sachin Kajarekar, Devang Naik, John Bridle
Subject: Multi-Task Learning

# Multi-Task Learning for Speaker Verification and Voice Trigger Detection

> Summary

- In this study, we investigate training a single network to perform automatic speech transcription and speaker recognition, both tasks jointly.
- We train the network in a supervised multi-task learning setup, where the speech transcription branch of the network is trained to minimise a phonetic connectionist temporal classification (CTC) loss while the speaker recognition branch of the network is trained to label the input sequence with the correct label for the speaker.
- Results demonstrate that the network is able to encode both phonetic and speaker information in its learnt representations while yielding accuracies at least as good as the baseline models for each task, with the same number of parameters as the independent models.

## Introduction

Voice trigger detection, which is interchangeably known as keyword spotting [4], wake-up word detection [5], or hotword detection [6], is treated as an acoustic modelling problem.

Their primary aim is to recognise the phonetic content (or the trigger phrase directly) in the input audio, with no regard for the identity of the speaker.

On the other hand, speaker verification systems aim to confirm the identity of the speaker by comparing an input utterance with a set of enrolment utterances which are collected when a user sets up their device.

Speaker verification algorithms can be characterised based on whether the phonetic content in the inputs is limited, which is known as text-dependent speaker verification [9].

We believe that knowledge of the speaker would help determine the phonetic content in the acoustic signal and vice versa, therefore estimating both properties is similar to solving simultaneous equations. 

In this study, the main research question we try to answer is “can a single network efficiently represent both phonetic and speaker specific information?”.

From a practical standpoint, being able to share computation between the two tasks can save on-device memory, computation time or latency and the amount of power/battery consumed.

More generally, we are interested in studying whether a single model can perform multiple speech understanding tasks rather than designing a separate model for each task.

We train a joint network to perform a phonetic labelling task and a speaker recognition task.

We evaluate the 2 branches of the model on a voice trigger detection task and a speaker verification task, respectively.

It it is possible for a single network to encode both speaker and phonetic information and yield similar accuracies as the baseline models without requiring any additional parameters.

## Voice Trigger Detection Baseline

We extract 40-dimensional log-filterbanks from the audio at 100 frame-per-second (FPS). At every step, 7 frames are spliced together to form symmetric windows and finally this sequence of windows is sub-sampled by a factor of 3, yielding a 280-dimensional input vector to the model at a rate of 33 FPS. 

The features are input to a stack of 4 bidirectional LSTM layers with 256 units in each layer (Figure 1). This is followed by a fully connected layer and an output softmax layer over context-independent phonemes and additional sentence and word boundary symbols, resulting in a total of 53 output symbols and 6 million model parameters. This model is then trained by minimising the CTC loss function [16].

The training data for this model is 5000 hours of anonymised audio data that is manually transcribed, where all of the recordings are sampled from intentional voice assistant invocations and are assumed to be near-field.

![Untitled](images/MTL_for_SV&VTD/Untitled.png)

Fig. 1. The left branch of the model represents the voice trigger detector, the right branch is the speaker verification model. Solid horizontal arrows represent layers with tied weights, dashed arrows represent layers with weights that may or may not be tied.

## Speaker Verification Baseline

We use a simple location-based attention mechanism [18] to summarise the encoder activations as a fixed-dimensional vector. 

We found the attention mechanism to be particularly effective in the text-independent setting.

During inference, given a test utterance x, the speaker embedding is obtained by removing the final softmax layer and using the 128-dimensional activations of the previous layer.

Each training utterance is of the form “Trigger phrase, payload” for e.g.“Hey Siri (HS), play me something I’d like”. For every training example, we generate 3 segments: the trigger phrase, the payload and the whole utterance. We found that breaking the utterances up this way results in models that generalise significantly better

## Multi Task Learning

Note that most of the weights in the two baseline systems are in modules with the same structure (biLSTM layers). 

Furthermore the model must also learn to represent both phonetic and speaker information using the same activations of the final layer of the encoder. The second model relaxes this constraint by sharing only 3 biLSTM layers in the encoder, with separate final biLSTM layers for the voice trigger and speaker recognition branches (7.6 million parameters). 

Finally, we train a third model where 2 biLSTM layers have tied weights, with 2 additional biLSTM layers for each branch (Figure 1).

## Evaluation

![](images/MTL_for_SV&VTD/Untitled%201.png)

## Conclustions

Our results demonstrate that sharing the first two layers of the model between the speaker and phonetic tasks gives accuracies that are as good as the individual baselines.

This result indicates that it is possible to share some of the lowlevel computation between speech processing tasks without hurting accuracies.

---
- Link: **[Multi-Task Learning for Speaker Verification and Voice Trigger Detection](https://arxiv.org/abs/2001.10816)**