# Selected Topics in Visual Recognition using Deep Learning Homework 1
Code for (???) place solution in CS_T0828_HW1

## Abstract
In this work, I use API-Net to train my model

API-Net [paper](https://arxiv.org/pdf/2002.10191.pdf) | [github](https://github.com/PeiqinZhuang/API-Net)

I train K models for K different splits of training/testing data.

K-Fold [簡介](https://medium.com/@chih.sheng.huang821/%E4%BA%A4%E5%8F%89%E9%A9%97%E8%AD%89-cross-validation-cv-3b2c714b18db) | [summary](https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833)

In testing phase, I compute K logits by the K different models.

Add up the logits and choose the category with largest value.

## Hardware
The following specs were used to create the original solutions.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- 4x GeForce RTX 2080 Ti

## Reproducing Submission
To reproduct my submission, do the following steps:
1. [Installation](#installation)
2. [Prepare Data](#dataset-preparation)
3. [Train](#train-models)
4. [Make Submission](#make-submission)

## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is recommended.
```
virtualenv .
source bin/activate
pip3 install -r requirements.txt
```

## Dataset Preparation
You need to download the zip file "cs-t0828-2020-hw1.zip" by yourself.

And put the zip file into the same directory as main.py, the directory is structured as:
```
VRDL_HW1
  +- datasets
  +- model
  +- utils
  +- cs-t0828-2020-hw1.zip
  +- get_answer.py
  +- main.py
  +- train.py
```

## Train Models
You can simply run the following command to train models and make submission.
```
$ python main.py
```
If you'd like to train in custom hyperparameters, change the hyperparameters to whatever you like.

Or you may try the following command.
```
$ python main.py --exp_name=custom --epochs==50 --KFold=3 --n_classes=20 --n_samples=3
```

You may interrupt your program at any time.

(for example: you're sharing the GPU with your classmates and they think that you use too many GPUs.)

This code saves the checkpoint in every epoch, so you may just input the same command line to resume the code.

The expected training time is:

GPUs | KFold | Image size | Training Epochs | Training Time
------------- | ------------- | ------------- | ------------- | -------------
4x 2080 Ti | 10 | 320 | 100 | 15 hours


## Make Submission
In main.py, after we finish training K models, it will directly call
```
python3 get_answer.py (exp_name)
```
It will generate a file (exp_name).csv which is the prediction of the testing dataset

Use the csv file to make your submission.

