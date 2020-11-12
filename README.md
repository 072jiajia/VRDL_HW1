# Selected Topics in Visual Recognition using Deep Learning Homework 1
Code for (???) place solution in CS_T0828_HW1

## Abstract
In this work, I use API-Net to train my model<br>
API-Net [Paper](https://arxiv.org/pdf/2002.10191.pdf) | [GitHub](https://github.com/PeiqinZhuang/API-Net)

I trained K models for K different splits of training/validation data.<br>
K-Fold [簡介](https://medium.com/@chih.sheng.huang821/%E4%BA%A4%E5%8F%89%E9%A9%97%E8%AD%89-cross-validation-cv-3b2c714b18db) | [summary](https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833)

In testing phase, I computed K logits by the K different models.<br>
Add up the logits and choose the category with the largest value.

- The submission with the highest score I made was obtained from 5 models with the best validation accuracy.
- In fact, it's just 0.02% better than the submission obtained from all the models.

## Hardware
The following specs were used to create the solutions.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- 4x GeForce RTX 2080 Ti

## Reproducing Submission
To reproduct submission, do the following steps:
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
You need to download the zip file "cs-t0828-2020-hw1.zip" by yourself.<br>
And put the zip file into the same directory as main.py, the directory is structured as:
```
VRDL_HW1
  +- datasets
  +- model
  +- utils
  +- get_answer.py
  +- main.py
  +- train.py
  +- cs-t0828-2020-hw1.zip
```

## Train Models
You can simply run the following command to train your models and make submission.
```
$ python main.py
```
If you'd like to train in custom hyperparameters, change the hyperparameters to whatever you like.<br>
Or you may try the following command.
```
$ python main.py --exp_name=custom --epochs==50 --KFold=3 --n_classes=20 --n_samples=3
```

You may interrupt your program at any time.<br>
(ex: You're sharing the GPUs with your classmates and they think that you're using too many GPUs or you have occupied them for too long.)

This code records the checkpoint in every epoch, so you may just input the same command line to resume the code.<br>
The expected training time is:

GPUs | KFold | Image size | Training Epochs | Training Time
------------- | ------------- | ------------- | ------------- | -------------
4x 2080 Ti | 10 | 320 | 100 | 19 hours


## Make Submission
In main.py, after we finish training K models, it will directly call
```
python3 get_answer.py (exp_name)
```
It will generate a file (exp_name).csv which is the prediction of the testing dataset<br>
Use the csv file to make your submission!

