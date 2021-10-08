# klue-level2-nlp-16

## Table of Contents
  1. [Project Overview](#Project-Overview)
  2. [Getting Started](#Getting-Started)
  3. [Hardware](#Hardware)
  3. [Code Structure](#Code-Structure)
  4. [Detail](#Detail)

## Project Overview
  * 목표
    - 문장 속에서 단어(Entity)에 대한 속성과 관계를 예측하는 관계 추출(Relation Extraction)
  * 모델
    - klue/roberta-large 
  * Data
    - KLUE data
    - column : sentence, subject_entity, object_entity, relation(label)
  * Result
    - Public score : micro_f1 70.465 auprc 76.309
    - Private score : micro_f1 69.182 auprc 77.534 
  * Contributors
    * 김아경([github](https://github.com/EP000)): Exploratory Data Analysis, Stop Words
    * 김현욱([github](https://github.com/powerwook)): Hyperparameter tuning 및 MLOps 조사
    * 김황대([github](https://github.com/kimhwangdae)): Model testing, Baseline code refactoring, Speical token 추가 및 testing, Hyperparameter tuning
    * 박상류([github](https://github.com/psrpsj)): Data Augmentation 구현 및 testing
    * 사공진([github](https://github.com/tkrhdwls)): baseline 코드 분석 및 실험
    * 정재현([github](https://github.com/JHyunJung)): 프로젝트 Git 환경 설정
    * 최윤성([github](https://github.com/choi-yunsung)): Back Translation, Input Sequence Alteration, Custom Loss, soruce_based multi-model training, WandB연동

## Getting Started
  * Install requirements
    ``` bash
      # AEDA를 사용하기 위한 jdk 설치
      apt install default-jdk
      
      # requirement 설치
      pip install -r requirements.txt 
    ```
  * Train model
    ``` bash
      # 사전 argument arg.txt에 다음과 같은 형식으로 입력.
      # 아래는 require argument만 입력되어있고, 다른 hyperparameter 또한 설정 가능.
      --project_name [Project Name] --model_name [Model Name] --run_name [Run Name] --input_style [Input style(baseline, relation_token, daum)]

      # arg.txt 저장 후 shell script 실행.
      ./run.sh
    ```
  * Inference Model
    ```bash
      python inference.py --model_name [Model Name] --input_style [Input style(baseline, relation_token, daum)]
    ```
## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Code Structure
```text
├── code/                   
│   ├── augmentation.py                         # augmentations
│   ├── inference.py
│   ├── load_data.py
│   ├── loss.py                                 # customized loss functions
│   ├── model.py                                # define or load models
│   ├── tokenizer.py                            # load tokenizer and tokenized dataset
│   ├── train.py                    
│   ├── trainer.py                              # customized trainer for customized loss
│   ├── utils.py                                # utilities
│   ├── args.txt                                # arguments for training
│   ├── run.sh                                  # shell script for train.py
│   ├── dict_label_to_num.pkl
│   └── dict_num_to_label.pkl
│
└── dataset/                     
    ├── backtranslation/                        # back translation dataset
    │   ├── bt_realation_dataset.csv
    │   ├── bt_under_500_dataset.csv
    │   └── bt_under_1000_dataset.csv
    ├── stopwords/                              # stopwords dataset
    │   ├── StopWords_klue_bert-base_5.txt
    │   ├── StopWords_klue_roberta-base_5.txt
    │   └── StopWords_roberta-s5.txt
    ├── test/
    │   └── test_data.csv
    └── train/
        └── train.csv
```
## Detail
  * Data Augmentation Methods
    * Backtranslation
      * 문장을 다른 언어(영어)로 번역한 뒤 다시 원래 언어(한국어)로 번역하여 문장의 뉘앙스만 변경시킨 데이터를 생성(Papago번역, Pororo라이브러리 활용)
    * Easy Data Augmentation
      * Random Swap: 문장 중 임의의 두 단어 위치를 바꾸어 문장 생성
      * Random Delete: 문장 중 subject 와 object를 제외한 단어를 지운 문장 생성
    * A Easier Data Augmentation(AEDA): 문장 중간간에 랜덤하게 문장부호를 넣어 문장을 생성
    * Switch Subject and Object: Subject와 Object가 변경 가능한 관계는 변경한 문장을 생성
    * Subject, Object Dictionary: 각 Subject와 Object의 entity를 key로한 dictionary 제작 후, 각 entity에 맞는 임의의 단어로 변경한 문장을 생성


