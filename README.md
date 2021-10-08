# klue-level2-nlp-16

## Table of Contents
  1. [Project Overview](#Project-Overview)
  2. [Getting Started](#Getting-Started)
  3. [Hardware](#Hardware)
  3. [Code Structure](#Code-Structure)
  4. [Detail](#Detail)

## Project Overview
  * 목표
  * 모델
  * Data
  * Result
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

## Detail


