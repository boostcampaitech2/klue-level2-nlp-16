# klue-level2-nlp-16

## Table of Contents
  1. [Project Overview](#Project-Overview)
  2. [Getting Started](#Getting-Started)
  3. [Code Structure](#Code-Structure)
  4. [Detail](#Detail)

## Project Overview
  * 목표
  * 모델
  * Data
  * Result
  * Contributors
    * 김아경([github](https://github.com/EP000))
    * 김현욱([github](https://github.com/powerwook))
    * 김황대([github](https://github.com/kimhwangdae))
    * 박상류([github](https://github.com/psrpsj))
    * 사공진([github](https://github.com/tkrhdwls))
    * 정재현([github](https://github.com/JHyunJung))
    * 최윤성([github](https://github.com/choi-yunsung))

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
      # 사전 argument arg.txt에 다음과 같은 형식으로 입력
      # 아래는 require argument만 입력되어있고, 다른 하이퍼피라미터도 제어 가능
      --project_name [Project Name] --model_name [Model Name] --run_name [Run Name] --input_style [Input style(baseline, relation_token, daum)]

      # arg.txt 저장 후 shell script 실행
      ./run.sh
    ```
  * Inference Model
    ```bash
      python inference.py --model_name [Model Name] --input_style [Input style(baseline, relation_token, daum)]
    ```

## Code Structure

## Detail


