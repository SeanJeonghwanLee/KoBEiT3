# BEiT3 based Korean VQA Model #

## Basic information ##
### Model ###
  - Base Model : beit3_large_indomain_patch16_224 (https://github.com/microsoft/unilm/tree/master/beit3)
    - best epoch : 8
    - learning rate : 2e-5
    - resources : rtx3090 * 4 (about 9h/epoch)
    - fixed seed : 42

  - Tokenizer
    - korean sentencepiece tokenizer trained on korean wikipedia
    
### Dataset ###
  - KoBEiT3
    - aihub 시각정보 기반 질의응답 (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=104)
      * Only Korean can access to the dataset
  - Tokenizer
    - kowiki-latest-pages-articles.xml.bz2 (https://dumps.wikimedia.org/kowiki/latest/)

## How to Finetune ##
1. Prepare Korean Sentencepiece tokenizer
  - TBD
  * but, it can be found by searching on Google

2. Directory structure
```
│KoBEiT3/
├── beit3/
│   ├── ...
│
├── models/
│   ├── smp.model
│   ├── beit3_large_indomain_patch16_224.zip
│
├── images/
│   ├── ...
│
├── labels/
│   ├── train_labels
│   │   ├── categories
│   │   │   ├── 상(중하)_train_categories
│   │   │   │   ├── annotation.json
│   │   │   │   ├── images.json
│   │   │   │   ├── question.json
│   │   ├── ...
│   │
│   ├── valid_labels
│   │   ├── ...
│
├── answer2label.txt
├── vqa.test.json
├── vqa.train.json
├── vqa.val.json
```

3. Make asnwer2label.txt and json files
  - run beit3/indexing.py
  after running the file, you will get answer2label.txt and json files

4. Fintuning (DistributedDataParallel)
  - run the code below on your terminal, read and customize arguments based on your situation
  ```
  sh finetuning.sh
  ```
  - if your running environment is using SLURM, then use the code below
  ```
  sh finetuning_batch.sh
  ```

## How To Predict ##
1. Prediction
  - run the code below on your terminal
  ```
  sh prediction.sh
  ```
  - if your running environment is using SLURM, then use the code below
  ```
  sh prediction_batch.sh
  ```
  - if you are looking for the code only for one question, check the svqa pipeline
  https://github.com/SeanJeonghwanLee/SpeechVQAPipeline