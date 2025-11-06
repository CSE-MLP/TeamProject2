# DeBERTa-v3-xsamll Two Input classification 모델
## 학습 예시
    !python train.py \
    --train_path "/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data/llm-classification-finetuning/train.csv" \
    --chkpt_path "/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data/chkpoint/TID_lr_5e-5" \
    --model_name "microsoft/deberta-v3-xsmall" \
    --device "cuda" \
    --seed 42 \
    --grad_clip 1.0 \
    --epochs 3 \
    --lr 5e-5 \
    --batch_size 64 \
    --amp

## checkpoint
https://drive.google.com/drive/folders/1ZEyEMlX6hIelzzMcEyGt0KC2eZEhqqzM?usp=sharing

## colab 예시
https://colab.research.google.com/drive/1jHRm4G2ixnArNr855xSM8ilKfyyXX-Im?usp=sharing
- [lr 5e-5] val_acc : 0.4855

# DeBERTa-v3-xsamll One Input classification 모델
## 학습 예시
    !python train_OID.py \
    --train_path "/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data/llm-classification-finetuning/train.csv" \
    --chkpt_path "/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data/chkpoint/OID_lr_5e-5" \
    --model_name "microsoft/deberta-v3-xsmall" \
    --device "cuda" \
    --seed 42 \
    --grad_clip 1.0 \
    --epochs 3 \
    --lr 5e-5 \
    --batch_size 64 \
    --amp

## checkpoint
https://drive.google.com/drive/folders/1a-ZcgxovMIhVQId_IzufcmwIfiiJBAoy?usp=sharing

## colab 예시
https://colab.research.google.com/drive/1RpBUgl8EDPJPDQ2wdMIYeB2cBE8_p5DC?usp=sharing
- [lr 5e-5] val_acc : 0.4792