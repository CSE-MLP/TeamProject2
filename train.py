import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer,BitsAndBytesConfig

from peft import LoraConfig

from utils import set_seed, save, accuracy, get_args
from model import BertClassification
from data import CombineThreeSentencesDataset, pad_collate_fn


# BASE_PATH = '/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data'
# TRAIN_PATH  = os.path.join(BASE_PATH, 'llm-classification-finetuning/train.csv')
# TEST_PATH   = os.path.join(BASE_PATH, 'llm-classification-finetuning/test.csv')
# SAMPLE_PATH = os.path.join(BASE_PATH, 'llm-classification-finetuning/sample_submission.csv')
# OUT_PATH = os.path.join(BASE_PATH, 'llm-classification-finetuning/submission.csv')
# CHKPT_PATH = os.path.join(BASE_PATH, 'chkpoint')

# device = "cuda"
# seed = 42
# batch_size = 32
# lr = 2e-5
# epochs = 3
# steps_per_epoch = 1797 # 57477(학습 데이터 수) / 128 (배치 사이즈)
# num_training_steps = 5391 # 1797 (steps_per_epoch) * 3 (epochs)
# num_warmup_steps = 269 # 5391 (num_training_steps) * 0.05 (warmup 비율)
# num_cycles = 0.5
# betas=(0.9, 0.999)
# weight_decay=0.01
# label_smoothing=0.05
# model_name = "microsoft/deberta-v3-small"
# max_length = 600 # tokenizer 최대 길이
# amp = False # GradScaler 사용 여부
# grad_clip=1.0 # 기울기 손실 방지

################################## 학습 시작 #############################
if __name__ == '__main__':
    args = get_args()
    BASE_PATH = args.base_path
    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path
    SAMPLE_PATH = args.sample_path
    OUT_PATH = args.out_path
    CHKPT_PATH = args.chkpt_path

    device = args.device
    seed = args.seed
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs

    steps_per_epoch = args.steps_per_epoch
    num_training_steps = args.num_training_steps
    num_warmup_steps = args.num_warmup_steps
    num_cycles = args.num_cycles

    betas = args.betas
    weight_decay = args.weight_decay
    label_smoothing = args.label_smoothing

    model_name = args.model_name
    max_length = args.max_length

    amp = args.amp
    grad_clip = args.grad_clip

    set_seed(seed)
    # 데이터 로더
    df_train = pd.read_csv(TRAIN_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_loader = DataLoader(
        CombineThreeSentencesDataset(df_train,tokenizer,max_length=max_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: pad_collate_fn(b, pad_id=tokenizer.pad_token_id)
    )
    ########### 양자화
    compute_dtype = torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    ############ LORA
    lora_cfg = LoraConfig(
        r=8,                      # rank 낮춤
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["query_proj", "value_proj"],  # 범위 제한
    )
    #############
    # 모델
    device = device if torch.cuda.is_available() else "cpu"
    model = BertClassification(
        num_labels = 3,
        dropout = 0.0,
        lora_cfg = lora_cfg,
        quant_config = quant_config,
    )
    model.to(device)
    # 옵티마이저
    # optimizer = optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=1e-8)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas)
    # 스케줄러
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles)
    # 스케일러
    scaler = GradScaler(enabled=amp)
    # 손실함수
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    @torch.no_grad()
    def accuracy(logits, targets):
        # logits: [B, C] (CrossEntropyLoss 기준)
        # targets: [B] (인덱스) 또는 [B, C] (one-hot)
        if targets.ndim > 1:              # one-hot or soft label
            targets = targets.argmax(dim=1)
        else:
            targets = targets.long()
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()
    def save(path, model, optimizer, scheduler, epoch, best_metric=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_metric": best_metric,
        }, path)


    for epoch in range(1, epochs+1):
        model.train()
        epoch_start = time.time()
        train_loss, train_acc, n = 0.0, 0.0, 0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp, dtype=compute_dtype):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()

            # grad_clip 적용
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * batch_size
            train_acc  += accuracy(logits, labels) * batch_size
            n += batch_size
        train_loss /= n; train_acc /= n
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start
        print(f"[EPOCH {epoch:02d}] train_loss : {train_loss}, train_acc : {train_acc}, lr : {current_lr}, elapsed_time : {elapsed}")
        save(os.path.join(CHKPT_PATH, f"epoch{epoch:02d}.ckpt"), model, optimizer, scheduler, epoch)