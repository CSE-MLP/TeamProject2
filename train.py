import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup
from transformers import AutoTokenizer

from utils import set_seed, save, accuracy, get_args
from model import TIDAutoBertClassification
from data import TwoInputDatasetv2, pad_collate_fn_TID

# BASE_PATH = '/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data'
# TRAIN_PATH  = os.path.join(BASE_PATH, 'llm-classification-finetuning/train.csv')
# TEST_PATH   = os.path.join(BASE_PATH, 'llm-classification-finetuning/test.csv')
# SAMPLE_PATH = os.path.join(BASE_PATH, 'llm-classification-finetuning/sample_submission.csv')
# OUT_PATH = os.path.join(BASE_PATH, 'llm-classification-finetuning/submission.csv')
# CHKPT_PATH = os.path.join(BASE_PATH, 'chkpoint')

# device = "cuda"
# seed = 42
# batch_size = 32
# lr = 2e-4
# epochs = 3
# betas=(0.9, 0.999)
# weight_decay=0.01
# label_smoothing=0.05
# model_name = "microsoft/deberta-v3-small"
# max_length = 600 # tokenizer 최대 길이
# amp = False # GradScaler 사용 여부
# grad_clip=1.0 # 기울기 손실 방지
# quantization = True
# lora = True

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for input_ids1, attention_mask1, input_ids2, attention_mask2, labels in loader:
        bs = labels.size(0)
        input_ids1 = input_ids1.to(device)
        attention_mask1 = attention_mask1.to(device)
        input_ids2 = input_ids2.to(device)
        attention_mask2 = attention_mask2.to(device)
        labels = labels.to(device)
        logits = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, labels) * bs
        n += bs
    return total_loss / n, total_acc / n

if __name__ == '__main__':
    args = get_args()
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

    betas = args.betas
    weight_decay = args.weight_decay
    label_smoothing = args.label_smoothing

    model_name = args.model_name

    amp = args.amp
    grad_clip = args.grad_clip

    set_seed(seed)
    # 데이터 로더
    df = pd.read_csv(TRAIN_PATH)
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=seed)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_loader = DataLoader(
        TwoInputDatasetv2(df_train,tokenizer),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: pad_collate_fn_TID(b, pad_token_id=tokenizer.pad_token_id)
    )
    valid_loader = DataLoader(
        TwoInputDatasetv2(df_valid,tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: pad_collate_fn_TID(b, pad_token_id=tokenizer.pad_token_id)
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    #############
    # 모델
    device = device if torch.cuda.is_available() else "cpu"
    model = TIDAutoBertClassification(
        model_name = model_name,
        pooling="mean",
        dropout=0.1,
        num_labels=3
    )
    model.to(device)
    # 옵티마이저
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=betas)
    # 스케줄러
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, 0.5)
    # 스케일러
    scaler = GradScaler(enabled=amp)
    # 손실함수
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for epoch in range(1, epochs+1):
        model.train()
        epoch_start = time.time()
        train_loss, train_acc, n = 0.0, 0.0, 0

        for input_ids1, attention_mask1, input_ids2, attention_mask2, labels in train_loader:
            bs = labels.size(0)
            input_ids1 = input_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            input_ids2 = input_ids2.to(device)
            attention_mask2 = attention_mask2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                logits = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()

            # grad_clip 적용
            scaler.unscale_(optimizer)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * bs
            train_acc  += accuracy(logits, labels) * bs
            n += bs
        train_loss /= n; train_acc /= n
        current_lr = optimizer.param_groups[0]["lr"]
        val_loss, val_acc = evaluate(model, valid_loader, device, criterion)

        elapsed = time.time() - epoch_start
        print(f"[EPOCH {epoch:02d}] train_loss : {train_loss}, train_acc : {train_acc}, val_loss : {val_loss}, val_acc : {val_acc}, lr : {current_lr}, elapsed_time : {elapsed}")
        
        save(os.path.join(CHKPT_PATH, f"epoch_{epoch}"), model, tokenizer)