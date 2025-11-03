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
from model import OIDBertClassification
from data import OneInputDataset, pad_collate_fn_OID


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
    quantization = args.quantization
    lora = args.lora

    set_seed(seed)
    # 데이터 로더
    df_train = pd.read_csv(TRAIN_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_loader = DataLoader(
        OneInputDataset(df_train,tokenizer,max_length=max_length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: pad_collate_fn_OID(b, pad_token_id=tokenizer.pad_token_id)
    )
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)



    ########### 양자화
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quant_config = None
    if quantization:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    ############ LORA
    lora_cfg = None
    if lora:
        lora_cfg = LoraConfig(
            r=8,                      # rank 낮춤
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["query_proj", "key_proj", "value_proj", "dense"],  # 범위 제한
        )
    #############
    # 모델
    device = device if torch.cuda.is_available() else "cpu"
    model = OIDBertClassification(
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
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, num_cycles)
    # 스케일러
    scaler = GradScaler(enabled=amp)
    # 손실함수
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)


    for epoch in range(1, epochs+1):
        model.train()
        epoch_start = time.time()
        train_loss, train_acc, n = 0.0, 0.0, 0

        for input_ids, attention_mask, labels in train_loader:
            bs = labels.size(0)
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
        elapsed = time.time() - epoch_start
        print(f"[EPOCH {epoch:02d}] train_loss : {train_loss}, train_acc : {train_acc}, lr : {current_lr}, elapsed_time : {elapsed}")
        save(os.path.join(CHKPT_PATH, f"epoch{epoch:02d}.ckpt"), model, optimizer, scheduler, epoch)