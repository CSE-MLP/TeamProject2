import os
import random
import numpy as np
import torch
import argparse

def set_seed(seed):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(False)  # 속도/재현성 트레이드오프에 맞게 조절

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

def get_args():
    parser = argparse.ArgumentParser(description="Fine-tuning for LLM text classification")

    # 경로 설정
    parser.add_argument('--train_path', type=str, default=None, help='Path to train.csv')
    parser.add_argument('--test_path', type=str, default=None, help='Path to test.csv')
    parser.add_argument('--sample_path', type=str, default=None, help='Path to sample_submission.csv')
    parser.add_argument('--out_path', type=str, default=None, help='Path to output submission.csv')
    parser.add_argument('--chkpt_path', type=str, default=None, help='Path to checkpoint directory')

    # 학습 설정
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')

    # 스케줄러 관련
    parser.add_argument('--steps_per_epoch', type=int, default=1797, help='Steps per epoch')
    parser.add_argument('--num_training_steps', type=int, default=5391, help='Total number of training steps')
    parser.add_argument('--num_warmup_steps', type=int, default=269, help='Number of warmup steps')
    parser.add_argument('--num_cycles', type=float, default=0.5, help='Number of cosine cycles')

    # 옵티마이저 관련
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='AdamW betas')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing factor')

    # 모델 관련
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-small', help='Model name')
    parser.add_argument('--max_length', type=int, default=170, help='Max token length')

    # 기타 설정
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision (GradScaler)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping threshold')

    args = parser.parse_args()

    # 기본 경로 자동 설정
    if args.train_path is None:
        args.train_path = os.path.join(args.base_path, '/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data/llm-classification-finetuning/train.csv')
    if args.test_path is None:
        args.test_path = os.path.join(args.base_path, '/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data/llm-classification-finetuning/test.csv')
    if args.sample_path is None:
        args.sample_path = os.path.join(args.base_path, '/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data/llm-classification-finetuning/sample_submission.csv')
    if args.out_path is None:
        args.out_path = os.path.join(args.base_path, '/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data/llm-classification-finetuning/submission.csv')
    if args.chkpt_path is None:
        args.chkpt_path = os.path.join(args.base_path, '/content/drive/MyDrive/머신러닝프로젝트01분반 team12/project2/data/chkpoint')

    return args