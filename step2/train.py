import pandas as pd
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from tqdm import tqdm

# 디바이스 및 배치 크기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INITIAL_BATCH_SIZE = 70 if torch.cuda.is_available() else 2
print(f"Device: {device}, Batch size: {INITIAL_BATCH_SIZE}")

def get_embeddings(texts, tokenizer, model, batch_size):
    """DeBERTa 모델로 텍스트를 임베딩 벡터로 변환 (단순 함수)"""
    model.eval()
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

def safe_get_embeddings(texts, tokenizer, model):
    """배치 크기를 자동 조정하며 임베딩 생성"""
    batch_size = INITIAL_BATCH_SIZE
    
    while batch_size >= 1:
        try:
            print(f"Trying with batch size: {batch_size}")
            return get_embeddings(texts, tokenizer, model, batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batch_size = batch_size // 2
                print(f"OOM! Reducing batch size to {batch_size}")
            else:
                raise e
    
    raise RuntimeError("Cannot process even with batch size 1")

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 필요 컬럼 확인
print(train.head())
print(test.head())

# 0: model_a wins, 1: model_b wins, 2: tie
train['label'] = train.apply(lambda x: 0 if x['winner_model_a']==1 else (1 if x['winner_model_b']==1 else 2), axis=1)

model_name = "microsoft/deberta-v3-small"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
print(f"Model loaded successfully!")

print("\nGenerating embeddings for training data...")

# 텍스트 결합
train_texts_a = (train['prompt'] + " " + train['response_a']).tolist()
train_texts_b = (train['prompt'] + " " + train['response_b']).tolist()

train_emb_a = safe_get_embeddings(train_texts_a, tokenizer, model)
train_emb_b = safe_get_embeddings(train_texts_b, tokenizer, model)

# 임베딩 concat
X = np.concatenate([train_emb_a, train_emb_b], axis=1)
y = train['label'].values

# Test 데이터 임베딩
print("\nGenerating embeddings for test data...")
test_texts_a = (test['prompt'] + " " + test['response_a']).tolist()
test_texts_b = (test['prompt'] + " " + test['response_b']).tolist()

test_emb_a = safe_get_embeddings(test_texts_a, tokenizer, model)
test_emb_b = safe_get_embeddings(test_texts_b, tokenizer, model)
X_test = np.concatenate([test_emb_a, test_emb_b], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining classifier...")
clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf.fit(X_train, y_train)

pred_val = clf.predict_proba(X_val)
val_log_loss = log_loss(y_val, pred_val)
print(f"Validation Log Loss: {val_log_loss:.5f}")

print("\nSaving models...")

# Logistic Regression 저장
joblib.dump(clf, "./deberta_model/deberta_classifier.pkl")

# Tokenizer와 DeBERTa 모델 저장
tokenizer.save_pretrained("./deberta_model")
model.save_pretrained("./deberta_model")
