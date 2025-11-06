import pandas as pd
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 디바이스 및 배치 크기 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INITIAL_BATCH_SIZE = 70 if torch.cuda.is_available() else 16
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

print("Loading saved models...")
tokenizer = AutoTokenizer.from_pretrained("./deberta_model", use_fast=False, trust_remote_code=True)
model = AutoModel.from_pretrained("./deberta_model", trust_remote_code=True).to(device)
clf = joblib.load("./deberta_model/deberta_classifier.pkl")

print("Loading test data...")
test = pd.read_csv("data/test.csv")

print("Generating embeddings for test data...")
test_texts_a = (test['prompt'] + " " + test['response_a']).tolist()
test_texts_b = (test['prompt'] + " " + test['response_b']).tolist()

test_emb_a = safe_get_embeddings(test_texts_a, tokenizer, model)
test_emb_b = safe_get_embeddings(test_texts_b, tokenizer, model)
X_test = np.concatenate([test_emb_a, test_emb_b], axis=1)

print("Generating predictions...")
pred_test_proba = clf.predict_proba(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "winner_model_a": pred_test_proba[:, 0],
    "winner_model_b": pred_test_proba[:, 1],
    "winner_tie": pred_test_proba[:, 2],
})

submission.to_csv("./result/submission.csv", index=False)
print("\nInference completed!")
