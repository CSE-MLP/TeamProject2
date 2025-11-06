## Overview
This project trains a **Logistic Regression classifier** using **DeBERTa-v3-small** embeddings for text classification.

### Training Pipeline
```
Input Data: prompt, response_a, response_b
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│ prompt +         │    │ prompt +         │
│ response_a       │    │ response_b       │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐
│ DeBERTa-v3-small    │  │ DeBERTa-v3-small    │
└────────┬────────────┘  └────────┬────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│ 768-D Embedding │      │ 768-D Embedding │
└────────┬────────┘      └────────┬────────┘
         │                        │
         └───────────┬────────────┘
                     │ Concatenate
                     ▼
            ┌─────────────────┐
            │ 1536-D Vector   │
            │ [emb_a | emb_b] │
            └────────┬────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Logistic Regression        │
        │ Input: 1536-D (768 + 768)  │
        │ Output: 3 classes          │
        │  0: model_a wins           │
        │  1: model_b wins           │
        │  2: tie                    │
        └─────────────┬──────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Class Prediction│
             └─────────────────┘
```

## Kaggle API Setup
1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
2. Scroll down to **API** section
3. Click **Create New Token** (downloads `kaggle.json`)
4. Extract credentials from the downloaded file:
   - `username`: Your Kaggle username
   - `key`: Your API key (long alphanumeric string)

```bash
cd step2

export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
```

## GPU Environment (Linux + NVIDIA GPU)
```bash
sudo docker build -f cuda/Dockerfile -t mlp-inference .
```

## CPU Environment (Mac M Series)
```bash
sudo docker build -f mac/Dockerfile -t mlp-inference .
```

## Run
```bash
sudo docker run --gpus all \
  -e KAGGLE_USERNAME=$KAGGLE_USERNAME \
  -e KAGGLE_KEY=$KAGGLE_KEY \
  -v $(pwd)/result:/app/result \
  mlp-inference
```

After execution, check `result/submission.csv` in your current directory (mounted volume).
