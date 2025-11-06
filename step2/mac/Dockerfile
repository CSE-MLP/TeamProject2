FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CPU 전용 PyTorch 먼저 설치
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
# PyTorch 제외한 나머지 패키지 설치
RUN grep -v "^torch==" requirements.txt | pip install --no-cache-dir -r /dev/stdin

COPY download_data.py .
COPY train.py .
COPY inference.py .

RUN mkdir -p /app/deberta_model /app/result

CMD python download_data.py && python train.py && python inference.py
