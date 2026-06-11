FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install requests numpy opencv-python-headless supabase datasets huggingface-hub

WORKDIR /app
COPY multi_account_collector_optimized.py .

CMD ["python", "-u", "multi_account_collector_optimized.py"]
