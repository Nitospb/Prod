FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lookalike_pipeline/ ./lookalike_pipeline/
COPY api_state.py .
COPY pipeline.py .
COPY check_pipeline_time.py .
COPY app.py .
COPY train.py .
COPY predict.py .


EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
