FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline/ ./pipeline/
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "pipeline.dashboard:server"]
