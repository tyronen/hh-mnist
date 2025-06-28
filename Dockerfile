FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml .

RUN uv sync --group inference

RUN mkdir -p /app/data
COPY data/complex.pth /app/data

COPY models.py .
COPY utils.py .
COPY webserver.py .
