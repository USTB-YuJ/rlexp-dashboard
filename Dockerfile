FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 7860

CMD ["rl-exp-dashboard", "serve", "--workspace", "/data", "--host", "0.0.0.0", "--port", "7860"]
