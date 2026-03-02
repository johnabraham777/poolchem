# Example Dockerfile for Render (use your existing base image and deps)
# Your app likely has: FROM python:3.x-slim, COPY, pip install -r requirements.txt, etc.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Render injects PORT at runtime; default 8000 for local Docker
ENV PORT=8000
EXPOSE ${PORT}

# Shell form so ${PORT} is expanded at runtime when Render sets PORT
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
