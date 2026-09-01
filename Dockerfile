FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY vla_platform ./vla_platform
COPY embodied_datasets ./embodied_datasets
ENV PYTHONUNBUFFERED=1
