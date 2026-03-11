# Stage 1: Install Python deps
FROM python:3.10-slim AS python-deps

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Final runtime image
FROM python:3.10-slim

ENV CONTAINER_HOME=/var/www

WORKDIR $CONTAINER_HOME

COPY --from=python-deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY requirements.txt $CONTAINER_HOME/requirements.txt
COPY src/ $CONTAINER_HOME/

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
