# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./

RUN npm install

COPY frontend/ ./

ARG VITE_TURNSTILE_SITE_KEY
ENV VITE_TURNSTILE_SITE_KEY=$VITE_TURNSTILE_SITE_KEY

RUN npm run build

# Stage 2: Install Python deps
FROM python:3.10-slim AS python-deps

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 3: Final runtime image
FROM python:3.10-slim

ENV CONTAINER_HOME=/var/www

WORKDIR $CONTAINER_HOME

COPY --from=python-deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin
COPY src/ $CONTAINER_HOME/src/
COPY --from=frontend-build /app/frontend/dist $CONTAINER_HOME/frontend/dist
COPY data/ $CONTAINER_HOME/data/

EXPOSE 5001

CMD ["python3", "src/app.py"]
