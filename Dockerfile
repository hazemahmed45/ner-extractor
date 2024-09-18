FROM python:3.10
ARG DEBIAN_FRONTEND=noninteractive
ARG PORT
ENV TZ=Africa
WORKDIR /app
COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/ ./src
COPY ./.env/ ./.env
COPY ./ckpt/ ./ckpt

ENV API_HOST=0.0.0.0
ENV API_PORT=8080

EXPOSE ${API_PORT}/tcp

ENTRYPOINT gunicorn -k uvicorn.workers.UvicornWorker --log-level 'info' --bind 0.0.0.0:${API_PORT} 'src.api.run:app' --timeout 10000