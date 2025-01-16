FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy mpi4py

WORKDIR /app

COPY ncm /app/ncm
COPY bin /app/bin

CMD ["app/bin/cmd"]