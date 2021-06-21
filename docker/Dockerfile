FROM registry.ritc.jp/ricos/frontistr/fistr1:ricos
LABEL maintainer "Masanobu Horie <yellowshippo@gmail.com>"

RUN apt-get update \
 && apt-get install -y git python3-pip python3-venv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip poetry

COPY ./pyproject.toml /src/pyproject.toml
# COPY ./poetry.lock /src/poetry.lock
WORKDIR /src
RUN poetry config virtualenvs.in-project true \
 && poetry install
