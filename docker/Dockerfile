FROM registry.ritc.jp/ricos/frontistr/fistr1:ricos
LABEL maintainer "Masanobu Horie <yellowshippo@gmail.com>"

RUN apt-get update \
 && apt-get install -y \
 git \
 libglu1-mesa-dev \
 libxrender1 \
 python3.9 \
 python3.9-dev \
 python3.9-distutils \
 python3-pip \
 python3-venv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHON /usr/bin/python3.9
ENV PYTHONPATH /usr/bin/python3.9
RUN pip3 install --upgrade pip poetry

COPY ./pyproject.toml /src/pyproject.toml
# COPY ./poetry.lock /src/poetry.lock
WORKDIR /src
RUN \
 poetry env use /usr/bin/python3.9 \
 && poetry install \
 && poetry run python3 -m pip install vtk==9.0.20210626.dev0 \
 && poetry run python3 -m pip install mayavi --no-binary :all: \
 && poetry run python3 -m pip install PyQt5
