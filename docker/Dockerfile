FROM registry.ritc.jp/ricos/frontistr/fistr1:ricos
LABEL maintainer "Masanobu Horie <yellowshippo@gmail.com>"

RUN apt-get update \
  && apt-get install -y \
  git \
  curl \
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

# RUN pip3 install --upgrade pip poetry \
#  && poetry config virtualenvs.in-project true

COPY ./pyproject.toml /src/pyproject.toml
# COPY ./poetry.lock /src/poetry.lock
WORKDIR /src

ENV PATH $PATH:/root/.poetry/bin

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3.9 \
  && sed -i.bak s/python3/python3.9/g ~/.poetry/bin/poetry \
  && python3.9 -m pip install -U pip \
  && python3.9 -m pip install -U setuptools \
  && python3.9 -m pip install -U wheel \
  && poetry config virtualenvs.create false \
  && poetry install \
  && python3.9 -m pip install vtk==9.0.3 \
  && python3.9 -m pip install mayavi==4.7.3 --no-binary :all: \
  && python3.9 -m pip install PyQt5
