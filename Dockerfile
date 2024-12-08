FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["bash"]