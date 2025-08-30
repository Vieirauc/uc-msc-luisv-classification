FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV TZ=Europe/Lisbon
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependências de sistema e Python
RUN apt-get update && \
    apt-get install -y \
    tzdata \
    git \
    openssh-client \
    curl \
    wget \
    nano \
    vim \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libffi-dev \
    libssl-dev && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup de SSH para clonar repo privado
#RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
#    touch /root/.ssh/known_hosts && ssh-keyscan github.com >> /root/.ssh/known_hosts
#COPY id_rsa_git /root/.ssh/id_rsa
#RUN chmod 600 /root/.ssh/id_rsa

# Clonar repositório
RUN git clone git@github.com:Vieirauc/uc-msc-luisv-cfg-classification.git /workspace/uc-msc-luisv-cfg-classification
WORKDIR /workspace/uc-msc-luisv-cfg-classification

# Instalar PyTorch compatível com CUDA 11.8
#RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Instalar dependências restantes do projeto
RUN pip install -r requirements.txt
