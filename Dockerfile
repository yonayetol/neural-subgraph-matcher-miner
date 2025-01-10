# Use Ubuntu as base
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python 3.7
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    build-essential \
    pkg-config \
    git \
    libfreetype6-dev \
    libpng-dev \
    libqhull-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.7
RUN curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py \
    && python3.7 get-pip.py \
    && rm get-pip.py

# Create symbolic links for python and pip
RUN ln -sf /usr/bin/python3.7 /usr/bin/python \
    && ln -sf /usr/local/bin/pip3.7 /usr/bin/pip \
    && ln -sf /usr/local/bin/pip3.7 /usr/bin/pip3

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install packages in stages
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first
RUN pip install --no-cache-dir numpy==1.21.6

# Install remaining scientific packages
RUN pip install --no-cache-dir \
    matplotlib==2.1.1 \
    scikit-learn==0.21.3 \
    seaborn==0.9.0

# Install CPU-only PyTorch
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric dependencies (compatible with PyTorch 1.4.0 and Python 3.7)
RUN pip install --no-cache-dir \
    torch-scatter==2.0.2 \
    torch-sparse==0.6.1 \
    torch-cluster==1.5.4 \
    torch-spline-conv==1.2.0 \
    torch-geometric==1.4.3 \
    --find-links https://data.pyg.org/whl/torch-1.4.0+cpu.html

# Install remaining packages
RUN pip install --no-cache-dir \
    deepsnap==0.1.2 \
    networkx==2.4 \
    test-tube==0.7.5 \
    tqdm==4.43.0

# Copy your application code
COPY . .
