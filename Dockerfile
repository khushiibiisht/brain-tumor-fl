# Dockerfile
# Creates a container with everything needed to run
# the Brain Tumor FL project on CPU
#
# Build :  docker build -t brain-tumor-fl .
# Server:  docker run brain-tumor-fl python server.py
# Client:  docker run brain-tumor-fl python client.py 0

# Official Python 3.11 slim image
# slim = smaller size (~150MB vs ~900MB for full)
FROM python:3.11-slim

# Who built this image — good practice for team projects
LABEL maintainer="Khushi Bisht"
LABEL project="Brain Tumor Detection using Federated Learning"

# Set working directory inside container
# All files will live here
WORKDIR /app

# Install system dependencies needed by Python packages
# gcc/g++ needed to compile some packages from source
# && rm -rf cleans up apt cache to keep image small
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST before copying code
# Why? Docker caches each step — if requirements.txt
# hasn't changed, Docker skips reinstalling packages
# This makes rebuilds much faster
COPY requirements-docker.txt .

# Install CPU version of PyTorch first
# Separate from other packages because it needs
# a different download URL
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
RUN pip install --no-cache-dir -r requirements-docker.txt

# Now copy all project files into container
# Done AFTER pip install so code changes don't
# force a full package reinstall
COPY . .

# Create directories that the code expects
RUN mkdir -p data logs

# Show Python version when container starts
# (overridden by docker-compose commands)
CMD ["python", "--version"]