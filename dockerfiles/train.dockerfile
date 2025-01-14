# Base image
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files for caching
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir --verbose

# Copy the rest of the project
COPY . .

# Install the project package
RUN pip install . --no-deps --no-cache-dir --verbose

# Default entrypoint
ENTRYPOINT ["python", "-u", "-m", "src.train"]
