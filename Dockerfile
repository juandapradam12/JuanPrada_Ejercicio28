FROM python:3.12-slim-bookworm

# Open MPI for the numerical demos
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        openmpi-bin \
        libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Unsupervised Learning Lab — default analytics pipeline
CMD ["python", "run_analysis.py"]
