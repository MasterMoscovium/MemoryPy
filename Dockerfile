FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for numpy, scipy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run the Uvicorn server on host 0.0.0.0 and port 8000 explicitly
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
