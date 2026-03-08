FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src/backend:/app/src/frontend:/app/src \
    HF_HOME=/app/models \
    PORT=8501

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

# Expose the port Streamlit will run on
EXPOSE ${PORT}

# Healthcheck for AWS Load Balancer / App Runner
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Default command to run the Streamlit UI
CMD ["streamlit", "run", "src/frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
