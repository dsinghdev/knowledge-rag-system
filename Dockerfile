FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app/src/backend:/app/src/frontend:/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

# Default to Streamlit UI
CMD ["streamlit", "run", "src/frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
