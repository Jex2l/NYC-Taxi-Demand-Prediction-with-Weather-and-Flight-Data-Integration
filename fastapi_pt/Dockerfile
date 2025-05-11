# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# 1. Copy requirements and install
COPY fastapi_pt/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy your FastAPI app code
COPY fastapi_pt/app.py .

# 3. Copy your trained model
COPY models/xgb_model_100.pth models/xgb_model_100.pth

# 4. Expose the port the app will run on
EXPOSE 8000

# 5. Launch via Uvicorn (ASGI) for FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
