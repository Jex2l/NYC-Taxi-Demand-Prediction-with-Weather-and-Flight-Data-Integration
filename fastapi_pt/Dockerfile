# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# 1. Copy & install Python dependencies
COPY fastapi_pt/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy your FastAPI app code
COPY fastapi_pt/app.py .

# 3. Copy your trained XGB model
COPY models/xgb_model_100.pth models/xgb_model_100.pth

# 4. Expose port and launch via Uvicorn
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
