# Use a minimal Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# 1) Copy & install only your FastAPI requirements
COPY fastapi_pt/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy the FastAPI app
COPY fastapi_pt/app.py .

# 3) Copy the trained model artifact
COPY models/xgb_model_100.pth models/xgb_model_100.pth

# 4) Expose port & launch via Uvicorn
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
