# Use a minimal Python base image
FROM python:3.11-slim

# Work inside /app in the container
WORKDIR /app

# 1) Copy & install only FastAPI’s requirements
COPY fastapi_pt/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy the FastAPI app
COPY fastapi_pt/app.py .

# 3) Copy your trained XGBoost model
COPY models/xgb_model_100.pth models/xgb_model_100.pth

# 4) Expose the inference port and launch with Uvicorn
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
