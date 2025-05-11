# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY fastapi_pt/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY fastapi_pt/app.py .

# (If you have other modules/packages under fastapi_pt/, uncomment the next line
#  to bring them in as well)
# COPY fastapi_pt/ ./ 

# Ensure model file is available inside the container
COPY models/xgb_model_100.pth

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
