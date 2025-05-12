**NYC Taxi Demand Prediction with Weather and Flight Data Integration** 

**Value Proposition** 

**NYC Taxi Demand Prediction with Weather and Flight Data Integration** is a project aimed at improving urban mobility through better forecasting of taxi demand. Using historical NYC Taxi and Limousine Commission (TLC) trip records enriched with weather and flight arrival data, we will predict hourly taxi pickup counts in New York City. By anticipating surges or lulls in demand (for example, due to rain or incoming flights), taxi fleets and dispatchers can proactively allocate drivers, reducing passenger wait times and avoiding oversupply in low-demand periods. This leads to more efficient service for passengers and higher utilization for drivers, addressing urban transit needs.  

The business impact is significant i.e. better demand forecasts mean improved rider satisfaction and driver revenue, and city authorities can use these insights for traffic management. We will evaluate our models with Root Mean Square Error as the primary metric, to quantitatively measure prediction accuracy in number of trips. A lower RMSE on held-out data will indicate a better model, and this metric directly ties to business goals by representing how close our predictions are to actual taxi usage. 


# NYC Taxi Demand Prediction on Chameleon

An end‑to‑end MLOps pipeline—provision VMs, ingest & ETL data, train models, serve via FastAPI/Flask, schedule with Airflow, and monitor with Prometheus/Grafana.

---

## Table of Contents

1. [1️⃣ Launch & VM Setup](#1-launch--vm-setup)  
2. [2️⃣ Object Storage (MinIO)](#2-object-storage-minio)  
3. [3️⃣ Block Storage & ETL](#3-block-storage--etl)  
4. [4️⃣ Model Training & Offline Eval](#4-model-training--offline-eval)  
5. [5️⃣ FastAPI Inference & Metrics](#5-fastapi-inference--metrics)  
6. [6️⃣ Online Streaming Inference](#6-online-streaming-inference)  
7. [7️⃣ Airflow Orchestration](#7-airflow-orchestration)  
8. [8️⃣ Flask Production Service](#8-flask-production-service)  
9. [🧹 Cleanup](#cleanup)  

---

## 1️⃣ Launch & VM Setup

Log into your Chameleon JupyterLab and in a new cell run:

```python
from chi import server, context
import os

# Configure & provision
context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")

username = os.getenv("USER")
s = server.Server(
    name=f"node-persist-{username}",
    image_name="CC-Ubuntu24.04",
    flavor_name="m1.xlarge"
)
s.submit(idempotent=True)
s.associate_floating_ip()
```
``` python
security_groups = [
    {"name":"allow-ssh",   "port":22,   "description":"SSH"},
    {"name":"allow-3000",  "port":3000, "description":"Ray dashboard"},
    {"name":"allow-5000",  "port":5000, "description":"Flask inference"},
    {"name":"allow-8000",  "port":8000, "description":"FastAPI / MLflow"},
    {"name":"allow-8080",  "port":8080, "description":"Custom service"},
    {"name":"allow-8081",  "port":8081, "description":"cAdvisor"},
    {"name":"allow-8888",  "port":8888, "description":"Jupyter"},
    {"name":"allow-9000",  "port":9000, "description":"MinIO API"},
    {"name":"allow-9001",  "port":9001, "description":"MinIO UI"},
    {"name":"allow-9090",  "port":9090, "description":"Prometheus UI"},
]

os_conn     = chi.clients.connection()
nova_server = os_conn.nova().servers.get(s.id)

for sg in security_groups:
    if not os_conn.get_security_group(sg["name"]):
        os_conn.create_security_group(sg["name"], sg["description"])
    os_conn.create_security_group_rule(
        sg["name"],
        port_range_min=sg["port"], port_range_max=sg["port"],
        protocol="tcp", remote_ip_prefix="0.0.0.0/0"
    )
    nova_server.add_security_group(sg["name"])
```
```python
ssh -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```
```python
curl -sSL https://get.docker.com/ | sudo sh
sudo groupadd -f docker
sudo usermod -aG docker $USER
```
**System Diagram** 

The diagram below shows the end-to-end system architecture, including data sources, the ETL pipeline, the training cluster, model registry, and the serving and monitoring components: 

![title](images/diagram.png)
![title](images/IMG_3636.png)
![title](images/IMG_7317.png)
![title](images/IMG_9916.png)


Offline evaluation : Docker/Dockerfile.jupyter-onnx-cpu<br>
Fast api : Docker/docker-compose-fastapi.yaml<br>
Production : Docker/docker-compose-production.yaml<br>
Airflow : airflow/docker-compose-airflow.yaml<br>
fastapi_pt : fastapi_pt/app.py<br>
flask_app :flask_app/app.py<br>
Models : models/xgb_model_100.pth<br>
Mlflow-server : mlflow-server/docker-compose-block.yaml<br>
