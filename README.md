<div align="center">

# 🤖 End-to-End MLOps System

**Auto-trains 7 ML models → picks the best → serves predictions via API**

[![CI/CD](https://github.com/ETAMURAJESH/end-to-end-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/ETAMURAJESH/end-to-end-mlops/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Hub-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/etamurajesh/end-to-end-mlops)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

### 🌐 [Live API](https://titanic-dataset-y7xg.onrender.com) &nbsp;•&nbsp; 📖 [API Docs](https://titanic-dataset-y7xg.onrender.com/docs) &nbsp;•&nbsp; 🐳 [Docker Hub](https://hub.docker.com/r/etamurajesh/end-to-end-mlops)

</div>

---

## What does this do?

You give it a CSV file. It trains 7 ML models, compares them, picks the best one, and serves predictions through a REST API — automatically.

```
Your CSV file
     ↓
Train 7 models automatically
     ↓
Pick best model (by cross-validation)
     ↓
Serve predictions via API
     ↓
https://your-api.onrender.com/predict
```

Works on **any dataset** — just change two flags:

```bash
python -m src.train --dataset data/your_file.csv --target your_column
```

---

## 🗂️ Datasets Used

| Dataset | Rows | Goal |
|---------|------|------|
| Titanic Survival | 418 | Predict who survived |
| IBM HR Attrition | 1,470 | Predict employee resignation |
| Telco Customer Churn | 7,043 | Predict customer cancellation |

---

## 🏗️ System Architecture

```
You push code to GitHub
         │
         ▼
 GitHub Actions CI/CD
 ┌─────────────────────┐
 │ 1. Check formatting │
 │ 2. Run tests        │
 │ 3. Train model      │
 │ 4. Build Docker     │
 └──────────┬──────────┘
            │
            ▼
       Docker Hub
  etamurajesh/end-to-end-mlops:v1
            │
            ▼
        Render.com
  https://titanic-dataset-y7xg.onrender.com
            │
         ┌──┴──┐
    /predict  /health
```

---

## 🛠️ Tech Stack

| What | Tool |
|------|------|
| ML models | scikit-learn |
| API | FastAPI |
| Experiment tracking | MLflow |
| Containers | Docker + Docker Compose |
| Image registry | Docker Hub |
| CI/CD | GitHub Actions |
| Deployment | Render.com |
| Orchestration | Kubernetes |
| Code quality | Black + Flake8 + pre-commit |
| Tests | Pytest |

---

## 📁 Project Structure

```
end-to-end-mlops/
│
├── src/
│   ├── app.py              # FastAPI server
│   ├── train.py            # Training pipeline
│   ├── preprocessing.py    # Data cleaning
│   ├── model_factory.py    # 7 ML models registry
│   └── config_loader.py    # Config reader
│
├── data/                   # CSV datasets
├── models/                 # Saved model (pipeline.pkl)
├── k8s/                    # Kubernetes manifests
├── tests/                  # API tests (15 tests)
├── .github/workflows/      # CI/CD pipeline
│
├── Dockerfile
├── docker-compose.yml
├── config.yaml
└── requirements.txt
```

---

## 🚀 Run Locally

**1. Clone and install**
```bash
git clone https://github.com/ETAMURAJESH/end-to-end-mlops.git
cd end-to-end-mlops
pip install -r requirements-dev.txt
```

**2. Train a model**
```bash
python -m src.train --dataset data/tested.csv --target Survived
```

**3. Start the API**
```bash
uvicorn src.app:app --reload
```

**4. Test it**
```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","model_ready":true}
```

---

## 🐳 Run with Docker

```bash
# Option 1 — pull from Docker Hub (fastest)
docker pull etamurajesh/end-to-end-mlops:v1
docker run -p 8000:8000 etamurajesh/end-to-end-mlops:v1

# Option 2 — build locally
docker build -t end-to-end-mlops .
docker run -p 8000:8000 end-to-end-mlops

# Option 3 — Docker Compose (API + MLflow together)
docker compose up
# API   → http://localhost:8000/docs
# MLflow → http://localhost:5000
```

---

## 📖 API Reference

**Base URL:** `https://titanic-dataset-y7xg.onrender.com`

### Check if API is running
```bash
curl https://titanic-dataset-y7xg.onrender.com/health
```
```json
{"status": "ok", "model_ready": true}
```

### See which model is loaded
```bash
curl https://titanic-dataset-y7xg.onrender.com/model/info
```
```json
{"model_name": "LogisticRegression", "is_retraining": false}
```

### Make a prediction (Titanic)
```bash
curl -X POST https://titanic-dataset-y7xg.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{
      "Pclass": 3,
      "Age": 22,
      "Sex": "male",
      "Fare": 7.25,
      "Embarked": "S",
      "SibSp": 1,
      "Parch": 0
    }]
  }'
```
```json
{"predictions": [0], "model_name": "LogisticRegression", "record_count": 1}
```
> `0` = Did not survive, `1` = Survived

### Trigger retraining
```bash
curl -X POST https://titanic-dataset-y7xg.onrender.com/retrain
```
```json
{"status": "accepted", "message": "Retraining started in background"}
```

📖 Try all endpoints interactively: **https://titanic-dataset-y7xg.onrender.com/docs**

---

## ☸️ Kubernetes

```bash
kubectl apply -f k8s/
kubectl get pods        # see 3 replicas running
kubectl get hpa         # see auto-scaler (1 → 10 pods)
```

---

## 📊 Model Results (Titanic)

| Model | CV Accuracy | Selected? |
|-------|-------------|-----------|
| Logistic Regression | 80.1% | ✅ Best |
| Random Forest | 79.3% | |
| Gradient Boosting | 78.6% | |
| SVM | 77.1% | |
| KNN | 75.3% | |
| Decision Tree | 72.3% | |
| Naive Bayes | 70.1% | |

> Model selected by **5-fold cross-validation mean** — not test accuracy alone.

---

## 🔮 What's Next

- [ ] MLflow Model Registry (staging → production)
- [ ] Monitoring with Grafana
- [ ] Data drift detection
- [ ] Multi-cloud Kubernetes (AWS EKS / GCP GKE)

---

## 👤 Author

**Etamu Rajesh**

[![GitHub](https://img.shields.io/badge/GitHub-ETAMURAJESH-181717?logo=github)](https://github.com/ETAMURAJESH)
[![Docker Hub](https://img.shields.io/badge/Docker_Hub-etamurajesh-2496ED?logo=docker)](https://hub.docker.com/r/etamurajesh/end-to-end-mlops)

---

<div align="center">
⭐ Star this repo if you found it useful!
</div>
