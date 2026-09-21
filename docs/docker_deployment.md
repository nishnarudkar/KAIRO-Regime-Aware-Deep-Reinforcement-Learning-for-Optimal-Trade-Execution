# Stage 14 — Docker Containerization & Deployment Reference

## 🎯 Overview

Stage 14 containerizes all components of the KAIRO platform using Docker and Docker Compose:
1. **FastAPI Backend Service** (`kairo_backend`): Port `8000`
2. **Next.js Product UI Dashboard** (`kairo_frontend`): Port `3000`
3. **MLflow Experiment Tracking Server** (`kairo_mlflow`): Port `5000`

---

## 🚀 Quickstart Commands

### 1. Launch Full Stack with Docker Compose

```bash
# Build images and start container stack in background
docker compose up -d --build
```

### 2. Verify Running Container Services

```bash
docker compose ps
```

| Service | Container Name | Host Port | Status |
|---|---|---|---|
| `backend` | `kairo_backend` | `http://localhost:8000` | Healthy |
| `frontend` | `kairo_frontend` | `http://localhost:3000` | Running |
| `mlflow` | `kairo_mlflow` | `http://localhost:5000` | Running |

### 3. Service Access Points

- **Next.js Web Dashboard**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Tracking Dashboard**: [http://localhost:5000](http://localhost:5000)

### 4. Stop Stack

```bash
docker compose down
```

---

## 🛠 Container Architecture

```
                    ┌─────────────────────────┐
                    │ Next.js Product UI      │
                    │ Container (Port 3000)   │
                    └────────────┬────────────┘
                                 │ HTTP API Requests
                                 ▼
                    ┌─────────────────────────┐
                    │ FastAPI Backend Engine  │
                    │ Container (Port 8000)   │
                    └────────────┬────────────┘
                                 │ Artifact & Metric Logging
                                 ▼
                    ┌─────────────────────────┐
                    │ MLflow Tracking Server  │
                    │ Container (Port 5000)   │
                    └─────────────────────────┘
```
