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


---

## Configuration notes

* **API URL for the browser.** `NEXT_PUBLIC_API_URL` is inlined into the client bundle **at build time** and is
  called by the user's *browser*, so it must be reachable from the host (default `http://localhost:8000`), not the
  compose service name `backend`. It is passed as a build argument:
  `NEXT_PUBLIC_API_URL=http://my-host:8000 docker compose up -d --build`.
* **CORS.** Set `KAIRO_CORS_ORIGINS` to the dashboard's public origin if it is not `http://localhost:3000`.
* **Protected routes.** Set `KAIRO_API_KEY` to enable `/paper` and `/kill-switch`; they are refused without it.
* **Models and results** are copied into the backend image from `models/` and `results/`. Run
  `python scripts/train_models.py` and `python scripts/run_experiments.py` before building to refresh them.
* **Persistence.** The SQLite execution store lives on the `kairo_data` volume (`/app/data/kairo.db`).
* The backend image uses Python 3.12, CPU-only PyTorch and runs as a non-root user. See `.env.example` for all variables.
