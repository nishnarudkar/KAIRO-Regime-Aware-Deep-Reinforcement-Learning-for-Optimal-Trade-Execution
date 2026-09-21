# FastAPI Backend

The API wraps the research engine. It shares the market generator, windows and rollouts with the experiments
(`src/evaluation/protocol.py`), and **serves trained models from disk**: nothing is trained inside a request.

```bash
python scripts/train_models.py                         # once: produces models/*.zip, kairo_hmm.joblib, registry.json
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Interactive docs: `http://127.0.0.1:8000/docs`.

---

## Endpoints

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/execution/simulate` | One execution of one policy on an **out-of-sample** window of a synthetic market. Learned policy without a checkpoint → `503`. |
| `POST` | `/api/execution/backtest` | Several policies over `n_windows` (1–30) identical out-of-sample windows. Returns mean, std, bootstrap 95% CI and the paired difference vs TWAP with its CI. Policies that cannot be evaluated are listed in `errors`, never dropped silently. |
| `GET` | `/api/execution/{id}` `/metrics` `/trajectory` | Stored results. The trajectory includes prices, inventory, actions, the regime per step and (for learned policies) the observations the network saw. |
| `GET` | `/api/execution/{id}/explain?step=N` | Explanation of the trained network's decision at step *N*. Learned policies only. |
| `POST` | `/api/execution/explain` | Explain a decision of a trained policy for a given state vector (7-dim, or 12-dim for regime-aware). |
| `GET` | `/api/execution/risk-status` | Pre-trade risk-gate configuration. |
| `POST` | `/api/execution/paper` | 🔒 Route an order slice through the risk gates to the paper executor. |
| `POST` | `/api/execution/kill-switch` | 🔒 Trigger / reset the emergency halt. |
| `GET` | `/api/regime/current` | Causal HMM regime at the latest bar (pooled HMM loaded from disk; no per-request fitting). |
| `GET` | `/api/models` | Learned policies with real status: `trained` only if a checkpoint exists; includes timesteps, training date and the held-out evaluation. |
| `GET` | `/api/experiments` | `completed` only when result files exist in `results/`, otherwise `not_run`. |
| `GET` | `/api/experiments/results` | Recorded research results: per-strategy summary, paired comparisons, HMM validation. |
| `GET` | `/api/baselines` | Baseline strategy descriptions. |
| `GET` | `/health` | Liveness. |

Request limits: `horizon_steps` 5–120, `quantity` ≤ 10,000,000, `n_windows` ≤ 30. Unknown scenarios / policies /
sides return `400`; invalid payloads `422`.

`status` of an execution is `completed` only when fully filled, otherwise `partial`.

Agents were trained with a 30-step horizon and 100,000-share orders. Other horizons and sizes run (observations are
fractions of the order and of the horizon) but are outside the training distribution.

---

## Security

* 🔒 routes (`/paper`, `/kill-switch`) require the header `X-API-Key` matching `KAIRO_API_KEY`. If no key is
  configured they are **refused** (`403`) unless `KAIRO_ALLOW_UNAUTHENTICATED=1` (local development only).
* CORS allows only the origins in `KAIRO_CORS_ORIGINS` (default: the local dashboard), without credentials.
* Unhandled errors return a generic `500` without exception text.

## Persistence

Executions are stored in SQLite (`KAIRO_DB_PATH`, default `data/kairo.db`) and survive restarts. In Docker the
database lives on the `kairo_data` volume.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `KAIRO_API_KEY` | Enables 🔒 routes | unset |
| `KAIRO_ALLOW_UNAUTHENTICATED` | Dev override for 🔒 routes | unset |
| `KAIRO_CORS_ORIGINS` | Comma-separated allowed origins | `http://localhost:3000,http://127.0.0.1:3000` |
| `KAIRO_DB_PATH` | SQLite file (`:memory:` allowed) | `data/kairo.db` |
| `KAIRO_MODELS_DIR` | Checkpoints + `registry.json` | `models/` |
| `KAIRO_RESULTS_DIR` | Experiment results | `results/` |
