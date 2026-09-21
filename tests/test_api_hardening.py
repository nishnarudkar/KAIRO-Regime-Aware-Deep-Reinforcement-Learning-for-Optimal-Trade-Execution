"""
Tests for API security, persistence, honest model status, error surfacing, and the
train-once / serve-from-disk path.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.agents import registry
from src.api.main import app
from src.api.schemas import ExecutionResponse, ExecutionTrajectoryResponse
from src.api.store import ExecutionStore, global_store

client = TestClient(app)
safe_client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    global_store.clear()
    monkeypatch.delenv("KAIRO_API_KEY", raising=False)
    monkeypatch.delenv("KAIRO_ALLOW_UNAUTHENTICATED", raising=False)
    registry.clear_cache()
    yield
    registry.clear_cache()


@pytest.fixture
def empty_models(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setenv("KAIRO_MODELS_DIR", str(d))
    registry.clear_cache()
    return d


# ── Security ──────────────────────────────────────────────────────────────────

PAPER = {"symbol": "AAPL", "side": "BUY", "quantity": 100.0, "current_price": 100.0,
         "arrival_price": 100.0, "target_inventory": 10000.0}


def test_protected_routes_are_refused_when_no_key_configured():
    assert client.post("/api/execution/kill-switch", json={"active": True}).status_code == 403
    assert client.post("/api/execution/paper", json=PAPER).status_code == 403


def test_protected_routes_require_the_right_key(monkeypatch):
    monkeypatch.setenv("KAIRO_API_KEY", "s3cret")
    assert client.post("/api/execution/kill-switch", json={"active": False}).status_code == 401
    bad = client.post("/api/execution/kill-switch", json={"active": False}, headers={"X-API-Key": "nope"})
    assert bad.status_code == 401
    ok = client.post("/api/execution/kill-switch", json={"active": False}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


def test_explicit_dev_override_allows_protected_routes(monkeypatch):
    monkeypatch.setenv("KAIRO_ALLOW_UNAUTHENTICATED", "1")
    assert client.post("/api/execution/kill-switch", json={"active": False}).status_code == 200


def test_cors_only_allows_configured_origins():
    good = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert good.headers.get("access-control-allow-origin") == "http://localhost:3000"
    bad = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers
    assert "access-control-allow-credentials" not in good.headers


def test_unhandled_errors_do_not_leak_details(monkeypatch):
    from src.api import engine

    def boom(*a, **k):
        raise RuntimeError("super-secret-internal-detail")

    monkeypatch.setattr(engine, "run_policy", boom)
    r = safe_client.post("/api/execution/simulate", json={"policy": "TWAP", "scenario": "normal"})
    assert r.status_code == 500
    assert "super-secret-internal-detail" not in r.text


def test_request_validation():
    assert client.post("/api/execution/simulate", json={"horizon_steps": 1}).status_code == 422
    assert client.post("/api/execution/simulate", json={"quantity": -5}).status_code == 422
    assert client.post("/api/execution/simulate", json={"policy": "MAGIC"}).status_code == 400
    assert client.post("/api/execution/simulate", json={"side": "HOLD", "policy": "TWAP"}).status_code == 400
    assert client.post("/api/execution/backtest", json={"n_windows": 500}).status_code == 422


# ── Persistence ───────────────────────────────────────────────────────────────

def test_executions_persist_across_store_instances(tmp_path):
    resp = client.post("/api/execution/simulate", json={"policy": "TWAP", "scenario": "normal", "seed": 3})
    assert resp.status_code == 200
    rec = ExecutionResponse(**resp.json())
    traj = ExecutionTrajectoryResponse(**client.get(f"/api/execution/{rec.execution_id}/trajectory").json())

    db = str(tmp_path / "kairo.db")
    ExecutionStore(db).save_execution(rec, traj)
    reopened = ExecutionStore(db)           # a fresh process would see the same data
    assert reopened.get_execution(rec.execution_id) == rec
    assert reopened.get_trajectory(rec.execution_id) == traj
    assert reopened.get_execution("missing") is None


def test_status_reflects_the_fill():
    r = client.post("/api/execution/simulate", json={"policy": "POV", "scenario": "liquidity_shock", "seed": 5,
                                                      "horizon_steps": 10}).json()
    assert r["status"] == ("completed" if r["metrics"]["completion_rate"] >= 0.9999 else "partial")


# ── Honest model status / no training in requests ─────────────────────────────

def test_models_report_untrained_when_no_checkpoints(empty_models):
    data = client.get("/api/models").json()
    assert len(data) == 4
    assert all(m["status"] == "untrained" and m["trained"] is False for m in data)
    assert all(m["evaluation"] is None and m["timesteps"] is None for m in data)


def test_rl_policy_without_checkpoint_returns_503_not_a_fake_result(empty_models):
    r = client.post("/api/execution/simulate", json={"policy": "Regime-Aware DQN", "scenario": "normal"})
    assert r.status_code == 503
    assert "train_models.py" in r.json()["detail"]


def test_backtest_surfaces_failures_instead_of_dropping_them(empty_models):
    r = client.post("/api/execution/backtest", json={"scenario": "normal", "n_windows": 3})
    assert r.status_code == 200
    body = r.json()
    assert {x["policy"] for x in body["results"]} == {"TWAP", "VWAP", "POV"}
    assert {e["policy"] for e in body["errors"]} == {"DQN", "Regime-Aware DQN", "PPO", "Regime-Aware PPO"}
    assert all("train_models.py" in e["detail"] for e in body["errors"])


def test_baseline_backtest_is_paired_and_reports_uncertainty(empty_models):
    body = client.post("/api/execution/backtest",
                       json={"scenario": "normal", "n_windows": 8, "policies": ["TWAP", "POV"]}).json()
    assert body["n_windows"] == 8
    by = {x["policy"]: x for x in body["results"]}
    assert by["TWAP"]["vs_twap_bps"] is None
    assert by["POV"]["vs_twap_bps"] is not None
    assert by["POV"]["vs_twap_ci_low"] <= by["POV"]["vs_twap_bps"] <= by["POV"]["vs_twap_ci_high"]
    assert by["TWAP"]["ci_low"] <= by["TWAP"]["implementation_shortfall_bps"] <= by["TWAP"]["ci_high"]


def test_experiments_are_not_reported_completed_without_results(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIRO_RESULTS_DIR", str(tmp_path))
    exp = client.get("/api/experiments").json()
    assert exp[0]["status"] == "not_run" and exp[0]["results_available"] is False
    assert client.get("/api/experiments/results").status_code == 404


def test_experiment_results_endpoint_serves_recorded_files(tmp_path, monkeypatch):
    (tmp_path / "window_results.csv").write_text("scenario,seed,strategy\nnormal,1,TWAP\n")
    (tmp_path / "run_config.json").write_text(json.dumps({"seeds": [1], "scenarios": ["normal"], "train_timesteps": 10}))
    (tmp_path / "summary_table.csv").write_text("strategy,mean,std,min,max,n_runs\nTWAP,10.0,,10.0,10.0,1\n")
    (tmp_path / "paired_comparisons.csv").write_text(
        "rq,treatment,control,scope,level,n,mean,ci_low,ci_high,p_value,win_rate\n"
        "RQ1,DQN,TWAP,ALL,window,5,-1.0,,,,0.6\n")
    monkeypatch.setenv("KAIRO_RESULTS_DIR", str(tmp_path))
    exp = client.get("/api/experiments").json()
    assert exp[0]["status"] == "completed" and exp[0]["seeds"] == [1]
    res = client.get("/api/experiments/results")
    assert res.status_code == 200                       # NaN cells become null, so it is valid JSON
    body = res.json()
    assert body["summary"][0]["std"] is None
    assert body["comparisons"][0]["ci_low"] is None


# ── Train once, serve from disk (tiny model, real serving path) ───────────────

@pytest.fixture
def tiny_models(empty_models):
    from src.evaluation import protocol as P
    from src.evaluation.scenarios import generate_scenario_data

    df = generate_scenario_data("normal", P.N_BARS, 1000)
    sp = P.make_split(df)
    hmm = P.fit_hmm([sp.train])
    hmm.save(str(empty_models / registry.HMM_FILENAME))
    for mid in ("dqn_base", "dqn_regime"):
        spec = registry.MODEL_SPECS[mid]
        agent = P.train_agent(spec["algo"], spec["kind"], [sp.train], hmm, seed=0, timesteps=400, cut=sp.cut)
        agent.save(str(empty_models / mid))
    registry.write_registry({"models": {"dqn_base": {"timesteps": 400, "trained_at": "now",
                                                      "evaluation": {"is_bps_mean": 1.0}}}})
    registry.clear_cache()
    return empty_models


def test_served_model_is_loaded_from_disk_and_reported_trained(tiny_models):
    models = {m["model_id"]: m for m in client.get("/api/models").json()}
    assert models["dqn_base"]["status"] == "trained" and models["dqn_base"]["timesteps"] == 400
    assert models["ppo_base"]["status"] == "untrained"


def test_simulate_with_served_model_records_trajectories_and_explains(tiny_models):
    r = client.post("/api/execution/simulate", json={"policy": "Regime-Aware DQN", "scenario": "normal", "seed": 4})
    assert r.status_code == 200
    rec = r.json()
    traj = client.get(f"/api/execution/{rec['execution_id']}/trajectory").json()
    assert len(traj["observation_trajectory"][0]) == 12
    assert traj["regime_trajectory"] is not None and len(traj["regime_trajectory"]) > 0
    assert traj["window_start"] >= 2100                        # strictly inside the test region

    e = client.get(f"/api/execution/{rec['execution_id']}/explain?step=2")
    assert e.status_code == 200
    body = e.json()
    assert len(body["feature_attributions"]) == 12
    assert "log_return" in body["feature_attributions"]          # real env feature names
    assert body["action_label"].startswith(("Wait", "Execute"))  # real action meanings
    assert body["regime_influence_score"] >= 0


def test_explain_uses_the_real_network_and_validates_input(tiny_models):
    ok = client.post("/api/execution/explain", json={"policy": "DQN", "state": [0.1] * 7})
    assert ok.status_code == 200
    assert client.post("/api/execution/explain", json={"policy": "DQN", "state": [0.1] * 12}).status_code == 400
    assert client.post("/api/execution/explain", json={"policy": "TWAP", "state": [0.1] * 7}).status_code == 400
    assert client.post("/api/execution/explain", json={"policy": "PPO", "state": [0.1] * 7}).status_code == 503


def test_baseline_executions_cannot_be_explained():
    rec = client.post("/api/execution/simulate", json={"policy": "TWAP", "scenario": "normal"}).json()
    assert client.get(f"/api/execution/{rec['execution_id']}/explain").status_code == 400


def test_regime_endpoint_uses_saved_hmm_and_validates_scenario(tiny_models):
    r = client.get("/api/regime/current?scenario=stress&seed=1")
    assert r.status_code == 200
    body = r.json()
    assert abs(sum(body["regime_probabilities"].values()) - 1.0) < 1e-6
    assert body["true_regime"] in body["regime_probabilities"]
    assert client.get("/api/regime/current?scenario=nope").status_code == 400
