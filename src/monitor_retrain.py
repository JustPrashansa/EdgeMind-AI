"""EdgeMind AI - model-quality monitoring and retraining workflow.

    python -m src.train --offline                 # (once) creates the approved champion model
    python -m src.monitor_retrain --prepare       # (once) creates the incoming labelled batch
    python -m src.monitor_retrain                 # Tasks A-D: validate, evaluate, retrain, quality gate
    python -m src.monitor_retrain --promote       # OPTIONAL, explicit, recorded promotion of the candidate

The workflow NEVER replaces the approved model by itself; it only records a decision.
"""
import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (APPROVED_MODEL, DATASET_FILE, FEATURES, MODELS_DIR, REGISTRY_DIR,
                        REGISTRY_INDEX, ROOT, TARGET)
from src.preprocessing import clean

INCOMING_DIR = ROOT / "data" / "incoming"
NEW_BATCH = INCOMING_DIR / "new_batch.csv"
REPORT_DIR = ROOT / "reports"
CANDIDATE_DIR = MODELS_DIR / "candidate"
CANDIDATE = CANDIDATE_DIR / "candidate.joblib"
REPORT_JSON = REPORT_DIR / "evaluation_report.json"
DECISION_LOG = REPORT_DIR / "decision_log.csv"
ID_COL = "UDI"

# ---- quality-gate settings -------------------------------------------------
GATE_METRIC = "f1"       # only ~3.4% of rows are failures -> accuracy is misleading; F1 balances recall & precision
MIN_IMPROVEMENT = 0.0    # candidate F1 must be >= champion F1 + this value
MIN_RECALL = 0.50        # same floors as the existing registry rule in src/mlops.py
MIN_F1 = 0.50
MAX_RECALL_DROP = 0.05   # a missed failure is costly: candidate may not lose more than this recall
SEED = 42


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def metrics(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0))}


def _original_split():
    """Reproduce src.train's stratified 80/20 split (same seed) WITHOUT touching data/processed.
    Returns cleaned raw frame (with UDI), train index, unseen index."""
    raw = pd.read_csv(DATASET_FILE)
    ids = raw[ID_COL]
    df = clean(raw)
    df[ID_COL] = ids.loc[df.index]
    tr_idx, te_idx = train_test_split(df.index, test_size=0.2, stratify=df[TARGET].astype(int), random_state=SEED)
    return df, tr_idx, te_idx


def prepare(drift: bool):
    """The champion was trained on 80% of AI4I. The other 20% (2,000 rows) was never used for
    training, so it is used here as the 'incoming labelled batch'. --drift adds mild wear/torque shift."""
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    df, _, te_idx = _original_split()
    new = df.loc[te_idx, [ID_COL] + FEATURES + [TARGET]].copy()
    if drift:
        rng = np.random.default_rng(0)
        new["Tool wear [min]"] = (new["Tool wear [min]"] * rng.normal(1.15, 0.05, len(new))).round()
        new["Torque [Nm]"] = (new["Torque [Nm]"] + rng.normal(1.5, 0.5, len(new))).round(1)
    new.to_csv(NEW_BATCH, index=False)
    log(f"Created {NEW_BATCH.relative_to(ROOT)} with {len(new)} rows (drift={'on' if drift else 'off'})")


def run_workflow():
    REPORT_DIR.mkdir(exist_ok=True); CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    if not APPROVED_MODEL.exists():
        raise SystemExit("No approved model. Run: python -m src.train --offline")
    if not NEW_BATCH.exists():
        raise SystemExit("No incoming batch. Run: python -m src.monitor_retrain --prepare")

    # ============ TASK A: prepare new data ============
    log("TASK A - Prepare new data")
    old_df, tr_idx, _ = _original_split()
    train_df = old_df.loc[tr_idx]
    raw_new = pd.read_csv(NEW_BATCH)
    required = [ID_COL] + FEATURES + [TARGET]
    missing_cols = [c for c in required if c not in raw_new.columns]
    if missing_cols:
        raise SystemExit(f"Missing required columns: {missing_cols}")
    log(f"Required columns present: OK {required}")
    na = raw_new[FEATURES + [TARGET]].isna().sum(); na = {k: int(v) for k, v in na.items() if v}
    dup_ids = int(raw_new[ID_COL].duplicated().sum()); dup_rows = int(raw_new.duplicated().sum())
    overlap = len(set(raw_new[ID_COL]) & set(train_df[ID_COL]))
    log(f"Missing values: {na or 'none'} | duplicate IDs: {dup_ids} | duplicate rows: {dup_rows} | "
        f"IDs already in training data: {overlap}")
    new = clean(raw_new.drop_duplicates(subset=ID_COL)).reset_index(drop=True)  # EdgeMind's own clean()
    new[TARGET] = new[TARGET].astype(int)
    log(f"New batch: {new.shape[0]} rows | existing training data: {len(train_df)} rows")
    log(f"Target distribution (new): {new[TARGET].value_counts().to_dict()} -> {new[TARGET].mean():.2%} failures")
    log(f"Target distribution (training): {train_df[TARGET].astype(int).value_counts().to_dict()} -> "
        f"{train_df[TARGET].mean():.2%} failures")

    # ============ TASK B: evaluate existing model ============
    log("TASK B - Evaluate existing champion on the new labelled batch")
    champion = joblib.load(APPROVED_MODEL)
    X_new, y_new = new[FEATURES], new[TARGET]
    champ_pred = champion.predict(X_new)       # champion never saw these rows
    champ_m = metrics(y_new, champ_pred)
    log("Champion: " + ", ".join(f"{k}={v:.4f}" for k, v in champ_m.items()))

    # ============ TASK C: retrain candidate ============
    log("TASK C - Retrain candidate (existing training data + new batch)")
    X_old, y_old = train_df[FEATURES], train_df[TARGET].astype(int)
    # Fair comparison: 5-fold out-of-fold predictions, so the candidate is never scored on rows it trained on.
    oof = np.zeros(len(new), dtype=int)
    for fit_i, val_i in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_new, y_new):
        m = clone(champion).fit(pd.concat([X_old, X_new.iloc[fit_i]]), pd.concat([y_old, y_new.iloc[fit_i]]))
        oof[val_i] = m.predict(X_new.iloc[val_i])
    cand_m = metrics(y_new, oof)
    # Final candidate: existing training data + the WHOLE new batch (same RF hyper-parameters / same clean()).
    candidate = clone(champion).fit(pd.concat([X_old, X_new]), pd.concat([y_old, y_new]))
    joblib.dump(candidate, CANDIDATE)
    log(f"Combined training rows: {len(X_old) + len(X_new)} | candidate saved separately: {CANDIDATE.relative_to(ROOT)}")

    # ============ TASK D: quality gate ============
    log("TASK D - Quality gate (champion vs candidate, same 5-fold-validated new batch)")
    log(f"{'metric':<11}{'champion':>10}{'candidate':>11}{'delta':>9}")
    for k in champ_m:
        log(f"{k:<11}{champ_m[k]:>10.4f}{cand_m[k]:>11.4f}{cand_m[k] - champ_m[k]:>+9.4f}")
    reasons = []
    if cand_m[GATE_METRIC] - champ_m[GATE_METRIC] < MIN_IMPROVEMENT:
        reasons.append(f"{GATE_METRIC} {cand_m[GATE_METRIC]:.3f} is below champion {champ_m[GATE_METRIC]:.3f}")
    if cand_m["recall"] < MIN_RECALL or cand_m["f1"] < MIN_F1:
        reasons.append(f"below absolute floors (recall>={MIN_RECALL}, f1>={MIN_F1})")
    if cand_m["recall"] - champ_m["recall"] < -MAX_RECALL_DROP:
        reasons.append(f"recall dropped by more than {MAX_RECALL_DROP}")
    eligible = not reasons
    decision = "CANDIDATE ELIGIBLE FOR PROMOTION" if eligible else "RETAIN EXISTING CHAMPION"
    log(f"DECISION: {decision}" + (f" | reasons: {'; '.join(reasons)}" if reasons else ""))

    # ---- MLflow tracking (same local experiment as src.train) ----
    run_id = None
    try:
        import mlflow
        from src import mlops
        mlops.setup_mlflow()
        with mlflow.start_run(run_name="monitoring_retrain_candidate") as run:
            mlflow.log_params({"algorithm": "RandomForestClassifier", "train_rows": len(X_old) + len(X_new),
                               "new_batch_rows": len(X_new), "gate_metric": GATE_METRIC, **champion.get_params()})
            mlflow.log_metrics({**{f"candidate_{k}": v for k, v in cand_m.items()},
                                **{f"champion_{k}": v for k, v in champ_m.items()}})
            mlflow.set_tag("quality_gate_decision", decision)
            mlflow.log_artifact(str(CANDIDATE))
            run_id = run.info.run_id
        log(f"MLflow run logged: {run_id}")
    except Exception as exc:
        log(f"MLflow tracking skipped ({exc})")

    ts = datetime.now().isoformat(timespec="seconds")
    report = {"timestamp": ts,
              "data_validation": {"new_rows": len(new), "training_rows": len(train_df), "missing_values": na,
                                  "duplicate_ids": dup_ids, "ids_overlapping_training": overlap,
                                  "target_distribution_new": new[TARGET].value_counts().to_dict(),
                                  "failure_rate_new": float(new[TARGET].mean()),
                                  "failure_rate_training": float(train_df[TARGET].mean())},
              "champion_metrics_on_new_batch": champ_m, "candidate_metrics_5fold_on_new_batch": cand_m,
              "quality_gate": {"metric": GATE_METRIC, "min_improvement": MIN_IMPROVEMENT, "min_recall": MIN_RECALL,
                               "min_f1": MIN_F1, "max_recall_drop": MAX_RECALL_DROP, "eligible": eligible,
                               "decision": decision, "reasons": reasons, "promoted": False},
              "artifacts": {"champion": str(APPROVED_MODEL.relative_to(ROOT)),
                            "candidate": str(CANDIDATE.relative_to(ROOT))},
              "mlflow_run_id": run_id}
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    new_file = not DECISION_LOG.exists()
    with DECISION_LOG.open("a") as f:
        if new_file:
            f.write("timestamp,gate_metric,champion,candidate,decision,promoted\n")
        f.write(f"{ts},{GATE_METRIC},{champ_m[GATE_METRIC]:.4f},{cand_m[GATE_METRIC]:.4f},{decision},no\n")
    log(f"Saved {REPORT_JSON.relative_to(ROOT)} and {DECISION_LOG.relative_to(ROOT)}; approved model NOT changed")


def promote():
    """Explicit human-triggered step: register the candidate as a new approved version, with the decision recorded."""
    rep = json.loads(REPORT_JSON.read_text())
    if not rep["quality_gate"]["eligible"]:
        raise SystemExit("Candidate is not eligible for promotion (see reports/evaluation_report.json).")
    if rep["quality_gate"]["promoted"]:
        raise SystemExit("This candidate was already promoted.")
    import shutil
    reg = json.loads(REGISTRY_INDEX.read_text())
    version = len(reg["versions"]) + 1
    target = REGISTRY_DIR / f"model_v{version}.joblib"
    shutil.copy(CANDIDATE, target)
    for v in reg["versions"]:
        if v["status"] == "approved":
            v["status"] = "archived"
    m = rep["candidate_metrics_5fold_on_new_batch"]
    reg["versions"].append({"version": version, "file": target.name, "trained_at": rep["timestamp"],
                            "algorithm": "RandomForestClassifier", "params": {}, "metrics": m,
                            "mlflow_run_id": rep["mlflow_run_id"], "data_source": "AI4I 2020 + incoming batch",
                            "status": "approved",
                            "decision": "Promoted by monitoring workflow: " + rep["quality_gate"]["decision"]})
    reg["approved_version"] = version
    REGISTRY_INDEX.write_text(json.dumps(reg, indent=2))
    shutil.copy(CANDIDATE, APPROVED_MODEL)
    rep["quality_gate"]["promoted"] = True
    REPORT_JSON.write_text(json.dumps(rep, indent=2))
    with DECISION_LOG.open("a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')},{GATE_METRIC},,"
                f"{m[GATE_METRIC]:.4f},PROMOTED to registry v{version},yes\n")
    log(f"Candidate promoted to registry version {version}; decision recorded")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="create data/incoming/new_batch.csv")
    ap.add_argument("--drift", action="store_true", help="with --prepare: add mild sensor drift to the batch")
    ap.add_argument("--promote", action="store_true", help="explicitly promote the eligible candidate")
    a = ap.parse_args()
    if a.prepare:
        prepare(a.drift)
    elif a.promote:
        promote()
    else:
        run_workflow()
