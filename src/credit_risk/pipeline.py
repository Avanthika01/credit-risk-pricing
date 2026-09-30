"""End-to-end training pipeline: data -> features -> models -> evaluation -> reports.

Every number written to metrics.json is computed here from the data passed in.
Synthetic runs are stamped data_source="SYNTHETIC" and written to a separate,
gitignored directory so they can never be confused with real results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_risk import reports as rp
from credit_risk import tracking
from credit_risk.config import Config
from credit_risk.data import TARGET, load_population, parse_raw, scope_summary, select_population, time_split
from credit_risk.evaluate import calibration_table, pd_metrics
from credit_risk.explain import primary_reason_frequency, reason_codes
from credit_risk.fairness import LIMITATION_NOTE, adverse_impact, fairness_summary, zip3
from credit_risk.features import FeatureBuilder
from credit_risk.models import LGBMModel, PDModel, WoEScorecard
from credit_risk.monitoring import psi_report, psi_status
from credit_risk.pricing import benchmark_by_grade, cutoff_table, estimate_lgd, risk_based_rate

log = logging.getLogger(__name__)

MODEL_LABELS = {"scorecard": "WoE scorecard", "lightgbm": "LightGBM"}


def _split_stats(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "default_rate": float(df[TARGET].mean()),
        "issue_d_min": str(df["issue_d"].min().date()),
        "issue_d_max": str(df["issue_d"].max().date()),
    }


def choose_champion(val_metrics: dict, min_uplift: float) -> tuple[str, str]:
    uplift = val_metrics["lightgbm"]["gini"] - val_metrics["scorecard"]["gini"]
    if uplift >= min_uplift:
        return "lightgbm", (f"LightGBM validation Gini uplift {uplift:.4f} >= required {min_uplift:.2f}")
    return "scorecard", (f"LightGBM validation Gini uplift {uplift:.4f} < required {min_uplift:.2f}; "
                         "scorecard preferred for transparency")


def run(cfg: Config, *, synthetic_rows: int | None = None, out_dir: Path | None = None,
        models_dir: Path | None = None, use_mlflow: bool = True, refresh: bool = False) -> dict:
    synthetic = synthetic_rows is not None
    source = "SYNTHETIC" if synthetic else "lending_club_kaggle"
    out_dir = out_dir or (cfg.paths.root / "artifacts" / "synthetic" / "reports" if synthetic else cfg.paths.reports_dir)
    models_dir = models_dir or (cfg.paths.root / "artifacts" / "synthetic" / "models" if synthetic
                                else cfg.paths.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    subtitle = "SYNTHETIC DATA" if synthetic else f"{cfg.split.test_year} holdout"

    # ---- data ------------------------------------------------------------
    if synthetic:
        from credit_risk.synthetic import make_synthetic_loans

        parsed = parse_raw(make_synthetic_loans(n=synthetic_rows, seed=cfg.model.seed))
        scope = scope_summary(parsed, cfg)
        pop = select_population(parsed, cfg)
    else:
        pop, scope = load_population(cfg, refresh=refresh)
    train, val, test = time_split(pop, cfg)
    log.info("train %d, val %d, test %d", len(train), len(val), len(test))

    # ---- features (fit on train only) -------------------------------------
    fb = FeatureBuilder(cfg.features).fit(train)
    Xtr, Xva, Xte = fb.transform(train), fb.transform(val), fb.transform(test)
    ytr, yva, yte = train[TARGET].to_numpy(), val[TARGET].to_numpy(), test[TARGET].to_numpy()

    # ---- models ------------------------------------------------------------
    sc = PDModel("scorecard", WoEScorecard(cfg.features, cfg.model).fit(Xtr, ytr)).calibrate(Xva, yva)
    gb = PDModel("lightgbm", LGBMModel(cfg.model).fit(Xtr, ytr, Xva, yva)).calibrate(Xva, yva)
    models = {"scorecard": sc, "lightgbm": gb}

    metrics_val, metrics_test_raw, metrics_test, preds_test = {}, {}, {}, {}
    for name, m in models.items():
        metrics_val[name] = pd_metrics(yva, m.base.predict_raw(Xva))
        metrics_test_raw[name] = pd_metrics(yte, m.base.predict_raw(Xte))
        preds_test[name] = m.predict_pd(Xte)
        metrics_test[name] = pd_metrics(yte, preds_test[name])

    champ_name, champ_reason = choose_champion(metrics_val, cfg.model.champion_min_gini_uplift)
    champ = models[champ_name]
    p_test = preds_test[champ_name]

    # ---- explainability ------------------------------------------------------
    contrib = champ.contributions(Xte)
    declined = p_test > cfg.decision.pd_cutoff
    reason_freq = primary_reason_frequency(contrib, mask=declined, n=cfg.model.n_reason_codes)
    example_idx = int(np.argmax(p_test))
    example_codes = reason_codes(contrib.iloc[[example_idx]], n=cfg.model.n_reason_codes)[0]

    # ---- monitoring (PSI train vs holdout) ---------------------------------
    if champ_name == "lightgbm":
        importance = gb.base.feature_importance
    else:
        importance = pd.Series({f: sc.base.binner.iv_[f] for f in sc.base.features_}).sort_values(ascending=False)
    top_feats = list(importance.index[:8])
    psi_feat = psi_report(Xtr, Xte, top_feats)
    score_psi = float(psi_report(pd.DataFrame({"pd": champ.predict_pd(Xtr)}),
                                 pd.DataFrame({"pd": p_test}), ["pd"])["psi"].iloc[0])

    # ---- pricing --------------------------------------------------------------
    lgd = estimate_lgd(train.loc[train[TARGET] == 1])
    test_priced = test.assign(pd=p_test, model_rate=risk_based_rate(p_test, lgd["lgd"], cfg.pricing))
    bench = benchmark_by_grade(test_priced)
    cutoffs = cutoff_table(yte, p_test, cfg.decision.cutoff_grid)

    # ---- fairness (geographic proxy) -----------------------------------------
    fair_df = test_priced.assign(zip3=zip3(test_priced["zip_code"]))
    fair = {}
    for col in ["addr_state", "zip3"]:
        t = adverse_impact(fair_df, col, cutoff=cfg.decision.pd_cutoff,
                           min_group_size=cfg.decision.min_group_size, air_threshold=cfg.decision.air_threshold)
        fair[col] = (t, fairness_summary(t))

    # ---- persist model bundle (used by the API) ---------------------------------
    bundle = {"feature_builder": fb, "champion": champ, "challenger": models[
        "lightgbm" if champ_name == "scorecard" else "scorecard"], "lgd": lgd["lgd"], "config": cfg,
              "data_source": source, "trained_at": datetime.now(timezone.utc).isoformat()}
    joblib.dump(bundle, models_dir / "model_bundle.joblib")

    # ---- metrics.json -------------------------------------------------------------
    sc_base = sc.base
    metrics = {
        "data_source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "population": scope,
        "splits": {"train": _split_stats(train), "validation": _split_stats(val), "test": _split_stats(test)},
        "features": {
            "n_features": len(fb.feature_names), "names": fb.feature_names,
            "dropped_constant_in_train": fb.dropped_constant_,
        },
        "scorecard": {
            "selected_features": sc_base.features_, "dropped_low_iv": sc_base.dropped_low_iv_,
            "dropped_sign_constraint": sc_base.dropped_sign_,
            "coefficients": {k: float(v) for k, v in sc_base.coefficients.items()},
            "information_value": {k: float(v) for k, v in sorted(sc_base.binner.iv_.items(),
                                                                  key=lambda kv: -kv[1])},
        },
        "lightgbm": {"best_iteration": gb.base.best_iteration_,
                     "top_features_gain": {k: float(v) for k, v in gb.base.feature_importance.head(15).items()}},
        "validation_uncalibrated": metrics_val,
        "test_uncalibrated": metrics_test_raw,
        "test_calibrated": metrics_test,
        "champion": {"model": champ_name, "rule": champ_reason,
                     "min_gini_uplift": cfg.model.champion_min_gini_uplift},
        "reason_codes": {
            "declined_share_test": float(declined.mean()),
            "frequency_among_declined": reason_freq.reset_index().to_dict(orient="records"),
            "example_highest_pd_applicant": {"pd": float(p_test[example_idx]), "codes": example_codes},
        },
        "psi": {"score": {"psi": score_psi, "status": psi_status(score_psi)},
                "features": psi_feat.to_dict(orient="records")},
        "lgd": lgd,
        "pricing": {"assumptions": asdict(cfg.pricing), "average_life_years": cfg.pricing.average_life_years,
                    "benchmark_by_grade": bench.to_dict(orient="records")},
        "cutoffs": cutoffs.to_dict(orient="records"),
        "policy_cutoff": {"pd_cutoff": cfg.decision.pd_cutoff,
                          **cutoffs.loc[np.isclose(cutoffs["pd_cutoff"], cfg.decision.pd_cutoff)].iloc[0].to_dict()},
        "fairness": {"note": LIMITATION_NOTE, "pd_cutoff": cfg.decision.pd_cutoff,
                     "min_group_size": cfg.decision.min_group_size, "air_threshold": cfg.decision.air_threshold,
                     "by_state": fair["addr_state"][1], "by_zip3": fair["zip3"][1]},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))

    # ---- markdown tables and charts ------------------------------------------------
    _write_tables(out_dir, metrics, bench, cutoffs, psi_feat, reason_freq, fair, sc_base, source)
    labels = {MODEL_LABELS[k]: v for k, v in preds_test.items()}
    aucs = {MODEL_LABELS[k]: v["auc"] for k, v in metrics_test.items()}
    rp.roc_chart(yte, labels, aucs, out_dir / "roc.png", subtitle)
    rp.calibration_chart({MODEL_LABELS[k]: calibration_table(yte, v) for k, v in preds_test.items()},
                         out_dir / "calibration.png", subtitle)
    rp.cutoff_chart(cutoffs, cfg.decision.pd_cutoff, out_dir / "cutoff_curve.png",
                    f"{MODEL_LABELS[champ_name]}, {subtitle}")
    rp.pricing_chart(bench, out_dir / "pricing_by_grade.png", subtitle)

    # ---- MLflow (optional) -------------------------------------------------------------
    with tracking.run(f"pd-{source}", enabled=use_mlflow, tracking_dir=cfg.paths.root / "mlruns") as t:
        t.params({"data_source": source, "champion": champ_name, "pd_cutoff": cfg.decision.pd_cutoff,
                  **{f"lgbm_{k}": v for k, v in cfg.model.lgbm_params.items()},
                  "woe_bins": cfg.features.woe_bins, "lgbm_best_iteration": gb.base.best_iteration_})
        for name, m in metrics_test.items():
            t.metrics(m, prefix=f"test_{name}_")
        t.metrics({"score_psi": score_psi, "lgd": lgd["lgd"]})
        t.artifact(out_dir / "metrics.json")
    return metrics


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, Path)):
        return str(o)
    raise TypeError(type(o))


def _write_tables(out_dir, metrics, bench, cutoffs, psi_feat, reason_freq, fair, sc_base, source):
    banner = ("> **SYNTHETIC DATA - NOT REAL RESULTS.** Generated for tests/CI only.\n\n"
              if source == "SYNTHETIC" else "")
    rows = []
    for split_key, label in [("test_uncalibrated", "uncalibrated"), ("test_calibrated", "calibrated")]:
        for name, m in metrics[split_key].items():
            rows.append({"model": MODEL_LABELS[name], "PD": label, **{k: m[k] for k in
                         ["auc", "gini", "ks", "brier", "log_loss", "mean_pd", "observed_default_rate"]}})
    perf = rp.to_markdown(pd.DataFrame(rows))
    val_rows = [{"model": MODEL_LABELS[n], **{k: m[k] for k in ["auc", "gini", "ks"]}}
                for n, m in metrics["validation_uncalibrated"].items()]
    pct = "{:.2%}"
    sections = [
        ("model_performance.md", "Test-period (holdout) performance", perf
         + "\n**Validation (uncalibrated; used for champion selection)**\n\n" + rp.to_markdown(pd.DataFrame(val_rows))),
        ("pricing_by_grade.md", "Model vs Lending Club pricing by LC grade (test period)",
         rp.to_markdown(bench, {"observed_default_rate": pct, "mean_model_pd": pct, "lc_mean_rate": pct,
                                "model_mean_rate": pct, "rate_diff": "{:+.2%}"})),
        ("cutoffs.md", "Approval rate vs bad rate among approved (champion, test period)",
         rp.to_markdown(cutoffs, {"pd_cutoff": "{:.3f}", "approval_rate": pct, "bad_rate_approved": pct,
                                  "bad_rate_declined": pct})),
        ("psi.md", "Population Stability Index, train vs test",
         f"Score PSI: {metrics['psi']['score']['psi']:.4f} ({metrics['psi']['score']['status']})\n\n"
         + rp.to_markdown(psi_feat)),
        ("reason_codes.md", "Reason-code frequency among declined applicants (test period)",
         rp.to_markdown(reason_freq.reset_index(), {"primary_reason_share": pct, "any_top_reason_share": pct})),
        ("scorecard.md", "Scorecard: selected features, coefficients and IV",
         rp.to_markdown(pd.DataFrame({"feature": sc_base.features_,
                                      "coefficient": [sc_base.coefficients[f] for f in sc_base.features_],
                                      "information_value": [sc_base.binner.iv_[f] for f in sc_base.features_]}))),
    ]
    for col, label in [("addr_state", "state"), ("zip3", "3-digit ZIP")]:
        t, s = fair[col]
        body = (f"{LIMITATION_NOTE}\n\nGroups with >= {metrics['fairness']['min_group_size']} loans: {s['n_groups']}; "
                f"flagged (AIR < {metrics['fairness']['air_threshold']}): {s['n_flagged']}\n\n")
        shown = t if col == "addr_state" else pd.concat([t.head(15), t.loc[t["flagged"]]]).drop_duplicates()
        body += rp.to_markdown(shown, {"approval_rate": pct, "mean_pd": pct, "observed_default_rate": pct,
                                       "air": "{:.3f}"})
        if col == "zip3":
            body += "\n(Showing the 15 lowest-AIR ZIP3 groups plus any flagged groups.)\n"
        sections.append((f"fairness_{col}.md", f"Adverse impact ratio by {label} (test period)", body))
    for fname, title, body in sections:
        (out_dir / fname).write_text(f"# {title}\n\n{banner}{body}")
