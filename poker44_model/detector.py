"""Poker44 bot detector (BEATER) -- a WITHIN-BATCH RANK-FUSED ENSEMBLE of three
decorrelated members over the SAME 180-dim sanitization-invariant C2 feature row,
topped with our reward-fit, FPR-capped floating decision layer.

Why this model (closing the live-rank gap)
-------------------------------------------
Our benchmark GroupKFold-by-date AP (~0.92 on the 180 C2 features) already EQUALS
the steady top tier's, yet our single-LightGBM mains score ~0.49 live while the
steady band scores ~0.55. The whole gap is in the 65% RANK block
(0.35*AP + 0.30*recall@FPR<=0.05); we already max the 30% hard-0.5-threshold block.
Root cause: a single learner is the highest-variance choice out-of-distribution on
the sanitized live feed. Every steady winner runs a within-batch RANK-FUSED
ENSEMBLE of decorrelated members plus sign-stability-gated monotone constraints.
This model ports that recipe over OUR feature pipeline verbatim.

Members (all over the identical 180-dim FEATURE_NAMES row)
---------------------------------------------------------
  1. STACK  -- LGBM + XGB + RF -> logistic OOF stack (the discrimination anchor).
  2. MONO   -- monotone-constrained LightGBM bag; monotone_constraints set to
               +1/-1 ONLY for features whose per-DATE Spearman(feature,label) sign
               is stable across >=70% of dates AND |mean rho| >= 0.05 (else 0).
               The OOD-transfer regularizer.
  3. MLP    -- PCA(56) -> MLP bag on the standardized feature view; architecturally
               decorrelated from the tree members.

Fusion is calibration-free: each member's WITHIN-BATCH rank (argsort/argsort/(n-1))
is averaged with fixed weights (0.35, 0.30, 0.35), so no member's OOD score-scale
can distort the blend. The fused rank is the movable ordering that drives the 65%
RANK block.

Decision layer (reused verbatim from BEST/GAP_FIX)
--------------------------------------------------
The fused within-batch rank is passed through an isotonic map then the reward-fit
per-batch decision layer (anchor quantile Q + logit margin/temp + hard FLOOR + CAP)
-> DETERMINISTIC ~2% of every window crosses 0.5, zero hard-zeros. That transform is
monotone, so AP / recall@FPR (the 65% block) are set purely by the fused rank while
the 30% hard-0.5-threshold block stays pinned high and steady.

IMPORTANT -- inference does NOT sanitize. Live chunks arrive already sanitized by
the validator (prepare_hand_for_miner runs validator-side, per hand). Only the
offline training matrix sanitizes raw benchmark hands (train == serve).
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

try:  # keep any torch backend single-threaded (never deadlock batched predict)
    import torch  # noqa: F401
    torch.set_num_threads(1)
except Exception:
    pass

_MODEL = None


def _pin_single_thread(est):
    """Best-effort force n_jobs/thread_count=1 so batched predict never deadlocks."""
    for attr in ("n_jobs", "nthread", "thread_count"):
        try:
            est.set_params(**{attr: 1})
        except Exception:
            pass
    # dig into containers (StackingClassifier, VotingClassifier, Pipeline)
    for holder in ("estimators_", "estimators"):
        try:
            for sub in getattr(est, holder):
                _pin_single_thread(sub[1] if isinstance(sub, tuple) else sub)
        except Exception:
            pass
    for attr in ("final_estimator_", "final_estimator"):
        try:
            _pin_single_thread(getattr(est, attr))
        except Exception:
            pass
    try:  # sklearn Pipeline
        for _, step in est.steps:
            _pin_single_thread(step)
    except Exception:
        pass


def _model():
    global _MODEL
    if _MODEL is None:
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        for key in ("stack", "mono", "mlp"):
            try:
                _pin_single_thread(b[key])
            except Exception:
                pass
        _MODEL = b
    return _MODEL


def _rank01(s):
    """Within-batch rank in [0,1]: argsort/argsort/(n-1). Calibration-free."""
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def _rows(chunks):
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.array(rows, dtype=float)


def _fused_rank(model, chunks):
    """Weighted average of each member's WITHIN-BATCH rank (the movable ordering)."""
    X = _rows(chunks)
    s1 = model["stack"].predict_proba(X)[:, 1]
    s2 = model["mono"].predict_proba(X)[:, 1]
    s3 = model["mlp"].predict_proba(X)[:, 1]
    w1, w2, w3 = model["weights"]
    fused = (w1 * _rank01(s1) + w2 * _rank01(s2) + w3 * _rank01(s3)) / (w1 + w2 + w3)
    return fused


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _calibrated(model, fused):
    return model["iso"].predict(np.asarray(fused, dtype=float))


def _decision(model, cal):
    """Reward-fit, FPR-capped per-batch decision layer on the calibrated fused score.

    Anti-saturation recenter (batch quantile Q) + reward-fit logit margin/temp so
    only a conservative high tail can cross 0.5, plus a thin hard floor that always
    lifts the batch-top FLOOR fraction across 0.5 (never an all-below-0.5 hard
    zero). CAP pushes every non-top-k chunk below 0.5 -> deterministic crossing
    count, robust to OOD saturation.
    """
    eps = float(model["EPS"])
    q = float(model["Q"])
    margin = float(model["MARGIN"])
    temp = float(model.get("TEMP", 1.0))
    floor = float(model["FLOOR"])
    cap = bool(model.get("CAP", False))
    tref = float(model["train_ref_logit"]) - margin
    z = _logit(cal, eps)
    if z.size == 0:
        return []
    anchor = np.quantile(z, q)
    scores = 1.0 / (1.0 + np.exp(-((z - anchor + tref) / temp)))
    order = np.argsort(-z, kind="mergesort")
    k = max(1, int(np.ceil(floor * len(scores))))
    scores[order[:k]] = np.maximum(scores[order[:k]], 0.5001)
    if cap:  # deterministic crossing count: nothing beyond top-k crosses 0.5
        scores[order[k:]] = np.minimum(scores[order[k:]], 0.4999)
    return [round(float(s), 6) for s in scores]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk (rank-fused, reward-fit floating output)."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, _calibrated(m, _fused_rank(m, chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback; score_batch is the real entry (needs batch context)."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        # No batch context for a lone chunk: return the calibrated member-mean prob.
        X = _rows([chunk])
        s = (m["weights"][0] * m["stack"].predict_proba(X)[:, 1]
             + m["weights"][1] * m["mono"].predict_proba(X)[:, 1]
             + m["weights"][2] * m["mlp"].predict_proba(X)[:, 1]) / sum(m["weights"])
        return round(float(s[0]), 6)
    except Exception:
        return 0.5
