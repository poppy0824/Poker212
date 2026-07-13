"""Poker44 bot detector -- MEGABAG_C2: an 8-way BAG of decorrelated C2-180
3-member rank-fused ensembles (STACK+MONO+MLP over the identical 180-dim
sanitization-invariant C2 feature row -- features.py chunk_features/FEATURE_NAMES).

Why the mega-bag. The single C2-180 rank-fused ensemble (BEATER2, served on
uid12) discriminates at the top-tier benchmark AP (~0.92) but SWINGS round-to-round
on the live feed (R2 0.560 #1 -> R3 0.498 #73, swing 0.062) because a single OOD
draw of its 3 members is the highest-variance choice on the sanitized live data.
The competition composite is the MEAN of a miner's scored rounds, so a steadier
per-round score at equal level scores better than a spiky one. Averaging the
WITHIN-BATCH fused ranks of 8 distinct-seed sibling ensembles cuts the per-round
variance (~1/sqrt(8) on the decorrelated component) while holding the mean level,
taming the C2-180 spikiness toward the orderstat family's steadiness.

Each sub-ensemble emits its within-batch fused rank (argsort/argsort/(n-1) of a
0.35/0.30/0.35 weighted member blend); the 8 sub-ranks are AVERAGED, then topped
with the reused reward-fit FPR-capped floating decision layer
(Q=0.7, MARGIN=3.0, TEMP=1.0, FLOOR=0.02, CAP=True): deterministic ~2% cross 0.5,
zero hard-zeros, un-collapsed max ~0.99. Inference does NOT sanitize (live chunks
arrive already sanitized by the validator). All estimators pinned single-thread
(batched-predict deadlock guard); torch capped to 1 thread.
"""
from __future__ import annotations
import os
import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

try:
    import torch  # noqa: F401
    torch.set_num_threads(1)
except Exception:
    pass

_MODEL = None


def _pin_single_thread(est):
    for attr in ("n_jobs", "nthread", "thread_count"):
        try:
            est.set_params(**{attr: 1})
        except Exception:
            pass
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
    try:
        for _, step in est.steps:
            _pin_single_thread(step)
    except Exception:
        pass


def _model():
    global _MODEL
    if _MODEL is None:
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        for sub in b["subs"]:
            for key in ("stack", "mono", "mlp"):
                try:
                    _pin_single_thread(sub[key])
                except Exception:
                    pass
        _MODEL = b
    return _MODEL


def _rank01(s):
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


def _bag_fused(model, chunks):
    """Average the WITHIN-BATCH fused rank of every sub-ensemble (variance reduction)."""
    X = _rows(chunks)
    acc = None
    for sub in model["subs"]:
        w1, w2, w3 = sub["weights"]
        s1 = sub["stack"].predict_proba(X)[:, 1]
        s2 = sub["mono"].predict_proba(X)[:, 1]
        s3 = sub["mlp"].predict_proba(X)[:, 1]
        f = (w1 * _rank01(s1) + w2 * _rank01(s2) + w3 * _rank01(s3)) / (w1 + w2 + w3)
        acc = f if acc is None else acc + f
    return acc / len(model["subs"])


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _decision(model, cal):
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
    if cap:
        scores[order[k:]] = np.minimum(scores[order[k:]], 0.4999)
    return [round(float(s), 6) for s in scores]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk (8-bag rank-fused, reward-fit output)."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, m["iso"].predict(_bag_fused(m, chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback (no batch context): calibrated bag member-mean prob."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        X = _rows([chunk])
        acc = 0.0
        for sub in m["subs"]:
            w = sub["weights"]
            acc += (w[0] * sub["stack"].predict_proba(X)[:, 1]
                    + w[1] * sub["mono"].predict_proba(X)[:, 1]
                    + w[2] * sub["mlp"].predict_proba(X)[:, 1]) / sum(w)
        return round(float(acc[0] / len(m["subs"])), 6)
    except Exception:
        return 0.5
