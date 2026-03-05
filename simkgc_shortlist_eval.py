# simkgc_shortlist_eval.py
"""
Helper utilities to run PEMLM-style shortlist evaluation inside SimKGC.
"""

from typing import List, Dict, Tuple, Optional
import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random

def set_seed(seed: int = 42):
    import numpy as _np, random as _random, torch as _torch
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

def load_shortlist_map(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def load_soft_labels(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_allowed_tail_ids(entity2id_path, shortlist_map_path, relation_uri, shortlist_term="default"):
    """
    Load shortlist map and return (allowed_id_strings, allowed_global_ids).

    Robustness improvements:
      - Accept relation strings that are prefixed with "inverse " or "inv_" etc.
      - Try the raw URI first, then a few normalized candidates.
      - If relation appears as an inverse (e.g. "inverse https://..."), strip the prefix
        and try the base URI. Currently we return the same allowed-tail list for the inverse
        as for the forward relation (i.e. we do not swap heads<->tails). If you want swapped
        behavior, ask and I will update.
    """
    import json, logging
    from pathlib import Path

    shortlist_map_path = str(shortlist_map_path)
    p = Path(shortlist_map_path)
    if not p.exists():
        raise RuntimeError(f"Shortlist map not found at {shortlist_map_path}")

    with open(p, "r", encoding="utf-8") as fh:
        shortlist_map = json.load(fh)

    tried_candidates = []

    # If incoming relation looks like 'inverse https://...' strip the verbal prefix
    rel = relation_uri
    is_inverse = False
    # common textual prefixes
    for pref in ("inverse ", "inverse_", "inv_", "inv ", "inverse"):
        if isinstance(rel, str) and rel.startswith(pref):
            is_inverse = True
            rel = rel[len(pref):]
            break

    # Now build a list of candidate keys to try (order matters)
    candidates = []
    # 1) exact provided (original relation_uri)
    candidates.append(relation_uri)
    # 2) stripped version (if different)
    if rel != relation_uri:
        candidates.append(rel)
    # 3) raw URI (if relation_uri had a leading label like 'inverse ...', ensure raw rel present)
    if rel not in candidates:
        candidates.append(rel)
    # 4) try local-name variants: local name only, prefixed inv/local forms used in some outputs
    try:
        # derive local name
        if isinstance(rel, str):
            if "#" in rel:
                local = rel.split("#")[-1]
            else:
                local = rel.rstrip("/").split("/")[-1]
            candidates.append(local)
            candidates.append("inv_" + local)
            candidates.append("inverse_" + local)
            candidates.append("invt" + local)
            candidates.append("inverse " + local)
    except Exception:
        pass

    # Deduplicate while preserving order
    seen = set()
    candidates_norm = []
    for c in candidates:
        if c is None: continue
        if c in seen: continue
        seen.add(c)
        candidates_norm.append(c)

    # Try candidates
    for cand in candidates_norm:
        tried_candidates.append(cand)
        if cand in shortlist_map:
            allowed_id_strings = shortlist_map[cand].get(shortlist_term, shortlist_map[cand].get("default", []))
            # Ensure list
            if not isinstance(allowed_id_strings, list):
                raise RuntimeError(f"shortlist map entry for {cand} is not a list: {type(allowed_id_strings)}")
            # Load entity2id to map strings -> global numeric ids
            with open(entity2id_path, "r", encoding="utf-8") as fh:
                entity2id = json.load(fh)
            allowed_global_ids = []
            missing = []
            for s in allowed_id_strings:
                if s in entity2id:
                    allowed_global_ids.append(int(entity2id[s]))
                else:
                    missing.append(s)
            if missing:
                logging.getLogger(__name__).info(f"[shortlist] Missing tails in entity2id.json: {len(missing)}. Examples: {missing[:5]}")
            logging.getLogger(__name__).info(f"[shortlist] Checked {len(allowed_id_strings)} allowed tails for relation {cand} term {shortlist_term}")
            return allowed_id_strings, allowed_global_ids

    # nothing found
    raise RuntimeError(f"Relation {relation_uri} (tried candidates: {tried_candidates}) not present in shortlist map {shortlist_map_path}")

def expected_calibration_error(confidences: np.ndarray, hits: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() == 0:
            continue
        acc_bin = hits[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(conf_bin - acc_bin)
    return float(ece)

def compute_shortlist_metrics_from_logits(
        logits_all: torch.Tensor,
        candidate_tail_ids: torch.Tensor,
        hr_list: List[Tuple[int, str]],
        q_mat: torch.Tensor,
        id2entity: Dict[int, str],
        allowed_id_strings: List[str],
        tail_counts_per_hr: Optional[List[Dict[int, float]]] = None,
        batch_offset: int = 0,
        ece_bins: int = 10,
        device: str = "cpu"
    ) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Compute per-example shortlist metrics given full logits.
    """
    device_t = torch.device(device)
    logits_all = logits_all.to(device_t)
    candidate_tail_ids = candidate_tail_ids.to(device_t)

    shortlist_scores = logits_all[:, candidate_tail_ids]
    shortlist_scores = shortlist_scores - shortlist_scores.max(dim=1, keepdim=True).values
    log_p = F.log_softmax(shortlist_scores, dim=1)
    p = torch.exp(log_p)

    q = q_mat.to(device_t)
    eps = 1e-12
    q_safe = q.clamp(min=eps)

    ce_vec = -(q * log_p).sum(dim=1).detach().cpu().numpy()
    kl_vec = (q_safe * (torch.log(q_safe) - log_p)).sum(dim=1).detach().cpu().numpy()
    brier_vec = ((p - q) ** 2).sum(dim=1).detach().cpu().numpy()
    soft_acc_vec = (q * p).sum(dim=1).detach().cpu().numpy()

    top1_idx = torch.argmax(p, dim=1).detach().cpu().numpy()
    top1_conf = p.detach().cpu().numpy()[np.arange(p.shape[0]), top1_idx]

    support_idxs = [np.where(q[i].detach().cpu().numpy() > 0)[0].tolist() for i in range(q.shape[0])]

    confidences = []
    hits = []
    rows = []
    rows_triple = []

    allowed_global_ids = [int(x) for x in candidate_tail_ids.detach().cpu().tolist()]
    allowed_id_by_pos = {i: allowed_global_ids[i] for i in range(len(allowed_global_ids))}

    N = q.shape[0]
    for i in range(N):
        global_idx = batch_offset + i
        h_id, r_uri = hr_list[global_idx]
        pred_pos = int(top1_idx[i])
        pred_global = allowed_id_by_pos[pred_pos]
        conf = float(top1_conf[i])
        supp = support_idxs[i]
        hit = 1 if pred_pos in supp else 0

        confidences.append(conf)
        hits.append(hit)

        p_vec = p.detach().cpu().numpy()[i]
        tail_counts = (tail_counts_per_hr[global_idx] if (tail_counts_per_hr is not None and global_idx < len(tail_counts_per_hr)) else {allowed_global_ids[pred_pos]: 1})

        rows.append({
            "head": id2entity.get(h_id, str(h_id)),
            "relation": r_uri,
            "correct_tails": [id2entity.get(int(t), str(t)) for t in tail_counts.keys()],
            "correct_tail_weights": {id2entity.get(int(t), str(t)): float(c) / max(1.0, sum(tail_counts.values())) for t, c in tail_counts.items()},
            "predicted_tail_top1": id2entity.get(pred_global, str(pred_global)),
            "confidence_top1": conf,
            "hard_hit": hit,
            "soft_accuracy": float(soft_acc_vec[i]),
            "cross_entropy": float(ce_vec[i]),
            "kl_divergence": float(kl_vec[i]),
            "brier_score": float(brier_vec[i]),
        })

        true_gid = None
        for tid in tail_counts.keys():
            true_gid = int(tid); break

        rows_triple.append({
            "head": id2entity.get(h_id, str(h_id)),
            "relation": r_uri,
            "correct_tail": id2entity.get(true_gid, str(true_gid)) if true_gid is not None else "",
            "predicted_tail": id2entity.get(pred_global, str(pred_global)),
            "confidence": conf,
            "prob_true_tail": float(p_vec[supp[0]]) if (len(supp) > 0 and supp[0] < len(p_vec)) else float("nan"),
            "hard_hit": int(hit),
        })

    df_hr = pd.DataFrame(rows)
    df_triple = pd.DataFrame(rows_triple)

    confidences_arr = np.array(confidences)
    hits_arr = np.array(hits)
    ece_val = expected_calibration_error(confidences_arr, hits_arr, n_bins=ece_bins)

    model_metrics = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "n_examples": int(N),
        "mean_cross_entropy": float(np.mean(ce_vec)),
        "mean_kl_divergence": float(np.mean(kl_vec)),
        "mean_brier_score": float(np.mean(brier_vec)),
        "mean_soft_accuracy": float(np.mean(soft_acc_vec)),
        "hard_accuracy": float(np.mean(hits_arr)),
        "ece": float(ece_val)
    }

    return model_metrics, df_hr, df_triple