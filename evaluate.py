import os
import json
import tqdm
import torch

from time import time
from typing import List, Tuple
from dataclasses import dataclass, asdict

from config import args
from doc import load_data, Example
from predict import BertPredictor
from dict_hub import get_entity_dict, get_all_triplet_dict
from triplet import EntityDict
from rerank import rerank_by_graph
from logger_config import logger


def _setup_entity_dict() -> EntityDict:
    if args.task == 'wiki5m_ind':
        return EntityDict(entity_dict_dir=os.path.dirname(args.valid_path),
                          inductive_test_path=args.valid_path)
    return get_entity_dict()


entity_dict = _setup_entity_dict()
all_triplet_dict = get_all_triplet_dict()


@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    topk_score_info: str
    rank: int
    correct: bool


@torch.no_grad()
def compute_metrics(hr_tensor: torch.tensor,
                    entities_tensor: torch.tensor,
                    target: List[int],
                    examples: List[Example],
                    k=3, batch_size=256) -> Tuple:
    assert hr_tensor.size(1) == entities_tensor.size(1)
    total = hr_tensor.size(0)
    entity_cnt = len(entity_dict)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(hr_tensor.device)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0

    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        # batch_size * entity_cnt
        batch_score = torch.mm(hr_tensor[start:end, :], entities_tensor.t())
        assert entity_cnt == batch_score.size(1)
        batch_target = target[start:end]

        # re-ranking based on topological structure
        rerank_by_graph(batch_score, examples[start:end], entity_dict=entity_dict)

        # filter known triplets
        for idx in range(batch_score.size(0)):
            mask_indices = []
            cur_ex = examples[start + idx]
            gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
            if len(gold_neighbor_ids) > 10000:
                logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
            for e_id in gold_neighbor_ids:
                if e_id == cur_ex.tail_id:
                    continue
                mask_indices.append(entity_dict.entity_to_idx(e_id))
            mask_indices = torch.LongTensor(mask_indices).to(batch_score.device)
            batch_score[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(batch_score, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == batch_score.size(0)
        for idx in range(batch_score.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :k].tolist())

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, ranks


def predict_by_split():
    """
    Driver for evaluation. When args.use_shortlist_eval is True we only run the
    forward direction (PEMLM-style shortlist evaluation parity). Otherwise we
    keep the original behavior of running both forward and backward and
    averaging the metrics.
    """
    assert os.path.exists(args.valid_path)
    assert os.path.exists(args.train_path)

    predictor = BertPredictor()
    predictor.load(ckt_path=args.eval_model_path)
    entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

    use_shortlist = getattr(args, "use_shortlist_eval", False)

    if use_shortlist:
        # Only run forward shortlist evaluation (PEMLM-KGC parity)
        logger.info("Shortlist evaluation enabled -> running only forward direction for parity with PEMLM-KGC.")
        forward_metrics = eval_single_direction(predictor,
                                                entity_tensor=entity_tensor,
                                                eval_forward=True)
        # When using shortlist we treat forward metrics as the final metrics
        metrics = forward_metrics

        # write metrics file (same format/location as before)
        prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
        split = os.path.basename(args.valid_path)
        with open('{}/metrics_{}_{}.json'.format(prefix, split, basename), 'w', encoding='utf-8') as writer:
            writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
            writer.write('backward metrics: {}\n'.format(json.dumps({})))
            writer.write('average metrics: {}\n'.format(json.dumps(metrics)))

        logger.info('Shortlist (forward-only) metrics written and evaluation complete.')
        return metrics

    # --- original behavior: both forward and backward, then average ---
    forward_metrics = eval_single_direction(predictor,
                                            entity_tensor=entity_tensor,
                                            eval_forward=True)
    backward_metrics = eval_single_direction(predictor,
                                             entity_tensor=entity_tensor,
                                             eval_forward=False)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/metrics_{}_{}.json'.format(prefix, split, basename), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))

    return metrics


def eval_single_direction(predictor: BertPredictor,
                          entity_tensor: torch.tensor,
                          eval_forward=True,
                          batch_size=256) -> dict:
    start_time = time()
    examples = load_data(args.valid_path, add_forward_triplet=eval_forward, add_backward_triplet=not eval_forward)

    hr_tensor, _ = predictor.predict_by_examples(examples)
    hr_tensor = hr_tensor.to(entity_tensor.device)
    target = [entity_dict.entity_to_idx(ex.tail_id) for ex in examples]
    logger.info('predict tensor done, compute metrics...')

    # --- START: PEMLM-style shortlist evaluation integration (robust) ---
    import numpy as _np
    import pandas as _pd
    import simkgc_shortlist_eval as ssleval
    from pathlib import Path

    ssleval.set_seed(42)

    default_shortlist_map = "/content/SimKGC_caapt/data/shortlists/allowed_tails_by_relation_and_term.json"
    default_soft_labels = "/content/SimKGC_caapt/data/shortlists/soft_labels_by_hr.json"
    default_entity2id = "/content/SimKGC_caapt/data/queer/entity2id.json"
    default_output_dir = "/content/drive/MyDrive/hybrid_task5/SIM-KGC/"

    use_shortlist = getattr(args, "use_shortlist_eval", False)
    shortlist_map_path = getattr(args, "shortlist_map_path", default_shortlist_map)
    soft_labels_path = getattr(args, "soft_labels_path", default_soft_labels)
    shortlist_term = getattr(args, "shortlist_term", "all")
    output_dir = getattr(args, "shortlist_output_dir", default_output_dir)
    ece_bins = getattr(args, "shortlist_ece_bins", 10)

    # Initialize holders for classic (full-ranking) outputs in case shortlist is not used
    topk_scores = topk_indices = ranks = None

    if use_shortlist:
        shortlist_map_path = str(Path(shortlist_map_path))
        soft_labels_path = str(Path(soft_labels_path))
        entity2id_path = getattr(args, "entity2id_path", default_entity2id)
        entity2id_path = str(Path(entity2id_path))

        if not Path(shortlist_map_path).exists():
            raise RuntimeError(f"Shortlist mapping not found at {shortlist_map_path}")

        # Load soft labels if present
        soft_labels = {}
        if Path(soft_labels_path).exists():
            with open(soft_labels_path, "r", encoding="utf-8") as fh:
                soft_labels = json.load(fh)
            logger.info(f"[shortlist] Loaded soft-labels entries: {len(soft_labels)}")
        else:
            logger.info("[shortlist] No soft-labels found; using one-hot targets.")

        # Load entity2id mapping
        if not Path(entity2id_path).exists():
            candidate = "/content/SimKGC_caapt/data/queer/entity2id.json"
            if Path(candidate).exists():
                entity2id_path = candidate
            else:
                raise RuntimeError(f"entity2id.json not found at {entity2id_path} or {candidate}")
        with open(entity2id_path, "r", encoding="utf-8") as fh:
            entity2id = json.load(fh)
        id2entity = {int(v): k for k, v in entity2id.items()}

        # Build hr_list (head numeric id, relation string)
        hr_list = []
        for ex in examples:
            try:
                h_idx = entity_dict.entity_to_idx(ex.head_id)
            except Exception:
                if ex.head_id in entity2id:
                    h_idx = int(entity2id[ex.head_id])
                else:
                    raise RuntimeError(f"Head id {ex.head_id} not found in entity_dict or entity2id.json")
            hr_list.append((h_idx, ex.relation))

        relation_uris = sorted({r for (_h, r) in hr_list})
        if len(relation_uris) == 0:
            raise RuntimeError("No relations found in examples for shortlist evaluation.")
        relation_uri = relation_uris[0]

        # Load shortlist map and attempt flexible key matches
        shortlist_map = json.load(open(shortlist_map_path, "r", encoding="utf-8"))
        tried_candidates = []
        allowed_id_strings = []
        relation_used_for_shortlist = None

        # Candidate forms to try
        tried_candidates.append(relation_uri)
        if isinstance(relation_uri, str):
            if "#" in relation_uri:
                ln = relation_uri.split("#")[-1]
            else:
                ln = relation_uri.rstrip("/").split("/")[-1]
            tried_candidates.extend([ln, "inv_" + ln, "inverse_" + ln, "inv" + ln, "inv_" + relation_uri])
        # Also try the "inv_" prefix on full URI
        tried_candidates.append("inv_" + relation_uri)
        # Now find first match in shortlist_map
        for cand in tried_candidates:
            if cand in shortlist_map:
                relation_used_for_shortlist = cand
                val = shortlist_map[cand]
                if isinstance(val, dict):
                    allowed_id_strings = val.get("default", [])
                else:
                    allowed_id_strings = val
                break

        if relation_used_for_shortlist is None:
            raise RuntimeError(f"Relation {relation_uri} (tried candidates: {tried_candidates}) not present in shortlist map {shortlist_map_path}")

        # Normalize allowed list and map to global ids
        if isinstance(allowed_id_strings, dict):
            allowed_id_strings = allowed_id_strings.get("default", list(allowed_id_strings.keys()))
        allowed_id_strings = list(allowed_id_strings)
        allowed_global_ids = []
        missing = []
        for s in allowed_id_strings:
            if s not in entity2id:
                missing.append(s)
            else:
                allowed_global_ids.append(int(entity2id[s]))
        if missing:
            logger.warning(f"[shortlist] Missing tails in entity2id.json: {len(missing)}. Examples: {missing[:5]}")
        logger.info(f"[shortlist] Checked {len(allowed_id_strings)} allowed tails for relation {relation_used_for_shortlist} term {shortlist_term}")

        # Build q_rows; append missing true tails dynamically if needed
        q_rows = []
        tail_counts_per_hr = []
        for ex_idx, ex in enumerate(examples):
            head_uri = ex.head_id
            key = f"{head_uri}\t{ex.relation}"
            if key in soft_labels:
                d = soft_labels[key]
                q_row = _np.zeros(len(allowed_global_ids), dtype=_np.float32)
                counts = {}
                for tail_uri, w in d.items():
                    if tail_uri not in entity2id:
                        continue
                    gid = int(entity2id[tail_uri])
                    if gid in allowed_global_ids:
                        pos = allowed_global_ids.index(gid)
                        q_row[pos] += float(w)
                        counts[gid] = counts.get(gid, 0.0) + float(w)
                s = q_row.sum()
                if s <= 0:
                    true_tail = ex.tail_id if hasattr(ex, "tail_id") else (ex.tail if hasattr(ex, "tail") else None)
                    if true_tail and true_tail in entity2id:
                        gid = int(entity2id[true_tail])
                        if gid not in allowed_global_ids:
                            allowed_global_ids.append(gid)
                            # expand previous q_rows
                            for i in range(len(q_rows)):
                                q_rows[i] = _np.pad(q_rows[i], (0,1), 'constant', constant_values=0.0)
                            q_row = _np.zeros(len(allowed_global_ids), dtype=_np.float32)
                            q_row[-1] = 1.0
                            counts = {gid: 1}
                            logger.warning(f"[shortlist] True tail {true_tail} (gid={gid}) missing from allowed shortlist; appended dynamically.")
                        else:
                            pos = allowed_global_ids.index(gid)
                            q_row = _np.zeros(len(allowed_global_ids), dtype=_np.float32)
                            q_row[pos] = 1.0
                            counts = {gid: 1}
                    else:
                        raise RuntimeError(f"True tail for example {ex_idx} (tail={true_tail}) not in entity2id.json; cannot construct q_row.")
                else:
                    q_row = q_row / s
                tail_counts_per_hr.append(counts)
                q_rows.append(q_row)
            else:
                q_row = _np.zeros(len(allowed_global_ids), dtype=_np.float32)
                true_tail = ex.tail_id if hasattr(ex, "tail_id") else (ex.tail if hasattr(ex, "tail") else None)
                if true_tail and true_tail in entity2id:
                    gid = int(entity2id[true_tail])
                    if gid not in allowed_global_ids:
                        allowed_global_ids.append(gid)
                        for i in range(len(q_rows)):
                            q_rows[i] = _np.pad(q_rows[i], (0,1), 'constant', constant_values=0.0)
                        q_row = _np.zeros(len(allowed_global_ids), dtype=_np.float32)
                        q_row[-1] = 1.0
                        tail_counts_per_hr.append({gid: 1})
                        q_rows.append(q_row)
                        logger.warning(f"[shortlist] True tail {true_tail} (gid={gid}) missing from allowed shortlist; appended dynamically.")
                    else:
                        pos = allowed_global_ids.index(gid)
                        q_row[pos] = 1.0
                        tail_counts_per_hr.append({gid: 1})
                        q_rows.append(q_row)
                else:
                    raise RuntimeError(f"True tail for example {ex_idx} (tail={true_tail}) not available to build q_row.")

        if len(q_rows) == 0:
            raise RuntimeError("No q_rows could be constructed for shortlist evaluation. Aborting.")

        candidate_tail_ids = torch.tensor(allowed_global_ids, dtype=torch.long, device=entity_tensor.device)
        q_mat = torch.tensor(_np.stack(q_rows, axis=0), dtype=torch.float32, device=entity_tensor.device)

        with torch.no_grad():
            logits_all = hr_tensor @ entity_tensor.t()

        model_metrics, df_hr, df_triple = ssleval.compute_shortlist_metrics_from_logits(
            logits_all=logits_all,
            candidate_tail_ids=candidate_tail_ids,
            hr_list=hr_list,
            q_mat=q_mat,
            id2entity=id2entity,
            allowed_id_strings=allowed_id_strings,
            tail_counts_per_hr=tail_counts_per_hr,
            ece_bins=ece_bins,
            device=str(entity_tensor.device)
        )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        term = shortlist_term
        focus = getattr(args, "focus", "detail")
        model_csv = Path(output_dir) / f"SIMKGC_{term}_{focus}_model_summary.csv"
        per_hr_csv = Path(output_dir) / f"SIMKGC_{term}_{focus}_results_per_hr.csv"
        per_triple_csv = Path(output_dir) / f"SIMKGC_{term}_{focus}_results_per_triple.csv"

        df_model = _pd.DataFrame([model_metrics])
        # Resolve model directory from eval_model_path (prefix may not be defined earlier in this scope)
        prefix = os.path.dirname(args.eval_model_path)
        basename = os.path.basename(args.eval_model_path)

        # Attempt to load training metadata (best/stopped epoch + best_val_loss)
        meta_path = Path(prefix) / "metadata.json"
        best_epoch = None
        stopped_epoch = None
        best_val_loss = None
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                best_epoch = meta.get("best_epoch", None)
                stopped_epoch = meta.get("stopped_epoch", None)
                best_val_loss = meta.get("best_val_loss", None)
                logger.info(f"[shortlist] Loaded training metadata from {meta_path}: {meta}")
            except Exception as e:
                logger.warning(f"[shortlist] Failed to read metadata at {meta_path}: {e}")

        # Add metadata columns to the one-row model summary DataFrame
        df_model["best_epoch"] = [best_epoch]
        df_model["stopped_epoch"] = [stopped_epoch]
        df_model["best_val_loss"] = [best_val_loss]

        df_model.to_csv(model_csv, index=False)
        
        df_hr.to_csv(per_hr_csv, index=False)
        df_triple.to_csv(per_triple_csv, index=False)

        logger.info(f"[shortlist] Wrote model summary to {model_csv}")
        logger.info(f"[shortlist] Wrote per-HR CSV to {per_hr_csv}")
        logger.info(f"[shortlist] Wrote per-triple CSV to {per_triple_csv}")

        metrics = {
            "mean_cross_entropy": round(model_metrics.get("mean_cross_entropy", float("nan")), 6),
            "mean_kl_divergence": round(model_metrics.get("mean_kl_divergence", float("nan")), 6),
            "mean_brier_score": round(model_metrics.get("mean_brier_score", float("nan")), 6),
            "mean_soft_accuracy": round(model_metrics.get("mean_soft_accuracy", float("nan")), 6),
            "hard_accuracy": round(model_metrics.get("hard_accuracy", float("nan")), 6),
            "ece": round(model_metrics.get("ece", float("nan")), 6),
        }
    else:
        topk_scores, topk_indices, metrics, ranks = compute_metrics(hr_tensor=hr_tensor, entities_tensor=entity_tensor,
                                                                    target=target, examples=examples,
                                                                    batch_size=batch_size)

    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    pred_infos = []
    if use_shortlist:
        # use df_triple for per-example information
        for idx, row in df_triple.iterrows():
            pred_info = PredInfo(
                head=row.get("head", ""),
                relation=row.get("relation", ""),
                tail=row.get("correct_tail", ""),
                pred_tail=row.get("predicted_tail", ""),
                pred_score=float(row.get("confidence", 0.0)),
                topk_score_info=json.dumps({}),
                rank=int(row.get("hard_hit", 0)),
                correct=bool(row.get("hard_hit", 0))
            )
            pred_infos.append(pred_info)
    else:
        for idx, ex in enumerate(examples):
            cur_topk_scores = topk_scores[idx]
            cur_topk_indices = topk_indices[idx]
            pred_idx = cur_topk_indices[0]
            cur_score_info = {entity_dict.get_entity_by_idx(topk_idx).entity: round(topk_score, 3)
                              for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices)}

            pred_info = PredInfo(head=ex.head, relation=ex.relation,
                                 tail=ex.tail, pred_tail=entity_dict.get_entity_by_idx(pred_idx).entity,
                                 pred_score=round(cur_topk_scores[0], 4),
                                 topk_score_info=json.dumps(cur_score_info),
                                 rank=ranks[idx],
                                 correct=pred_idx == target[idx])
            pred_infos.append(pred_info)

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return metrics

if __name__ == '__main__':
    predict_by_split()
