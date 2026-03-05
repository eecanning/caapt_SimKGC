# ===== File: trainer.py =====
import glob
import json
import os
import torch
import shutil
import warnings

import torch.nn as nn
import torch.utils.data

from typing import Dict, Optional
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from doc import Dataset, collate
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer, get_entity_dict
from logger_config import logger
from pathlib import Path

# Keys to search for candidate id tensors (shortlist indices) in batch_dict / outputs
CANDIDATE_KEYS = [
    "candidate_ids", "candidate_idx", "candidates", "candidate_indices", "shortlist_ids", "shortlist",
    "cands", "candidate_id", "shortlist_idx"
]

# HR key formats to attempt when looking up q-matrix entries (matches your soft_labels_by_hr.json sample)
HR_KEY_FORMATS = [
    "{head}\t{rel}",   # head_uri<TAB>relation_uri
    "{head} #{rel}",   # fallback (not likely in your case)
]


class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # Option A artifacts
        self.soft_labels = None       # dict: hr_key -> { tail_uri: prob, ... }
        self.entity2id = None         # dict: entity_uri -> int
        self.vocab_size = None

        # Cache for entity embeddings used during validation (recomputed each epoch)
        self._cached_entity_tensor = None
        self._cached_entity_tensor_epoch = -1
        # how many entities to encode per batch when building entity_tensor
        self.entity_encode_batch = getattr(self.args, "entity_encode_batch", 512)

        # If soft labels path provided, load them now (used for Option A)
        if getattr(self.args, "soft_labels_path", ""):
            sp = self.args.soft_labels_path
            try:
                with open(sp, "r", encoding="utf-8") as fh:
                    # file is expected to be a JSON object mapping "head\trel" -> { tail_uri: prob }
                    self.soft_labels = json.load(fh)
                logger.info(f"[Shortlist] Loaded soft labels from {sp} (entries={len(self.soft_labels)})")
            except Exception as e:
                logger.warning(f"[Shortlist] Failed to load soft_labels_path {sp}: {e}")
                self.soft_labels = None

        # If entity2id_path provided, load mapping
        if getattr(self.args, "entity2id_path", ""):
            ep = self.args.entity2id_path
            try:
                with open(ep, "r", encoding="utf-8") as fh:
                    self.entity2id = json.load(fh)
                self.vocab_size = len(self.entity2id)
                logger.info(f"[Shortlist] Loaded entity2id (size={self.vocab_size}) from {ep}")
            except Exception as e:
                logger.warning(f"[Shortlist] Failed to load entity2id_path {ep}: {e}")
                self.entity2id = None
                self.vocab_size = None

        # --- NEW: try to load shortlist_map for trainer-side candidate injection (optional) ---
        self.shortlist_map = None
        shortlist_map_path = getattr(self.args, "shortlist_map_path", "")
        if shortlist_map_path:
            try:
                with open(shortlist_map_path, "r", encoding="utf-8") as fh:
                    self.shortlist_map = json.load(fh)
                logger.info(f"[Shortlist] Loaded shortlist_map from {shortlist_map_path} (relations={len(self.shortlist_map)})")
            except Exception as e:
                logger.warning(f"[Shortlist] Failed to load shortlist_map {shortlist_map_path}: {e}")
                self.shortlist_map = None
        else:
            logger.debug("[Shortlist] No shortlist_map_path configured in args; trainer will not inject global candidate_ids.")

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()

        # define loss function (criterion) and optimizer
        # for training we still use CE on integer labels (unless you later change training to soft labels)
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        train_dataset = Dataset(path=args.train_path, task=args.task)
        valid_dataset = Dataset(path=args.valid_path, task=args.task) if args.valid_path else None
        num_training_steps = args.epochs * len(train_dataset) // max(args.batch_size, 1)
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)

        # early stopping bookkeeping
        self.best_metric = None
        self.best_epoch = None
        self.no_improve_counter = 0
        self.stopped_epoch = None
        # convenience aliases for early-stopping config
        self.early_stopping_patience = getattr(self.args, "early_stopping_patience", 5)
        self.es_relative_delta = getattr(self.args, "es_relative_delta", 0.001)
        self.eval_every_epoch_flag = getattr(self.args, "eval_every_epoch", False)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

        self.train_loader = train_loader

        self.valid_loader = None
        if valid_dataset:
            self.valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=args.batch_size * 2,
                shuffle=True,
                collate_fn=collate,
                num_workers=args.workers,
                pin_memory=True)

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)

            # If user elected to evaluate at the end of every epoch, do so here and apply early stopping logic.
            if self.eval_every_epoch_flag and self.valid_loader:
                metric_dict, is_best = self._run_eval(epoch=epoch)
                # update counters based on improvement (is_best determined using es_relative_delta)
                if is_best:
                    self.no_improve_counter = 0
                else:
                    self.no_improve_counter += 1

                logger.info(f"[EarlyStopping] epoch={epoch} no_improve_counter={self.no_improve_counter} / patience={self.early_stopping_patience}")

                # If patience exhausted -> stop training
                if self.no_improve_counter >= self.early_stopping_patience:
                    self.stopped_epoch = epoch
                    logger.info(f"[EarlyStopping] Patience exceeded (no_improve_counter={self.no_improve_counter}). Stopping training at epoch {epoch}.")
                    self._save_metadata()  # save metadata file with best/stopped info
                    break
            else:
                # If not evaluating every epoch, we still rely on the existing behavior where
                # evals can be triggered every n steps inside train_epoch (unchanged).
                # Early stopping will not be applied in that mode by this implementation.
                pass

        # end for epochs
        # Save metadata even if we exit normally
        if self.stopped_epoch is None:
            # training finished normally
            self._save_metadata()
        logger.info("Training loop finished.")

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        """
        Run evaluation, determine whether this validation is the new 'best' according to
        validation loss (lower is better) and es_relative_delta. Save checkpoint as before.
        Returns (metric_dict, is_best).
        """
        metric_dict = self.eval_epoch(epoch)
        is_best = False

        if self.valid_loader:
            # Use 'loss' as the monitored metric (lower is better)
            new_loss = float(metric_dict.get('loss', 0.0))
            if self.best_metric is None:
                is_best = True
            else:
                prev_best_loss = float(self.best_metric.get('loss', float('inf')))
                # improvement test uses additive threshold: new_loss must be at least
                # es_relative_delta lower than previous best to count as improvement.
                if new_loss < prev_best_loss - float(self.es_relative_delta):
                    is_best = True

            if is_best:
                self.best_metric = metric_dict.copy()
                self.best_epoch = epoch

        # Save checkpoint (existing behavior)
        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

        logger.info(f"Eval done. epoch={epoch} loss={metric_dict.get('loss', None)} is_best={is_best}")
        return metric_dict, is_best

    def _example_get(self, item, key_candidates):
        """
        Extract a value from an example `item` which might be a dict or an object with attributes.
        Tries keys in order from key_candidates and returns first found value.
        """
        if item is None:
            return None
        # dict-like
        if isinstance(item, dict):
            for k in key_candidates:
                if k in item:
                    return item[k]
            return None
        # object-like: try attributes
        for k in key_candidates:
            if hasattr(item, k):
                return getattr(item, k)
        # also try get method for mapping-like objects
        if hasattr(item, "get"):
            for k in key_candidates:
                val = item.get(k, None)
                if val is not None:
                    return val
        return None

    # ------------------ NEW HELPER ------------------
    def _lookup_shortlist_allowed_global_ids(self, relation: str):
        """
        Given a relation URI/string, attempt to find the allowed tail URIs in self.shortlist_map,
        then map them to global integer ids using self.entity2id. Returns a list of ints (may be empty).
        Uses the same flexible key tries as evaluate.py (full URI, last segment, inv_ variants).
        """
        if not self.shortlist_map or not self.entity2id:
            return []

        tried_candidates = []
        if relation is None:
            return []

        # try full relation first
        tried_candidates.append(relation)
        if isinstance(relation, str):
            if "#" in relation:
                ln = relation.split("#")[-1]
            else:
                ln = relation.rstrip("/").split("/")[-1]
            tried_candidates.extend([ln, "inv_" + ln, "inverse_" + ln, "inv" + ln, "inv_" + relation])
        # also try prefixed inv_
        tried_candidates.append("inv_" + relation)

        for cand in tried_candidates:
            if cand in self.shortlist_map:
                val = self.shortlist_map[cand]
                if isinstance(val, dict):
                    allowed_id_strings = val.get("default", list(val.keys()))
                else:
                    allowed_id_strings = val
                # Normalize to list and map to ints using entity2id
                allowed_global_ids = []
                for s in list(allowed_id_strings):
                    if s in self.entity2id:
                        try:
                            allowed_global_ids.append(int(self.entity2id[s]))
                        except Exception:
                            continue
                return allowed_global_ids
        return []

    @torch.no_grad()
    def _build_entity_tensor_for_eval(self, device: torch.device, epoch: int):
        """
        Build (and cache per-epoch) a tensor of entity embeddings (V x D) by encoding
        the entity examples via the model's ent encoder. Returns tensor on `device`.

        This routine:
        - Attempts to use the repo `collate` for a chunk of EntityExample objects.
        - If collate fails because EntityExample is attribute-based (not subscriptable),
          falls back to tokenizing `EntityExample.entity_desc` strings with the repo tokenizer.
        - Caches the resulting entity tensor per-epoch (stored on CPU).
        """
        # If we already computed for this epoch, reuse
        if getattr(self, "_cached_entity_tensor_epoch", -1) == epoch and self._cached_entity_tensor is not None:
            return self._cached_entity_tensor.to(device)

        # Get entity examples
        entity_dict = get_entity_dict()
        entity_exs = getattr(entity_dict, "entity_exs", None)
        if entity_exs is None:
            raise RuntimeError("entity_exs not found in entity_dict; cannot build entity embeddings for shortlist CE.")

        ent_vectors = []
        total = len(entity_exs)
        bs = int(self.entity_encode_batch)
        model_obj = get_model_obj(self.model)

        for start in range(0, total, bs):
            chunk = entity_exs[start:start + bs]

            # First try the existing collate (keeps backward compat). If it fails because
            # EntityExample is not subscriptable, fall back to manual tokenization of entity_desc.
            try:
                batch_ent = collate(chunk)
            except Exception:
                # Fallback: EntityExample objects contain .entity_desc (raw text) and .entity_id.
                # Tokenize entity_desc using the repo tokenizer (build_tokenizer was called in __init__).
                # Try several ways to obtain the tokenizer object robustly.
                global_tokenizer = None
                try:
                    # common pattern: dict_hub exposes a tokenizer variable after build_tokenizer(args)
                    from dict_hub import tokenizer as _tk  # type: ignore
                    global_tokenizer = _tk
                except Exception:
                    global_tokenizer = None

                if global_tokenizer is None:
                    # Some implementations return the tokenizer from build_tokenizer or expose a getter.
                    try:
                        _maybe_tok = build_tokenizer(self.args)
                        if _maybe_tok is not None:
                            global_tokenizer = _maybe_tok
                    except Exception:
                        pass

                if global_tokenizer is None:
                    try:
                        from dict_hub import get_tokenizer  # type: ignore
                        global_tokenizer = get_tokenizer()
                    except Exception:
                        global_tokenizer = None

                if global_tokenizer is None:
                    raise RuntimeError("Tokenizer not found. Ensure dict_hub.build_tokenizer(args) was called and exposes a tokenizer.")

                # Collect texts from entity_ex objects (they have attribute entity_desc)
                texts = []
                for ex in chunk:
                    # defensive: accept entity_desc or entity if the description is missing
                    desc = getattr(ex, "entity_desc", None)
                    if desc is None:
                        desc = getattr(ex, "entity", None)
                    if desc is None:
                        raise RuntimeError("EntityExample missing both 'entity_desc' and 'entity' attributes; cannot tokenize.")
                    texts.append(str(desc))

                # Tokenize the list of texts in a single batch using the same max length as the rest of trainer.
                max_len = getattr(self.args, "max_num_tokens", 64)
                enc = global_tokenizer(texts, padding=True, truncation=True,
                                       max_length=max_len, return_tensors="pt")

                # Build batch_ent dictionary matching collate() output keys expected by predict_ent_embedding
                input_ids = enc["input_ids"]
                attention_mask = enc.get("attention_mask", (input_ids != 0).long())
                token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))

                batch_ent = {
                    "tail_token_ids": input_ids,
                    "tail_mask": attention_mask,
                    "tail_token_type_ids": token_type_ids
                }

            # move to device (compute on GPU if available) to use model efficiently
            batch_ent = move_to_cuda(batch_ent) if torch.cuda.is_available() else batch_ent

            # call model's predict_ent_embedding (will return {'ent_vectors': tensor})
            ent_out = model_obj.predict_ent_embedding(
                tail_token_ids=batch_ent['tail_token_ids'],
                tail_mask=batch_ent['tail_mask'],
                tail_token_type_ids=batch_ent['tail_token_type_ids']
            )

            ev = ent_out.get('ent_vectors') if isinstance(ent_out, dict) else ent_out
            if ev is None:
                raise RuntimeError("predict_ent_embedding did not return 'ent_vectors' as expected.")
            # move to CPU to reduce GPU memory pressure and append
            ent_vectors.append(ev.cpu())

        # concat on CPU then move to desired device
        entity_tensor = torch.cat(ent_vectors, dim=0).to(device)
        # cache (store on CPU and mark epoch)
        self._cached_entity_tensor = entity_tensor.cpu()
        self._cached_entity_tensor_epoch = epoch
        return self._cached_entity_tensor.to(device)

    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        if not self.valid_loader:
            return {}

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        # If using soft labels Option A, ensure we have the mapping loaded
        use_soft = (self.soft_labels is not None and self.entity2id is not None and self.vocab_size is not None)
        if use_soft:
            logger.info(f"[Eval] Soft labels enabled. Vocab_size={self.vocab_size}")

        # Precompute entity_tensor for this validation epoch if we will need it (Option A)
        need_entity_tensor = use_soft and (self.shortlist_map is not None or getattr(self.args, "shortlist_map_path", "") != "")
        entity_tensor = None
        if need_entity_tensor:
            # compute entity_tensor on the device we'll use for HR vectors; pick model device if available
            # We'll compute entity tensor on CPU then move as needed to batch device during loop
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            try:
                entity_tensor = self._build_entity_tensor_for_eval(device=dev, epoch=epoch)
                logger.info(f"[Eval] Built entity_tensor for eval: shape={tuple(entity_tensor.size())} device={entity_tensor.device}")
            except Exception as e:
                logger.exception(f"[Eval] Failed to build entity_tensor for eval: {e}")
                # fallback: leave entity_tensor None; downstream code will raise if entity_tensor is required
                entity_tensor = None

        for i, batch_dict in enumerate(self.valid_loader):
            if i == 0:
                logger.info(f"[DEBUG] batch_dict keys: {list(batch_dict.keys())}")
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            # ------------------ TRAINER-SIDE: inject global candidate_ids if possible ------------------
            # This is helpful so candidate lists are present for each example (used below when selecting shortlist columns).
            if 'candidate_ids' not in batch_dict and self.shortlist_map is not None:
                batch_examples = batch_dict.get('batch_data', None)
                if batch_examples is None:
                    logger.debug("[Shortlist] batch_data missing; cannot build candidate_ids for this batch.")
                else:
                    cand_lists = []
                    for item in batch_examples:
                        rel = self._example_get(item, ["relation", "rel", "r"])
                        allowed = self._lookup_shortlist_allowed_global_ids(rel)
                        if not allowed:
                            # fallback: try to include the true tail for this example if available
                            true_tail = self._example_get(item, ["tail_id", "tail"])
                            if true_tail and self.entity2id and (true_tail in self.entity2id):
                                allowed = [int(self.entity2id[true_tail])]
                                logger.debug(f"[Shortlist] relation {rel}: no allowed tails found; using true tail {true_tail}.")
                            else:
                                # final fallback: pick a safe single id (0) to avoid empty lists — log once
                                allowed = [0]
                                logger.debug(f"[Shortlist] relation {rel}: no allowed tails and true tail unavailable; using fallback id 0.")
                        cand_lists.append(allowed)

                    # If every example shares the identical candidate list, store as 1-D (compute_logits will broadcast)
                    first = cand_lists[0]
                    all_same = all(lst == first for lst in cand_lists)
                    if all_same:
                        batch_dict['candidate_ids'] = first  # list of ints
                    else:
                        # Pad variable-length rows to a rectangular matrix using last-id padding
                        maxlen = max(len(lst) for lst in cand_lists)
                        padded = []
                        for lst in cand_lists:
                            if len(lst) < maxlen:
                                pad_val = lst[-1] if len(lst) > 0 else 0
                                padded.append(lst + [pad_val] * (maxlen - len(lst)))
                            else:
                                padded.append(lst[:maxlen])
                        batch_dict['candidate_ids'] = padded  # list-of-lists (B x C)

            # ------------------ NEW: Defensive removal when entity_tensor missing ------------------
            # If we do NOT have entity_tensor (so we'll use the fallback gather path),
            # remove any trainer-injected global candidate_ids so compute_logits() will
            # produce a placeholder mapping that aligns with the model's logits columns.
            # This prevents column-count mismatches (q_batch vs logits).
            if entity_tensor is None:
                removed = False
                for k in CANDIDATE_KEYS:
                    if k in batch_dict:
                        # remove injected candidate lists so model can emit appropriate placeholder
                        del batch_dict[k]
                        removed = True
                if removed:
                    logger.debug("[Shortlist] Removed trainer-injected candidate_ids because entity_tensor unavailable; allowing model to emit placeholder arange(C).")

            # compute model forward to obtain hr_vector (we will compute logits_all against entity_tensor)
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs_raw = self.model(**batch_dict)
            else:
                outputs_raw = self.model(**batch_dict)

            # If we're in the soft-shortlist evaluation mode and have entity_tensor available,
            # compute logits over full vocab and then select shortlist columns using candidate global ids.
            if use_soft and (entity_tensor is not None):
                # hr_vector from model forward; ensure it's on same device as entity_tensor for matmul
                hr_vector = outputs_raw.get('hr_vector')
                if hr_vector is None:
                    raise RuntimeError("Model forward did not return 'hr_vector' required for full-vocab logits computation.")
                # move hr_vector to entity_tensor device
                hr_vec_dev = hr_vector.to(entity_tensor.device)
                # logits_all: (B, V)
                logits_all = hr_vec_dev @ entity_tensor.t()

                # Now prepare q_full as before (B, V) using self.soft_labels and self.entity2id
                V = self.vocab_size
                device = logits_all.device
                q_full = torch.zeros((batch_size, V), dtype=torch.float32, device=device)

                # build q_full rows using batch examples
                batch_examples = batch_dict.get('batch_data', None)
                if batch_examples is None:
                    batch_examples = getattr(batch_dict, 'batch_data', None)
                if batch_examples is None:
                    raise RuntimeError("Missing batch_data in batch_dict used for eval; cannot align soft labels.")

                for ridx in range(batch_size):
                    item = batch_examples[ridx]
                    head = self._example_get(item, ["head_id", "head", "h"])
                    rel = self._example_get(item, ["relation", "rel", "r"])
                    found = False
                    if head is None or rel is None:
                        continue
                    for fmt in HR_KEY_FORMATS:
                        hr_key = fmt.format(head=head, rel=rel)
                        entry = self.soft_labels.get(hr_key) if isinstance(self.soft_labels, dict) else None
                        if entry:
                            for tail_uri, p in entry.items():
                                eid = self.entity2id.get(tail_uri)
                                if eid is None:
                                    continue
                                try:
                                    val = float(p)
                                except Exception:
                                    continue
                                q_full[ridx, int(eid)] = val
                            found = True
                            break
                    if not found:
                        if (ridx % 50) == 0:
                            logger.debug(f"[Eval] soft-label: no entry for head={head} rel={rel} (row {ridx})")

                # Candidate ids must refer to global entity ids (0..V-1). Use those to extract logits_short and q_batch.
                candidate_ids = None
                # prefer trainer-injected batch_dict value
                for k in CANDIDATE_KEYS:
                    c = batch_dict.get(k, None)
                    if c is not None:
                        candidate_ids = c
                        break
                # fallback to outputs_raw if model provided something
                if candidate_ids is None:
                    for k in CANDIDATE_KEYS:
                        if isinstance(outputs_raw, dict) and k in outputs_raw:
                            candidate_ids = outputs_raw[k]
                            break

                if candidate_ids is None:
                    raise RuntimeError("candidate_ids not found in batch_dict or model outputs while computing shortlist CE against full-vocab logits.")

                # coerce candidate_ids to tensor on same device as logits_all
                if isinstance(candidate_ids, (list, tuple)):
                    candidate_ids = torch.tensor(candidate_ids, dtype=torch.long, device=logits_all.device)
                elif isinstance(candidate_ids, torch.Tensor):
                    candidate_ids = candidate_ids.to(device=logits_all.device, dtype=torch.long)
                else:
                    try:
                        candidate_ids = torch.tensor(candidate_ids, dtype=torch.long, device=logits_all.device)
                    except Exception:
                        raise RuntimeError(f"candidate_ids found but couldn't coerce to tensor. type={type(candidate_ids)}")

                if candidate_ids.dim() == 1:
                    candidate_ids = candidate_ids.unsqueeze(0).expand(batch_size, -1)

                if candidate_ids.size(0) != batch_size:
                    if candidate_ids.size(0) == 1:
                        candidate_ids = candidate_ids.expand(batch_size, -1)
                    else:
                        raise RuntimeError(f"candidate_ids first dimension {candidate_ids.size(0)} != batch_size {batch_size}")

                # Select shortlist logits (B, C) from logits_all (B, V) using global ids
                logits = torch.gather(logits_all, 1, candidate_ids)

                # Build q_batch by gathering q_full using the same candidate_ids (global ids)
                q_batch = torch.gather(q_full, 1, candidate_ids)

                # Diagnostic (first batch)
                if i == 0:
                    try:
                        qfull_sum = float(q_full.sum().item())
                        qfull_nonzero = int((q_full != 0).sum().item())
                        cand_shape = tuple(candidate_ids.size())
                        sample_cands = candidate_ids[:3, :5].cpu().tolist()
                        qbatch_row_sums = q_batch.sum(dim=1)[:4].cpu().tolist()
                        V = logits_all.size(1)
                        C = logits.size(1)
                        logger.info(f"[Shortlist-Diag] batch_size={batch_size} V={V} C={C} candidate_ids_found=True cand_shape={cand_shape}")
                        logger.info(f"[Shortlist-Diag] sample candidate_ids (first rows): {sample_cands}")
                        logger.info(f"[Shortlist-Diag] q_full.sum={qfull_sum:.6g} q_full.nonzero_count={qfull_nonzero}")
                        logger.info(f"[Shortlist-Diag] q_batch row sums (first 4) = {qbatch_row_sums}")
                    except Exception as e:
                        logger.exception(f"[Shortlist-Diag] diagnostic failed: {e}")

                # proceed to compute CE on logits (B, C) vs q_batch (B, C)
                log_p = torch.nn.functional.log_softmax(logits, dim=1)
                q_batch = q_batch.to(device=log_p.device, dtype=log_p.dtype)
                ce_per_example = -(q_batch * log_p).sum(dim=1)
                loss = ce_per_example.mean()
                losses.update(loss.item(), batch_size)

                # update acc metrics using integer labels if possible (we need label index within shortlist)
                try:
                    # labels in this path: we must map integer label (global id) to shortlist index in candidate_ids
                    labels_global = torch.LongTensor([self.entity2id.get(self._example_get(ex, ["tail_id", "tail"])) for ex in batch_dict['batch_data']])
                    labels_global = labels_global.to(candidate_ids.device)
                    # find per-row positions where candidate_ids == labels_global.unsqueeze(1)
                    eq = candidate_ids.eq(labels_global.unsqueeze(1))
                    # get index of True in each row
                    pos = torch.nonzero(eq, as_tuple=False)
                    # build integer label indices with default -1 if missing
                    label_idx = torch.full((batch_size,), -1, dtype=torch.long, device=candidate_ids.device)
                    if pos.numel() > 0:
                        # pos rows are (row_idx, col_idx) pairs
                        for r, c in pos.tolist():
                            label_idx[r] = c
                    # where missing label_idx == -1, set to 0 to avoid accuracy failures
                    label_idx_clamped = label_idx.clone()
                    label_idx_clamped[label_idx_clamped < 0] = 0
                    acc1, acc3 = accuracy(logits, label_idx_clamped, topk=(1, 3))
                    top1.update(acc1.item(), batch_size)
                    top3.update(acc3.item(), batch_size)
                except Exception:
                    # fallback: skip accuracy update for this batch if mapping fails
                    pass

                # finished this batch's soft-shortlist CE path; continue to next batch
                continue

            # --- FALLBACK: original path (no soft labels OR entity_tensor unavailable) ---
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs_raw, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels  # logits: (B, C) labels: (B,) int targets

            # If no soft labels requested, fall back to standard CE evaluation
            if not use_soft:
                loss = self.criterion(logits, labels)
                losses.update(loss.item(), batch_size)
                acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
                top1.update(acc1.item(), batch_size)
                top3.update(acc3.item(), batch_size)
                continue

            # If use_soft is True but entity_tensor was not available, we attempt the original q_full gather path
            # (this may still fail if candidate_ids are local indices; left as-is for backward compatibility)
            # --- Build q_batch_full of shape (B, V) using self.soft_labels and self.entity2id ---
            V = self.vocab_size
            device = logits.device
            q_full = torch.zeros((batch_size, V), dtype=torch.float32, device=device)

            # build q_full rows using batch examples
            batch_examples = batch_dict.get('batch_data', None)
            if batch_examples is None:
                batch_examples = getattr(batch_dict, 'batch_data', None)
            if batch_examples is None:
                raise RuntimeError("Missing batch_data in batch_dict used for eval; cannot align soft labels.")

            for ridx in range(batch_size):
                item = batch_examples[ridx]
                head = self._example_get(item, ["head_id", "head", "h"])
                rel = self._example_get(item, ["relation", "rel", "r"])
                found = False
                if head is None or rel is None:
                    continue
                for fmt in HR_KEY_FORMATS:
                    hr_key = fmt.format(head=head, rel=rel)
                    entry = self.soft_labels.get(hr_key) if isinstance(self.soft_labels, dict) else None
                    if entry:
                        for tail_uri, p in entry.items():
                            eid = self.entity2id.get(tail_uri)
                            if eid is None:
                                continue
                            try:
                                val = float(p)
                            except Exception:
                                continue
                            q_full[ridx, int(eid)] = val
                        found = True
                        break
                if not found:
                    if (ridx % 50) == 0:
                        logger.debug(f"[Eval] soft-label: no entry for head={head} rel={rel} (row {ridx})")

            # Now we have q_full (B, V); align with logits columns using candidate_ids found in batch_dict/outputs
            C = logits.size(1)
            if C == V:
                q_batch = q_full
            else:
                candidate_ids = None
                for k in CANDIDATE_KEYS:
                    c = batch_dict.get(k, None)
                    if c is not None:
                        candidate_ids = c
                        break
                if candidate_ids is None:
                    for k in CANDIDATE_KEYS:
                        if hasattr(outputs, k):
                            candidate_ids = getattr(outputs, k)
                            break
                if candidate_ids is None and isinstance(outputs_raw, dict):
                    for k in CANDIDATE_KEYS:
                        if k in outputs_raw:
                            candidate_ids = outputs_raw[k]
                            break
                if candidate_ids is None:
                    available_keys = list(batch_dict.keys()) + ([k for k in dir(outputs) if not k.startswith("_")])
                    raise RuntimeError(
                        f"Shape mismatch: q_full has V={V} but logits have C={C} columns and no candidate id list was found.\n"
                        f"Please ensure your model returns shortlist indices per example under one of: {CANDIDATE_KEYS}\n"
                        f"Available batch_dict keys: {list(batch_dict.keys())}\n"
                    )

                if isinstance(candidate_ids, (list, tuple)):
                    candidate_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
                elif isinstance(candidate_ids, torch.Tensor):
                    candidate_ids = candidate_ids.to(device=device, dtype=torch.long)
                else:
                    try:
                        candidate_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
                    except Exception:
                        raise RuntimeError(f"candidate_ids found but couldn't coerce to tensor. type={type(candidate_ids)}")

                if candidate_ids.dim() == 1:
                    candidate_ids = candidate_ids.unsqueeze(0).expand(batch_size, -1)

                if candidate_ids.size(0) != batch_size:
                    if candidate_ids.size(0) == 1:
                        candidate_ids = candidate_ids.expand(batch_size, -1)
                    else:
                        raise RuntimeError(f"candidate_ids first dimension {candidate_ids.size(0)} != batch_size {batch_size}")

                q_batch = torch.gather(q_full, 1, candidate_ids)

                if q_batch.size(1) != C:
                    raise RuntimeError(f"After gathering soft labels across candidate_ids, q_batch has {q_batch.size(1)} columns but logits has {C} columns.")

            # Diagnostic for fallback path (first batch)
            if i == 0:
                try:
                    candidate_ids_local = locals().get('candidate_ids', None)
                    C = logits.size(1)
                    cand_found = candidate_ids_local is not None
                    cand_shape = None
                    sample_cands = None
                    if isinstance(candidate_ids_local, torch.Tensor):
                        cand_shape = tuple(candidate_ids_local.size())
                        r = min(3, cand_shape[0])
                        c = min(5, cand_shape[1]) if len(cand_shape) > 1 else 0
                        if r > 0 and c > 0:
                            sample_cands = candidate_ids_local[:r, :c].cpu().tolist()
                    else:
                        cand_shape = str(type(candidate_ids_local))

                    qfull_sum = float(q_full.sum().item())
                    qfull_nonzero = int((q_full != 0).sum().item())
                    qbatch_row_sums = q_batch.sum(dim=1)[:4].cpu().tolist()

                    logger.info(f"[Shortlist-Diag] batch_size={batch_size} V={V} C={C} candidate_ids_found={cand_found} cand_shape={cand_shape}")
                    logger.info(f"[Shortlist-Diag] sample candidate_ids (first rows): {sample_cands}")
                    logger.info(f"[Shortlist-Diag] q_full.sum={qfull_sum:.6g} q_full.nonzero_count={qfull_nonzero}")
                    logger.info(f"[Shortlist-Diag] q_batch row sums (first 4) = {qbatch_row_sums}")
                except Exception as e:
                    logger.exception(f"[Shortlist-Diag] diagnostic failed: {e}")

            # compute negative log-likelihood for soft targets (per-example) for fallback path
            log_p = torch.nn.functional.log_softmax(logits, dim=1)
            q_batch = q_batch.to(device=device, dtype=log_p.dtype)
            ce_per_example = -(q_batch * log_p).sum(dim=1)
            loss = ce_per_example.mean()
            losses.update(loss.item(), batch_size)

            # For reporting keep ints acc (if labels present)
            try:
                acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
                top1.update(acc1.item(), batch_size)
                top3.update(acc3.item(), batch_size)
            except Exception:
                pass

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))

        for i, batch_dict in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            batch_size = len(batch_dict['batch_data'])

            # compute output
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_dict)
            else:
                outputs = self.model(**batch_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=batch_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            assert logits.size(0) == batch_size
            # head + relation -> tail
            loss = self.criterion(logits, labels)
            # tail -> head + relation
            # this assumes logits[:, :batch_size] shapes are compatible; keep previous behavior
            loss += self.criterion(logits[:, :batch_size].t(), labels)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            inv_t.update(outputs.inv_t, 1)
            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            if i % self.args.print_freq == 0:
                progress.display(i)
            # keep the original behavior that allows periodic evals inside epoch
            if (i + 1) % self.args.eval_every_n_step == 0:
                # Previously this call saved checkpoints and updated best if metric improved.
                # We keep that behavior but it will not feed back to early-stopping counters
                # unless eval_every_epoch mode is used (user asked to focus first on epoch-end ES).
                self._run_eval(epoch=epoch, step=i + 1)
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)

    # ------------------ metadata utilities ------------------
    def _save_metadata(self):
        """Write metadata JSON to model_dir/metadata.json similar to PEMLM style."""
        meta = {
            "best_epoch": self.best_epoch,
            "stopped_epoch": self.stopped_epoch,
            "best_val_loss": None
        }
        if self.best_metric and "loss" in self.best_metric:
            # store the numeric (float) loss for easy downstream comparison
            meta["best_val_loss"] = float(self.best_metric.get("loss", None))
        outpath = f"{self.args.model_dir}/metadata.json"
        try:
            with open(outpath, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
            logger.info(f"Saved metadata to {outpath}: {meta}")
        except Exception as e:
            logger.warning(f"Failed to write metadata to {outpath}: {e}")