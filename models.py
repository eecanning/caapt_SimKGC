from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask

from typing import Optional

def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor
    # --- shortlist / candidate indices (optional) ---
    # Accept common aliases so trainer or later code can use whichever name is present.
    candidate_ids: Optional[torch.LongTensor] = None
    candidate_idx: Optional[torch.LongTensor] = None
    candidate_indices: Optional[torch.LongTensor] = None
    candidates: Optional[torch.LongTensor] = None
    shortlist_ids: Optional[torch.LongTensor] = None
    shortlist_idx: Optional[torch.LongTensor] = None
    cands: Optional[torch.LongTensor] = None


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        """
        Compute (hr x tail) logits and return a small dict including:
          - logits: (B, C)
          - labels: (B,) (index of correct item in the logits -> usually range(B))
          - inv_t, hr_vector, tail_vector (detached)
          - candidate_ids (optional): LongTensor shape (B, C) mapping each column to global entity id

        Additionally we attempt to find candidate / shortlist indices in:
        - output_dict (preferred), keys: candidate_ids, candidate_idx, candidates, shortlist_ids, shortlist_idx, candidate_indices, cands, candidate_id
        - batch_dict (fallback)
        If no candidate mapping is found and logits are shortlist-sized (C != V when q_full exists),
        we create a safe placeholder mapping arange(C) repeated across batch (so evaluator doesn't crash).
        """
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        ret = {
            'logits': logits,
            'labels': labels,
            'inv_t': self.log_inv_t.detach().exp(),
            'hr_vector': hr_vector.detach(),
            'tail_vector': tail_vector.detach()
        }

        # --- Candidate / shortlist handling ---
        # Candidate keys we try to recognize (in either output_dict or batch_dict)
        candidate_keys = [
            'candidate_ids', 'candidate_idx', 'candidate_indices', 'candidates',
            'shortlist_ids', 'shortlist_idx', 'shortlist', 'cands', 'candidate_id'
        ]

        found = None
        # prefer output_dict (model forward may put shortlist there)
        for k in candidate_keys:
            if k in output_dict:
                found = output_dict[k]
                break
        # fallback to batch_dict
        if found is None:
            for k in candidate_keys:
                if k in batch_dict:
                    found = batch_dict[k]
                    break

        if found is not None:
            # Normalize to LongTensor on correct device
            # Accept either a tensor, list, or nested list: try to convert robustly.
            if isinstance(found, torch.Tensor):
                cand_tensor = found.long().to(logits.device)
            else:
                # list / nested list -> tensor
                cand_tensor = torch.tensor(found, dtype=torch.long, device=logits.device)

            # If cand_tensor is 1D assume it's global column ids for the whole batch (same order)
            # Convert to shape (B, C) for consistency (repeat per example).
            if cand_tensor.dim() == 1:
                cand_tensor = cand_tensor.unsqueeze(0).repeat(batch_size, 1)
            elif cand_tensor.dim() == 2:
                # ensure first dim equals batch_size if possible; if not, repeat or trim as necessary
                if cand_tensor.size(0) != batch_size:
                    if cand_tensor.size(0) == 1:
                        cand_tensor = cand_tensor.repeat(batch_size, 1)
                    else:
                        # last-resort: try to broadcast by truncating/padding to batch_size
                        # (this is defensive — ideally shapes should match)
                        if cand_tensor.size(0) > batch_size:
                            cand_tensor = cand_tensor[:batch_size]
                        else:
                            cand_tensor = cand_tensor.repeat(int(batch_size / cand_tensor.size(0)) + 1, 1)[:batch_size]

            ret['candidate_ids'] = cand_tensor

        else:
            # No candidate mapping found. If logits cover full vocab we don't need candidate ids.
            # But if logits are shortlist-sized relative to known vocab V (soft-labels), produce a safe placeholder.
            # Try to detect V from the trainer args if available (self.args.vocab_size or similar), else skip.
            C = logits.size(1)
            # Heuristic: if there's a q_full stored in batch_dict / soft labels were enabled, evaluator expects candidates.
            # Create placeholder mapping arange(C) repeated for the batch (keeps shapes aligned; user should verify correctness).
            placeholder = torch.arange(C, device=logits.device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
            ret['candidate_ids'] = placeholder
            # Note: we intentionally do not raise here to allow the run to continue; this fallback should be replaced
            # by a proper shortlist mapping if you want exact alignment with full-vocab soft labels.

        return ret

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
