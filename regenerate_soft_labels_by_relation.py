#!/usr/bin/env python3
"""
make_onehot_softlabels.py

Produces a JSON soft-label map keyed by "head_uri<TAB>relation_uri" -> { tail_uri: probability }

Usage (examples):
  python make_onehot_softlabels.py \
    --examples data/queer/valid_filtered_for_shortlist.json \
    --output data/shortlists/soft_labels_by_hr_onehot_counts.json \
    --relation-filter task_5_rnp

  # or to count across train+valid:
  python make_onehot_softlabels.py \
    --examples data/queer/train_aug.json data/queer/valid_filtered_for_shortlist.json \
    --output data/shortlists/soft_labels_by_hr_onehot_counts.json \
    --relation-filter task_5_rnp
"""
import argparse
import json
import os
from collections import defaultdict, Counter
from typing import List

POSSIBLE_HEAD_KEYS = ("head_id", "head", "h")
POSSIBLE_REL_KEYS = ("relation", "rel", "r")
POSSIBLE_TAIL_KEYS = ("tail_id", "tail", "t")

def load_examples(paths: List[str]):
    examples = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if not text:
                continue
            # try JSON array
            if text.startswith("["):
                objs = json.loads(text)
                examples.extend(objs)
            else:
                # fallback: lines of JSON objects or tab-separated triplets
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        # try to parse simple TSV "h\tr\tt"
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            examples.append({"head": parts[0], "relation": parts[1], "tail": parts[2]})
                        else:
                            # cannot parse: skip
                            continue
    return examples

def key_for_example(ex):
    # find head, relation, tail robustly
    head = None
    rel = None
    tail = None
    for k in POSSIBLE_HEAD_KEYS:
        if k in ex:
            head = ex[k]
            break
    for k in POSSIBLE_REL_KEYS:
        if k in ex:
            rel = ex[k]
            break
    for k in POSSIBLE_TAIL_KEYS:
        if k in ex:
            tail = ex[k]
            break
    # if any missing, try common raw keys
    if head is None and "h" in ex:
        head = ex.get("h")
    if rel is None and "relation" in ex:
        rel = ex.get("relation")
    if tail is None and "t" in ex:
        tail = ex.get("t")
    return head, rel, tail

def build_counts(examples, rel_filter=None):
    # mapping: key_str -> Counter(tail -> count)
    counts = defaultdict(Counter)
    for ex in examples:
        head, rel, tail = key_for_example(ex)
        if head is None or rel is None or tail is None:
            # skip malformed example
            continue
        if rel_filter:
            # accept partial match (user asked 'task_5_rnp'); using containment is robust to full URI vs fragment
            if rel_filter not in rel:
                continue
        key = f"{head}\t{rel}"
        counts[key][tail] += 1
    return counts

def normalize_counts(counts):
    out = {}
    for key, counter in counts.items():
        total = float(sum(counter.values()))
        if total <= 0:
            continue
        d = {}
        for tail, c in counter.items():
            d[tail] = c / total
        out[key] = d
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", "-e", nargs="+", required=True,
                        help="JSON file(s) containing examples (JSON array or ndjson).")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to write soft-label JSON (overwrites if exists).")
    parser.add_argument("--relation-filter", "-r", default=None,
                        help="Optional substring to filter relations (e.g. 'task_5_rnp'). If omitted, all relations are used.")
    args = parser.parse_args()

    examples = load_examples(args.examples)
    print(f"Loaded {len(examples)} example objects from {len(args.examples)} file(s).")
    counts = build_counts(examples, rel_filter=args.relation_filter)
    print(f"Built counts for {len(counts)} (head,relation) keys (after relation filter).")
    result = normalize_counts(counts)

    # write pretty JSON
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fo:
        json.dump(result, fo, indent=2, ensure_ascii=False)
    print(f"Wrote {len(result)} soft-label rows to {args.output}")

if __name__ == "__main__":
    main()