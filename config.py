# ===== File: config.py =====
import os
import random
import torch
import argparse
import warnings
import sys

import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='SimKGC arguments')

# --- existing core args ---
parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, metavar='N',
                    help='path to pretrained model')
parser.add_argument('--task', default='queer', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--train-path', default='', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--model-dir', default='', type=str, metavar='N',
                    help='path to model dir')
parser.add_argument('--warmup', default=400, type=int, metavar='N',
                    help='warmup steps')
parser.add_argument('--max-to-keep', default=5, type=int, metavar='N',
                    help='max number of checkpoints to keep')
parser.add_argument('--grad-clip', default=10.0, type=float, metavar='N',
                    help='gradient clipping')
parser.add_argument('--pooling', default='cls', type=str, metavar='N',
                    help='bert pooling')
parser.add_argument('--dropout', default=0.1, type=float, metavar='N',
                    help='dropout on final linear layer')
parser.add_argument('--use-amp', action='store_true',
                    help='Use amp if available')
parser.add_argument('--t', default=0.05, type=float,
                    help='temperature parameter')
parser.add_argument('--use-link-graph', action='store_true',
                    help='use neighbors from link graph as context')
parser.add_argument('--eval-every-n-step', default=10000, type=int,
                    help='evaluate every n steps')
parser.add_argument('--pre-batch', default=0, type=int,
                    help='number of pre-batch used for negatives')
parser.add_argument('--pre-batch-weight', default=0.5, type=float,
                    help='the weight for logits from pre-batch negatives')
parser.add_argument('--additive-margin', default=0.0, type=float, metavar='N',
                    help='additive margin for InfoNCE loss function')
parser.add_argument('--finetune-t', action='store_true',
                    help='make temperature as a trainable parameter or not')
parser.add_argument('--max-num-tokens', default=50, type=int,
                    help='maximum number of tokens')
parser.add_argument('--use-self-negative', action='store_true',
                    help='use head entity as negative')

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')

# only used for evaluation
parser.add_argument('--is-test', action='store_true',
                    help='is in test mode or not')
parser.add_argument('--rerank-n-hop', default=2, type=int,
                    help='use n-hops node for re-ranking entities, only used during evaluation')
parser.add_argument('--neighbor-weight', default=0.0, type=float,
                    help='weight for re-ranking entities')
parser.add_argument('--eval-model-path', default='', type=str, metavar='N',
                    help='path to model, only used for evaluation')

# --- NEW shortlist-related args (for training & eval parity) ---
parser.add_argument('--use-shortlist-eval', action='store_true',
                    help='Enable PEMLM-style shortlist evaluation during validation')
parser.add_argument('--shortlist-map-path', default='/content/SimKGC_caapt/data/shortlists/allowed_tails_by_relation_and_term.json', type=str,
                    help='path to shortlist mapping JSON')
parser.add_argument('--soft-labels-path', default='/content/SimKGC_caapt/data/shortlists/soft_labels_by_hr_onehot_counts.json', type=str,
                    help='path to soft-labels JSON (per-hr soft targets)')
parser.add_argument('--entity2id-path', default='/content/SimKGC_caapt/data/queer/entity2id.json', type=str,
                    help='path to entity2id.json used to map URIs->global ids')
parser.add_argument('--shortlist-term', default='queer', type=str,
                    help='term used to select per-term shortlist entries')
parser.add_argument('--shortlist-output-dir', default='', type=str,
                    help='output dir for shortlist CSVs (if empty uses model_dir)')
parser.add_argument('--shortlist-ece-bins', default=10, type=int,
                    help='number of bins for ECE calculation')

# --- NEW early stopping args (added per your request) ---
parser.add_argument('--early-stopping-patience', default=5, type=int,
                    help='number of validation checks with no improvement before stopping (default: 5)')
parser.add_argument('--es-relative-delta', default=0.001, type=float,
                    help='absolute additive improvement threshold for monitored metric (default: 0.001)')
parser.add_argument('--eval-every-epoch', action='store_true',
                    help='If set, run validation at the end of every epoch and apply early stopping there (overrides eval-every-n-step for epoch-end eval)')

# --- NOTE about parsing in notebook environments:
# Jupyter/Colab inject kernel args (e.g. "-f /.../json") which will confuse argparse.
# Use parse_known_args() so unknown kernel args are ignored.
args, _unknown = parser.parse_known_args()

# Backwards compatibility: accept either --use-shortlist-eval OR args.use_shortlist_eval
# (some external scripts set attributes with underscores). Normalize both ways.
if not hasattr(args, "use_shortlist_eval"):
    args.use_shortlist_eval = getattr(args, "use_shortlist_eval", False)
# also ensure attribute names with underscores exist for other shortlist settings (older code expects these names)
args.shortlist_map_path = getattr(args, "shortlist_map_path", args.shortlist_map_path)
args.soft_labels_path = getattr(args, "soft_labels_path", args.soft_labels_path)
args.entity2id_path = getattr(args, "entity2id_path", args.entity2id_path)
args.shortlist_term = getattr(args, "shortlist_term", args.shortlist_term)
args.shortlist_output_dir = getattr(args, "shortlist_output_dir", args.shortlist_output_dir)
args.shortlist_ece_bins = getattr(args, "shortlist_ece_bins", args.shortlist_ece_bins)

# --- existing validation & path checks (same as before) ---
assert not args.train_path or os.path.exists(args.train_path)
assert args.pooling in ['cls', 'mean', 'max']
assert args.task.lower() in ['wn18rr', 'fb15k237', 'wiki5m_ind', 'wiki5m_trans', 'all', 'black', 'gay', 'indian', 'orient', 'queer']
assert args.lr_scheduler in ['linear', 'cosine']

if args.model_dir:
    os.makedirs(args.model_dir, exist_ok=True)
else:
    # fallback: require eval_model_path to exist
    assert os.path.exists(args.eval_model_path), 'One of args.model_dir and args.eval_model_path should be valid path'
    args.model_dir = os.path.dirname(args.eval_model_path)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available, set use_amp=False')

if not torch.cuda.is_available():
    args.use_amp = False
    args.print_freq = 1
    warnings.warn('GPU is not available, set use_amp=False and print_freq=1')
