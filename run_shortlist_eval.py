# run_shortlist_eval.py -- wrapper that enforces shortlist eval settings (no CLI flags needed)
from pathlib import Path
# import args object from config and set required shortlist flags deterministically
from config import args

# --- Enforced shortlist evaluation settings ---
setattr(args, "use_shortlist_eval", True)
# Point to your filtered split (we created this earlier)
setattr(args, "valid_path", "data/queer/test_filtered_for_shortlist.json")
setattr(args, "train_path", "data/queer/train_aug.json")
setattr(args, "eval_model_path", "/content/drive/MyDrive/simkgc_checkpoints/queer_tiny_run/model_best.mdl")

# Shortlist / soft-label file locations (adjust if you placed them elsewhere)
setattr(args, "shortlist_map_path", "/content/SimKGC_caapt/data/shortlists/allowed_tails_by_relation_and_term.json")
setattr(args, "soft_labels_path", "/content/SimKGC_caapt/data/shortlists/soft_labels_by_hr_onehot_counts.json")
setattr(args, "shortlist_output_dir", "/content/drive/MyDrive/hybrid_task5/SIM-KGC/")
setattr(args, "shortlist_ece_bins", 10)
setattr(args, "entity2id_path", "/content/SimKGC_caapt/data/queer/entity2id.json")

# seed parity
setattr(args, "seed", 42)

# ensure evaluation mode
setattr(args, "is_test", True)

# Now import and call the evaluate entrypoint
import evaluate

if __name__ == "__main__":
    evaluate.predict_by_split()