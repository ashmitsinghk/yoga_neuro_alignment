import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# REPRODUCIBILITY
# -----------------------------

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# PATH CONFIGURATION
# -----------------------------

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

ALIGNMENT_FILE = PROJECT_ROOT / "data/processed/alignment_pairs.json"

BASE_MODEL_NAME = "intfloat/e5-base-v2"
DAPT_MODEL_PATH = str(PROJECT_ROOT / "models/e5-yoga-neuro-dapt")
ALIGNED_MODEL_PATH = str(PROJECT_ROOT / "models/e5-yoga-neuro-aligned")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_RANDOM_NEGATIVES = 50

# -----------------------------
# LOAD ALIGNMENT PAIRS
# -----------------------------

with open(ALIGNMENT_FILE, "r", encoding="utf-8") as f:
    alignment_pairs = json.load(f)

anchors = [pair["anchor"] for pair in alignment_pairs]
positives = [pair["positive"] for pair in alignment_pairs]

print(f"Loaded {len(anchors)} alignment pairs.")

# -----------------------------
# LOAD MODELS
# -----------------------------

print("Loading models...")
base_model = SentenceTransformer(BASE_MODEL_NAME, device=DEVICE)
dapt_model = SentenceTransformer(DAPT_MODEL_PATH, device=DEVICE)
aligned_model = SentenceTransformer(ALIGNED_MODEL_PATH, device=DEVICE)

# -----------------------------
# MARGIN COMPUTATION
# -----------------------------

def compute_margins(model):
    anchor_emb = model.encode(anchors, convert_to_tensor=True)
    positive_emb = model.encode(positives, convert_to_tensor=True)

    margins = []

    for i in range(len(anchors)):
        anchor_vec = anchor_emb[i].cpu().numpy().reshape(1, -1)
        positive_vec = positive_emb[i].cpu().numpy().reshape(1, -1)

        aligned_sim = cosine_similarity(anchor_vec, positive_vec)[0][0]

        # Random negatives (deterministic because seed fixed)
        rand_indices = np.random.choice(
            len(positives),
            NUM_RANDOM_NEGATIVES,
            replace=True
        )

        rand_sims = []
        for idx in rand_indices:
            rand_vec = positive_emb[idx].cpu().numpy().reshape(1, -1)
            sim = cosine_similarity(anchor_vec, rand_vec)[0][0]
            rand_sims.append(sim)

        random_mean = np.mean(rand_sims)
        margin = aligned_sim - random_mean
        margins.append(margin)

    return np.array(margins)

# -----------------------------
# COMPUTE MARGINS
# -----------------------------

print("Computing margins...")

base_margins = compute_margins(base_model)
dapt_margins = compute_margins(dapt_model)
aligned_margins = compute_margins(aligned_model)

# Save per-pair margins
df = pd.DataFrame({
    "base_margin": base_margins,
    "dapt_margin": dapt_margins,
    "aligned_margin": aligned_margins
})

output_csv = PROJECT_ROOT / "experiments/results/per_pair_margins.csv"
df.to_csv(output_csv, index=False)
print(f"Saved per-pair margins to {output_csv}")

# -----------------------------
# FIGURE 4 — MARGIN DISTRIBUTION
# -----------------------------

plt.figure()
plt.boxplot([base_margins, dapt_margins, aligned_margins])
plt.xticks([1, 2, 3], ["Base", "DAPT", "Aligned"])
plt.ylabel("Discriminative Margin")
plt.title("Margin Distribution Across Model Variants")

output_fig1 = PROJECT_ROOT / "paper/figures/margin_distribution.png"
plt.savefig(output_fig1, dpi=300)
plt.close()

print(f"Saved margin distribution plot to {output_fig1}")

# -----------------------------
# PAIRED COHEN'S d (CORRECT DIRECTION)
# -----------------------------

def paired_cohens_d(improved, baseline):
    diff = improved - baseline
    return np.mean(diff) / np.std(diff, ddof=1)

d_base_dapt = paired_cohens_d(dapt_margins, base_margins)
d_base_aligned = paired_cohens_d(aligned_margins, base_margins)
d_dapt_aligned = paired_cohens_d(aligned_margins, dapt_margins)

effects = np.array([d_base_dapt, d_base_aligned, d_dapt_aligned])
labels = ["DAPT vs Base", "Aligned vs Base", "Aligned vs DAPT"]

# -----------------------------
# FIGURE 5 — EFFECT SIZE PLOT
# -----------------------------

plt.figure()
plt.axvline(0)
plt.scatter(effects, np.arange(len(effects)))
plt.yticks(np.arange(len(effects)), labels)
plt.xlabel("Paired Cohen's d")
plt.title("Effect Sizes Between Model Variants")

output_fig2 = PROJECT_ROOT / "paper/figures/effect_size_plot.png"
plt.savefig(output_fig2, dpi=300)
plt.close()

print(f"Saved effect size plot to {output_fig2}")

# -----------------------------
# PRINT VERIFIED RESULTS
# -----------------------------

print("\n===== VERIFIED RESULTS =====")
print("Base mean margin:", np.mean(base_margins))
print("DAPT mean margin:", np.mean(dapt_margins))
print("Aligned mean margin:", np.mean(aligned_margins))

print("\nPaired Cohen's d:")
print("DAPT vs Base:", d_base_dapt)
print("Aligned vs Base:", d_base_aligned)
print("Aligned vs DAPT:", d_dapt_aligned)
