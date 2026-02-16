import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os

# -----------------------------
# CONFIG
# -----------------------------

# Get script directory and project root
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

base_model = SentenceTransformer(BASE_MODEL_NAME, device=DEVICE)
dapt_model = SentenceTransformer(DAPT_MODEL_PATH, device=DEVICE)
aligned_model = SentenceTransformer(ALIGNED_MODEL_PATH, device=DEVICE)

# -----------------------------
# FUNCTION TO COMPUTE MARGINS
# -----------------------------

def compute_margins(model):
    anchor_emb = model.encode(anchors, convert_to_tensor=True)
    positive_emb = model.encode(positives, convert_to_tensor=True)

    margins = []

    for i in range(len(anchors)):
        aligned_sim = cosine_similarity(
            anchor_emb[i].cpu().numpy().reshape(1, -1),
            positive_emb[i].cpu().numpy().reshape(1, -1)
        )[0][0]

        # random negatives
        rand_indices = np.random.choice(len(positives), NUM_RANDOM_NEGATIVES, replace=True)
        rand_sims = []

        for idx in rand_indices:
            sim = cosine_similarity(
                anchor_emb[i].cpu().numpy().reshape(1, -1),
                positive_emb[idx].cpu().numpy().reshape(1, -1)
            )[0][0]
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

# Save margins
df = pd.DataFrame({
    "base_margin": base_margins,
    "dapt_margin": dapt_margins,
    "aligned_margin": aligned_margins
})

output_csv = PROJECT_ROOT / "experiments/results/per_pair_margins.csv"
df.to_csv(output_csv, index=False)
print(f"Saved {output_csv}")

# -----------------------------
# FIGURE 4: MARGIN DISTRIBUTION
# -----------------------------

plt.figure()
plt.boxplot([base_margins, dapt_margins, aligned_margins])
plt.xticks([1, 2, 3], ["Base", "DAPT", "Aligned"])
plt.ylabel("Discriminative Margin")
plt.title("Margin Distribution Across Model Variants")
output_fig1 = PROJECT_ROOT / "paper/figures/margin_distribution.png"
plt.savefig(output_fig1, dpi=300)
plt.close()

print(f"Saved {output_fig1}")

# -----------------------------
# EFFECT SIZE COMPUTATION
# -----------------------------

def cohens_d(a, b):
    diff = a - b
    return np.mean(diff) / np.std(diff)

d_base_dapt = cohens_d(base_margins, dapt_margins)
d_base_aligned = cohens_d(base_margins, aligned_margins)
d_dapt_aligned = cohens_d(dapt_margins, aligned_margins)

effects = np.array([d_base_dapt, d_base_aligned, d_dapt_aligned])
labels = ["Base vs DAPT", "Base vs Aligned", "DAPT vs Aligned"]

# -----------------------------
# FIGURE 5: EFFECT SIZE PLOT
# -----------------------------

plt.figure()
plt.axvline(0)
plt.scatter(effects, np.arange(len(effects)))
plt.yticks(np.arange(len(effects)), labels)
plt.xlabel("Cohen's d")
plt.title("Effect Sizes Between Model Variants")
output_fig2 = PROJECT_ROOT / "paper/figures/effect_size_plot.png"
plt.savefig(output_fig2, dpi=300)
plt.close()

print(f"Saved {output_fig2}")

# -----------------------------
# PRINT VERIFIED SUMMARY
# -----------------------------

print("\n===== VERIFIED RESULTS =====")
print("Base mean margin:", np.mean(base_margins))
print("DAPT mean margin:", np.mean(dapt_margins))
print("Aligned mean margin:", np.mean(aligned_margins))

print("\nEffect Sizes:")
print("Base vs DAPT:", d_base_dapt)
print("Base vs Aligned:", d_base_aligned)
print("DAPT vs Aligned:", d_dapt_aligned)
