import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_MODEL = "intfloat/e5-base-v2"
DAPT_MODEL = "../models/e5-yoga-neuro-dapt"
ALIGNED_MODEL = "../models/e5-yoga-neuro-aligned"
ALIGNMENT_FILE = "../data/processed/alignment_pairs.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with open(ALIGNMENT_FILE, "r", encoding="utf-8") as f:
    pairs = json.load(f)

anchors = [p["anchor"] for p in pairs]
positives = [p["positive"] for p in pairs]

# --------------------------------------------------
# FUNCTION: Generate 2D Projection
# --------------------------------------------------

def generate_plot(model_path, title, output_name):

    model = SentenceTransformer(model_path, device=device)

    emb_a = model.encode(anchors)
    emb_p = model.encode(positives)

    all_embeddings = np.vstack([emb_a, emb_p])

    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.2,
        metric="cosine",
        random_state=42
    )

    embedding_2d = reducer.fit_transform(all_embeddings)

    n = len(anchors)

    anchor_points = embedding_2d[:n]
    positive_points = embedding_2d[n:]

    plt.figure(figsize=(10, 8))

    # Plot anchors
    plt.scatter(
        anchor_points[:, 0],
        anchor_points[:, 1],
        c="blue",
        label="Yoga Concepts",
        alpha=0.8
    )

    # Plot positives
    plt.scatter(
        positive_points[:, 0],
        positive_points[:, 1],
        c="red",
        label="Neuro Constructs",
        alpha=0.8
    )

    # Draw faint lines between aligned pairs
    for i in range(n):
        plt.plot(
            [anchor_points[i, 0], positive_points[i, 0]],
            [anchor_points[i, 1], positive_points[i, 1]],
            color="gray",
            alpha=0.2
        )

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()

# --------------------------------------------------
# GENERATE PLOTS
# --------------------------------------------------

generate_plot(BASE_MODEL, "Base Model Embedding Geometry", "base_umap.png")
generate_plot(DAPT_MODEL, "DAPT Model Embedding Geometry", "dapt_umap.png")
generate_plot(ALIGNED_MODEL, "Aligned Model Embedding Geometry", "aligned_umap.png")

print("UMAP visualizations saved.")
