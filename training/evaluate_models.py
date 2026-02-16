import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import random

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_MODEL = "intfloat/e5-base-v2"
DAPT_MODEL = "../models/e5-yoga-neuro-dapt"
ALIGNED_MODEL = "../models/e5-yoga-neuro-aligned"
ALIGNMENT_FILE = "../data/processed/alignment_pairs.json"

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

base_model = SentenceTransformer(BASE_MODEL, device=device)
dapt_model = SentenceTransformer(DAPT_MODEL, device=device)
aligned_model = SentenceTransformer(ALIGNED_MODEL, device=device)

# --------------------------------------------------
# LOAD ALIGNMENT DATA
# --------------------------------------------------

with open(ALIGNMENT_FILE, "r", encoding="utf-8") as f:
    pairs = json.load(f)

anchors = [p["anchor"] for p in pairs]
positives = [p["positive"] for p in pairs]

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

def get_embeddings(model, texts):
    return model.encode(texts, convert_to_tensor=True)

base_anchor_emb = get_embeddings(base_model, anchors)
base_pos_emb = get_embeddings(base_model, positives)

dapt_anchor_emb = get_embeddings(dapt_model, anchors)
dapt_pos_emb = get_embeddings(dapt_model, positives)

aligned_anchor_emb = get_embeddings(aligned_model, anchors)
aligned_pos_emb = get_embeddings(aligned_model, positives)

# --------------------------------------------------
# COSINE SIMILARITY FUNCTION
# --------------------------------------------------

def avg_similarity(anchor_emb, pos_emb):
    sims = util.cos_sim(anchor_emb, pos_emb)
    diagonal = sims.diag()
    return torch.mean(diagonal).item()

# --------------------------------------------------
# RANDOM NEGATIVE SIMILARITY
# --------------------------------------------------

def avg_random_similarity(anchor_emb, pos_emb):
    shuffled = pos_emb[torch.randperm(pos_emb.size(0))]
    sims = util.cos_sim(anchor_emb, shuffled)
    diagonal = sims.diag()
    return torch.mean(diagonal).item()

# --------------------------------------------------
# COMPUTE METRICS
# --------------------------------------------------

base_aligned = avg_similarity(base_anchor_emb, base_pos_emb)
dapt_aligned = avg_similarity(dapt_anchor_emb, dapt_pos_emb)
aligned_aligned = avg_similarity(aligned_anchor_emb, aligned_pos_emb)

base_random = avg_random_similarity(base_anchor_emb, base_pos_emb)
dapt_random = avg_random_similarity(dapt_anchor_emb, dapt_pos_emb)
aligned_random = avg_random_similarity(aligned_anchor_emb, aligned_pos_emb)

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n===== Cross-Domain Alignment Similarity =====")
print(f"Base Model:     {base_aligned:.4f}")
print(f"DAPT Model:     {dapt_aligned:.4f}")
print(f"Aligned Model:  {aligned_aligned:.4f}")

print("\n===== Random Pair Similarity =====")
print(f"Base Model:     {base_random:.4f}")
print(f"DAPT Model:     {dapt_random:.4f}")
print(f"Aligned Model:  {aligned_random:.4f}")

print("\n===== Improvement =====")
print("DAPT Improvement:", dapt_aligned - base_aligned)
print("Alignment Improvement:", aligned_aligned - dapt_aligned)
