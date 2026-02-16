import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from scipy import stats

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
# LOAD MODELS
# --------------------------------------------------

base_model = SentenceTransformer(BASE_MODEL, device=device)
dapt_model = SentenceTransformer(DAPT_MODEL, device=device)
aligned_model = SentenceTransformer(ALIGNED_MODEL, device=device)

# --------------------------------------------------
# LOAD DATA
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

base_a = get_embeddings(base_model, anchors)
base_p = get_embeddings(base_model, positives)

dapt_a = get_embeddings(dapt_model, anchors)
dapt_p = get_embeddings(dapt_model, positives)

aligned_a = get_embeddings(aligned_model, anchors)
aligned_p = get_embeddings(aligned_model, positives)

# --------------------------------------------------
# PER-PAIR SIMILARITIES
# --------------------------------------------------

def pairwise_diag(a, b):
    sims = util.cos_sim(a, b)
    return sims.diag().cpu().numpy()

base_sim = pairwise_diag(base_a, base_p)
dapt_sim = pairwise_diag(dapt_a, dapt_p)
aligned_sim = pairwise_diag(aligned_a, aligned_p)

# --------------------------------------------------
# RANDOM MARGINS
# --------------------------------------------------

np.random.seed(42)

def random_sim(model):
    shuffled = positives.copy()
    np.random.shuffle(shuffled)
    emb_a = model.encode(anchors, convert_to_tensor=True)
    emb_p = model.encode(shuffled, convert_to_tensor=True)
    return pairwise_diag(emb_a, emb_p)

base_rand = random_sim(base_model)
dapt_rand = random_sim(dapt_model)
aligned_rand = random_sim(aligned_model)

base_margin = base_sim - base_rand
dapt_margin = dapt_sim - dapt_rand
aligned_margin = aligned_sim - aligned_rand

# --------------------------------------------------
# PAIRED T-TESTS
# --------------------------------------------------

def paired_test(x, y):
    t_stat, p_val = stats.ttest_rel(x, y)
    return t_stat, p_val

def cohens_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

print("\n===== Paired T-Tests (Similarity) =====")
print("Base vs DAPT:", paired_test(base_sim, dapt_sim))
print("DAPT vs Aligned:", paired_test(dapt_sim, aligned_sim))
print("Base vs Aligned:", paired_test(base_sim, aligned_sim))

print("\n===== Paired T-Tests (Margin) =====")
print("Base vs DAPT:", paired_test(base_margin, dapt_margin))
print("DAPT vs Aligned:", paired_test(dapt_margin, aligned_margin))
print("Base vs Aligned:", paired_test(base_margin, aligned_margin))

print("\n===== Effect Sizes (Cohen's d on Margin) =====")
print("Base vs DAPT:", cohens_d(base_margin, dapt_margin))
print("DAPT vs Aligned:", cohens_d(dapt_margin, aligned_margin))
print("Base vs Aligned:", cohens_d(base_margin, aligned_margin))

print("\n===== Mean Margins =====")
print("Base:", np.mean(base_margin))
print("DAPT:", np.mean(dapt_margin))
print("Aligned:", np.mean(aligned_margin))

results = {
    "mean_margin": {
        "base": float(np.mean(base_margin)),
        "dapt": float(np.mean(dapt_margin)),
        "aligned": float(np.mean(aligned_margin))
    },
    "t_tests_margin": {
        "base_vs_dapt": [float(x) for x in paired_test(base_margin, dapt_margin)],
        "dapt_vs_aligned": [float(x) for x in paired_test(dapt_margin, aligned_margin)],
        "base_vs_aligned": [float(x) for x in paired_test(base_margin, aligned_margin)]
    },
    "effect_sizes_margin": {
        "base_vs_dapt": float(cohens_d(base_margin, dapt_margin)),
        "dapt_vs_aligned": float(cohens_d(dapt_margin, aligned_margin)),
        "base_vs_aligned": float(cohens_d(base_margin, aligned_margin))
    }
}

with open("../experiments/results/statistical_tests.json", "w") as f:
    json.dump(results, f, indent=2)

print("Statistical results saved.")