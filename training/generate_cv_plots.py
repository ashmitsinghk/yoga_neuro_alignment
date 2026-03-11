import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper", "figures")
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

# -------------------------------------------------
# PLOTTING FUNCTION
# -------------------------------------------------

def plot_cv_results(mode, title_str):
    file_path = os.path.join(RESULTS_DIR, f"cv_results_{mode}.json")
    
    if not os.path.exists(file_path):
        print(f"Error: Results file for {mode} not found at {file_path}. Skipping.")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    # Prepare DataFrame for Seaborn
    true_scores = data["true_scores"]
    random_scores = data["random_scores"]
    
    # Create DataFrame: Condition (Aligned vs Random) and Similarity
    # We want a boxplot showing distribution for Aligned and Random side-by-side
    
    df_true = pd.DataFrame({
        "Similarity": true_scores,
        "Condition": "Aligned"
    })
    
    df_random = pd.DataFrame({
        "Similarity": random_scores,
        "Condition": "Random"
    })
    
    # Combine
    df = pd.concat([df_true, df_random], ignore_index=True)

    # Calculate statistics for annotation
    mean_aligned = data["mean_true"]
    mean_random = data["mean_random"]
    d_val = data["cohens_d"]

    # Plot
    plt.figure(figsize=(7, 6))
    sns.set_theme(style="whitegrid")
    
    # Boxplot
    # palette for Aligned (greenish) and Random (grayish/reddish)
    palette = {"Aligned": "#2ca02c", "Random": "#7f7f7f"}
    
    ax = sns.boxplot(x="Condition", y="Similarity", data=df, palette=palette, width=0.5, showfliers=False)
    
    # Add stripplot to show individual points
    sns.stripplot(x="Condition", y="Similarity", data=df, color=".25", alpha=0.5, jitter=True)

    # Titles and Labels
    plt.title(f"{title_str}\n(Cohen's d = {d_val:.2f})", fontsize=14, pad=15)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.xlabel("") # Remove x-label 'Condition' as categories suffice
    
    # Set Y limits to include 0-1 range comfortably, usually [0, 1] for cosine sim
    plt.ylim(-0.1, 1.1)
    
    # Add horizontal line for 0
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    # Annotate Means
    # Aligned Mean
    plt.text(0, mean_aligned + 0.05, f"μ={mean_aligned:.3f}", 
             horizontalalignment='center', color='darkgreen', fontweight='bold', fontsize=11,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Random Mean
    plt.text(1, mean_random + 0.05, f"μ={mean_random:.3f}", 
             horizontalalignment='center', color='black', fontweight='bold', fontsize=11,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Save
    out_file = os.path.join(FIGURES_DIR, f"figure_{mode}_cv.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Figure saved to: {out_file}")

# -------------------------------------------------
# MAIN
# -------------------------------------------------

if __name__ == "__main__":
    print("Generating figures...")
    
    # Figure A
    plot_cv_results("kfold", "Figure A: 5-Fold Cross-Validation Similarity")
    
    # Figure B
    plot_cv_results("loo", "Figure B: Leave-One-Out Cross-Validation Similarity")

    print("Done.")
