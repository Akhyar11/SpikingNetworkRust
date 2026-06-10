import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300
})

# Path definitions
CONTROLLED_JSON = "experiment/file_model/full_eval_controlled.json"
ABLATION_JSON = "experiment/file_model/ablation_results_distil.json"
OUTPUT_DIR = "/home/akhyar/Dokumen/Peper/SpikingEmbedderRust/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# FIGURE 1: Bar Chart SOPs vs MACs (Log Scale)
# ----------------------------------------------------------------------
def plot_sops_vs_macs():
    print("Generating Figure 1: SOPs vs MACs...")
    with open(CONTROLLED_JSON) as f:
        data = json.load(f)["evaluation"]
    
    # Extract data
    transformer_macs = 10_223_616
    human_sops = data["Human-Only (STS-B)"]["results"]["_energy"]["snn_sops_per_sentence"]
    distil_sops = data["Knowledge Distillation (AI)"]["results"]["_energy"]["snn_sops_per_sentence"]
    simcse_sops = data["Unsupervised SimCSE"]["results"]["_energy"]["snn_sops_per_sentence"]
    
    categories = [
        "SNN (Distil AI)\n[Ours, Best]", 
        "SNN (Human-Only)\n[Ours]", 
        "SNN (SimCSE)\n[Ours]", 
        "MiniLM 6-Layer\n[Transformer]"
    ]
    operations = [distil_sops, human_sops, simcse_sops, transformer_macs]
    colors = ["#1f77b4", "#aec7e8", "#ffbb78", "#d62728"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, operations, color=colors, edgecolor='grey', width=0.6)
    
    # Log scale for readability
    ax.set_yscale('log')
    ax.set_ylabel("Operasi per Kalimat (Log Scale)", fontweight='bold')
    ax.set_title("Kompleksitas Komputasi: SNN (SOPs) vs Transformer (MACs)", fontweight='bold', pad=15)
    
    # Annotate values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    # Add energy ratio note
    ratio = transformer_macs / distil_sops
    plt.figtext(0.15, 0.75, f"Efisiensi Energi:\nDistil AI Butuh\n{ratio:.1f}× Lebih Sedikit\nOperasi Sinaptik!", 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_sops_vs_macs.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_sops_vs_macs.pdf"))
    plt.close()

# ----------------------------------------------------------------------
# FIGURE 2: Heatmap Pearson Lintas Dataset
# ----------------------------------------------------------------------
def plot_pearson_heatmap():
    print("Generating Figure 2: Pearson Correlation Heatmap...")
    with open(CONTROLLED_JSON) as f:
        data = json.load(f)["evaluation"]
        
    datasets = ["STS-B", "STS-12", "STS-13", "STS-14", "STS-15", "STS-16", "SICK-R"]
    models = ["Human-Only (STS-B)", "Knowledge Distillation (AI)", "Unsupervised SimCSE"]
    model_labels = ["Human-Only", "Distil AI (Teacher)", "SimCSE (Unsupervised)"]
    
    # Construct matrix
    matrix = []
    for model in models:
        row = []
        for ds in datasets:
            row.append(data[model]["results"][ds]["pearson"])
        matrix.append(row)
        
    df = pd.DataFrame(matrix, index=model_labels, columns=datasets)
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="Blues", cbar=True, 
                linewidths=.5, annot_kws={"weight": "bold", "size": 10}, ax=ax)
    
    ax.set_title("Korelasi Pearson Lintas Dataset Evaluasi (Zero-Shot)", fontweight='bold', pad=15)
    ax.set_ylabel("Strategi Pelatihan SNN", fontweight='bold')
    ax.set_xlabel("Dataset Evaluasi", fontweight='bold')
    
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_pearson_heatmap.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_pearson_heatmap.pdf"))
    plt.close()

# ----------------------------------------------------------------------
# FIGURE 3: Line Chart Pearson vs Time-Steps (T)
# ----------------------------------------------------------------------
def plot_time_steps():
    print("Generating Figure 3: Time-Steps Analysis...")
    with open(ABLATION_JSON) as f:
        data = json.load(f)["ablation_study"]
        
    # Extracted from json keys
    t_vals = [16, 32, 64]
    pearsons = [
        data["T=16 (with-attention)"]["pearson_correlation"],
        data["T=32 (with-attention)"]["pearson_correlation"],
        data["T=64 (with-attention)"]["pearson_correlation"]
    ]
    
    sops = [
        data["T=16 (with-attention)"]["average_sops"],
        data["T=32 (with-attention)"]["average_sops"],
        data["T=64 (with-attention)"]["average_sops"]
    ]
    
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    
    # Plot Pearson
    color = '#1f77b4'
    ax1.set_xlabel('Panjang Time-Steps (T)', fontweight='bold')
    ax1.set_ylabel('Korelasi Pearson (r)', color=color, fontweight='bold')
    line1 = ax1.plot(t_vals, pearsons, color=color, marker='o', linewidth=2, label='Pearson (r)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(t_vals)
    
    # Plot SOPs on secondary axis
    ax2 = ax1.twinx()  
    color = '#d62728'
    ax2.set_ylabel('SOPs per Kalimat', color=color, fontweight='bold')
    line2 = ax2.plot(t_vals, sops, color=color, marker='s', linestyle='--', linewidth=2, label='SOPs')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Layout and Title
    plt.title("Analisis Sensitivitas Temporal: Performa vs Beban Komputasi SNN", fontweight='bold', pad=15)
    
    # Legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_time_steps_ablation.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_time_steps_ablation.pdf"))
    plt.close()

if __name__ == "__main__":
    plot_sops_vs_macs()
    plot_pearson_heatmap()
    plot_time_steps()
    print("All figures successfully generated in:", OUTPUT_DIR)
