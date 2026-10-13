import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    results_dir = os.path.join("experiments", "results")
    summary_path = os.path.join(results_dir, "exp_noisy_neighbor_summary.json")
    raw_path = os.path.join(results_dir, "exp_noisy_neighbor_raw.json")

    if not os.path.exists(summary_path) or not os.path.exists(raw_path):
        print(f"Results not found in {results_dir}")
        return

    with open(summary_path, "r") as f:
        summaries = json.load(f)

    with open(raw_path, "r") as f:
        raw_data = json.load(f)

    df_raw = pd.DataFrame(raw_data)

    # Style configuration
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 16,
    })

    # Plot 1: Victim Slowdown vs Solo Baseline
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    policies = [s["policy_name"] for s in summaries if not s["policy_name"].startswith("vtc-wout")]
    means = [s["victim_slowdown"]["mean"] for s in summaries if not s["policy_name"].startswith("vtc-wout")]
    ci_lows = [s["victim_slowdown"]["ci95_low"] for s in summaries if not s["policy_name"].startswith("vtc-wout")]
    ci_his = [s["victim_slowdown"]["ci95_hi"] for s in summaries if not s["policy_name"].startswith("vtc-wout")]
    
    yerr = [
        [m - l for m, l in zip(means, ci_lows)],
        [h - m for m, h in zip(means, ci_his)]
    ]

    colors = ['#d9534f', '#f0ad4e', '#5cb85c', '#428bca', '#5bc0de']
    bars = ax.bar(policies, means, yerr=yerr, capsize=5, color=colors[:len(policies)], edgecolor='black', alpha=0.85)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, label='Solo Baseline (1.0x)')
    ax.set_ylabel("Victim Slowdown (Colocated / Solo Latency)")
    ax.set_title("FairLane: Victim Slowdown Under Flooding Neighbor (N=30 seeds, 95% CI)")
    ax.set_ylim(bottom=0)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.1, f'{h:.2f}x', ha='center', va='bottom', fontweight='bold')

    ax.legend()
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plot1_path = os.path.join(results_dir, "plot_victim_slowdown.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"Saved {plot1_path}")

    # Plot 2: Jain's Fairness Index & Output Weight Sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Subplot 1: Jain's Index
    j_policies = [s["policy_name"] for s in summaries if not s["policy_name"].startswith("vtc-wout")]
    j_means = [s["jains_index_sliding"]["mean"] for s in summaries if not s["policy_name"].startswith("vtc-wout")]
    j_lows = [s["jains_index_sliding"]["ci95_low"] for s in summaries if not s["policy_name"].startswith("vtc-wout")]
    j_his = [s["jains_index_sliding"]["ci95_hi"] for s in summaries if not s["policy_name"].startswith("vtc-wout")]
    j_yerr = [[m - l for m, l in zip(j_means, j_lows)], [h - m for m, h in zip(j_means, j_his)]]

    ax1.bar(j_policies, j_means, yerr=j_yerr, capsize=4, color='#337ab7', edgecolor='black', alpha=0.8)
    ax1.set_ylabel("Jain's Fairness Index (Sliding Window = 1k tok)")
    ax1.set_title("Resource Equity Across Backlogged Tenants")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(range(len(j_policies)))
    ax1.set_xticklabels(j_policies, rotation=25, ha='right')

    # Subplot 2: w_out weight sweep
    sweep_policies = [s for s in summaries if s["policy_name"].startswith("vtc-wout")]
    w_labels = [s["policy_name"].replace("vtc-wout-", "w_out=") for s in sweep_policies]
    w_slowdowns = [s["victim_slowdown"]["mean"] for s in sweep_policies]
    w_ci_low = [s["victim_slowdown"]["ci95_low"] for s in sweep_policies]
    w_ci_hi = [s["victim_slowdown"]["ci95_hi"] for s in sweep_policies]
    w_yerr = [[m - l for m, l in zip(w_slowdowns, w_ci_low)], [h - m for m, h in zip(w_slowdowns, w_ci_hi)]]

    ax2.plot(w_labels, w_slowdowns, marker='o', linewidth=2.5, markersize=8, color='#d9534f')
    ax2.errorbar(w_labels, w_slowdowns, yerr=w_yerr, fmt='none', capsize=5, color='#d9534f')
    ax2.set_ylabel("Victim Slowdown")
    ax2.set_title("Sensitivity to Output Token Weight (w_in = 1.0)")
    ax2.grid(True)

    plt.tight_layout()
    plot2_path = os.path.join(results_dir, "plot_fairness_and_weight_sweep.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved {plot2_path}")

if __name__ == "__main__":
    main()
