import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    results_dir = os.path.join("experiments", "results")
    summary_path = os.path.join(results_dir, "exp_heterogeneous_summary.json")

    if not os.path.exists(summary_path):
        print(f"Summary not found at {summary_path}")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    policies = data["policies"]
    names = [p["policy_name"] for p in policies]

    # Weighted token share
    chat_weighted = [
        p.get("chat_weighted_token_share", p.get("chat_token_share", {})).get("mean", 0) * 100
        for p in policies
    ]
    chat_weighted_err_lo = [
        (p.get("chat_weighted_token_share", {}).get("mean", 0) - p.get("chat_weighted_token_share", {}).get("ci95_low", 0)) * 100
        for p in policies
    ]
    chat_weighted_err_hi = [
        (p.get("chat_weighted_token_share", {}).get("ci95_hi", 0) - p.get("chat_weighted_token_share", {}).get("mean", 0)) * 100
        for p in policies
    ]
    rag_weighted = [
        p.get("rag_weighted_token_share", p.get("rag_token_share", {})).get("mean", 0) * 100
        for p in policies
    ]

    # Raw token share
    chat_raw = [
        p.get("chat_raw_token_share", p.get("chat_token_share", {})).get("mean", 0) * 100
        for p in policies
    ]
    rag_raw = [
        p.get("rag_raw_token_share", p.get("rag_token_share", {})).get("mean", 0) * 100
        for p in policies
    ]

    # Window Dispatches shares
    chat_disps = [p.get("chat_window_dispatches", {}).get("mean", 0) for p in policies]
    rag_disps = [p.get("rag_window_dispatches", {}).get("mean", 0) for p in policies]
    tot_disps = [c + r if (c + r) > 0 else 1.0 for c, r in zip(chat_disps, rag_disps)]
    chat_req_shares = [c / t * 100 for c, t in zip(chat_disps, tot_disps)]
    rag_req_shares = [r / t * 100 for r, t in zip(rag_disps, tot_disps)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    x = np.arange(len(names))
    width = 0.35

    # Subplot 1: Weighted Token Share (%)
    yerr_chat = [chat_weighted_err_lo, chat_weighted_err_hi]
    ax1.bar(x - width/2, chat_weighted, width, yerr=yerr_chat, capsize=4, label='Chat (128/64, w=256)', color='#2980b9', edgecolor='black')
    ax1.bar(x + width/2, rag_weighted, width, label='RAG (4k/128, w=4256)', color='#e67e22', edgecolor='black')
    ax1.axhline(50.0, color='crimson', linestyle='--', linewidth=1.5, label='Ideal 50% Token Fair Target')
    ax1.set_ylabel("Weighted Token Share (%) [w_in=1, w_out=2]")
    ax1.set_title("Exp 2: Weighted Token Share (Steady-State)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right')

    for i in range(len(names)):
        ax1.text(x[i] - width/2, chat_weighted[i] + 2.5, f"{chat_weighted[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax1.text(x[i] + width/2, rag_weighted[i] + 2.5, f"{rag_weighted[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

    # Subplot 2: Raw Token Share (%)
    ax2.bar(x - width/2, chat_raw, width, label='Chat Raw (192 tok)', color='#3498db', edgecolor='black')
    ax2.bar(x + width/2, rag_raw, width, label='RAG Raw (4128 tok)', color='#d35400', edgecolor='black')
    ax2.set_ylabel("Raw Token Share (%)")
    ax2.set_title("Exp 2: Raw Token Share (Steady-State)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right')

    for i in range(len(names)):
        ax2.text(x[i] - width/2, chat_raw[i] + 2.5, f"{chat_raw[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax2.text(x[i] + width/2, rag_raw[i] + 2.5, f"{rag_raw[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

    # Subplot 3: Completed Request Share (%)
    ax3.bar(x - width/2, chat_req_shares, width, label='Chat Reqs', color='#1abc9c', edgecolor='black')
    ax3.bar(x + width/2, rag_req_shares, width, label='RAG Reqs', color='#9b59b6', edgecolor='black')
    ax3.set_ylabel("Share of Completed Requests (%)")
    ax3.set_title("Exp 2: Completed Request Share")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylim(0, 105)
    ax3.legend(loc='upper right')

    for i in range(len(names)):
        ax3.text(x[i] - width/2, chat_req_shares[i] + 2.5, f"{chat_req_shares[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax3.text(x[i] + width/2, rag_req_shares[i] + 2.5, f"{rag_req_shares[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(results_dir, "plot_heterogeneous_shares.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
