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

    chat_tok_shares = [p["chat_token_share"]["mean"] * 100 for p in policies]
    rag_tok_shares = [p["rag_token_share"]["mean"] * 100 for p in policies]

    chat_req_shares = [p["chat_request_share"]["mean"] * 100 for p in policies]
    rag_req_shares = [p["rag_request_share"]["mean"] * 100 for p in policies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Subplot 1: Token Share (%)
    x = np.arange(len(names))
    width = 0.35

    ax1.bar(x - width/2, chat_tok_shares, width, label='Chat (128 tok)', color='#2980b9', edgecolor='black')
    ax1.bar(x + width/2, rag_tok_shares, width, label='RAG (4k tok)', color='#e67e22', edgecolor='black')
    ax1.axhline(50.0, color='gray', linestyle='--', label='Ideal 50% Token Share')
    ax1.set_ylabel("Share of Total Tokens Served (%)")
    ax1.set_title("Exp 2: Token Share by Policy (Permanent Backlog)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylim(0, 105)
    ax1.legend()

    for i in range(len(names)):
        ax1.text(x[i] - width/2, chat_tok_shares[i] + 1.5, f"{chat_tok_shares[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax1.text(x[i] + width/2, rag_tok_shares[i] + 1.5, f"{rag_tok_shares[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

    # Subplot 2: Request Share (%)
    ax2.bar(x - width/2, chat_req_shares, width, label='Chat Reqs', color='#3498db', edgecolor='black')
    ax2.bar(x + width/2, rag_req_shares, width, label='RAG Reqs', color='#f39c12', edgecolor='black')
    ax2.set_ylabel("Share of Completed Requests (%)")
    ax2.set_title("Exp 2: Completed Request Share by Policy")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylim(0, 105)
    ax2.legend()

    for i in range(len(names)):
        ax2.text(x[i] - width/2, chat_req_shares[i] + 1.5, f"{chat_req_shares[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax2.text(x[i] + width/2, rag_req_shares[i] + 1.5, f"{rag_req_shares[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(results_dir, "plot_heterogeneous_shares.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
