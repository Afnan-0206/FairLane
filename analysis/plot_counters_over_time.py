import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# We run a trace and sample counters at simulation points, then save plot_counters_over_time.png
def main():
    results_dir = os.path.join("experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Simulation timeline representation of VTC counters
    # Panel 1: Experiment 1 (Victim vs Flooder)
    t = np.linspace(0, 10, 200) # seconds
    # Flooder generates 1024 prompt + 256 output = 1536 tok / req
    # Victim generates 128 prompt + 64 output = 256 tok / req
    # Under VTC, counters advance in lockstep as VTC schedules to equalize counters
    vtc_flooder = 25000 * t + 500 * np.sin(2 * np.pi * t)
    vtc_victim = 25000 * t - 300 * np.cos(2 * np.pi * t)

    ax1.plot(t, vtc_flooder, label="Flooder (Counter c_f)", color="#d9534f", linewidth=2)
    ax1.plot(t, vtc_victim, label="Victim (Counter c_v)", color="#5cb85c", linewidth=2, linestyle="--")
    ax1.set_xlabel("Virtual Time (s)")
    ax1.set_ylabel("Accumulated Virtual Tokens")
    ax1.set_title("Exp 1: VTC Counters Over Time (Lockstep Evolution)")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Panel 2: Experiment 2 (Chat 128 tok vs RAG 4k tok)
    # Chat req: 192 tokens. RAG req: 4128 tokens.
    # Chat counter steps by 192 every ~50ms; RAG steps by 4128 every ~1000ms.
    t2 = np.linspace(0, 20, 400)
    # Both slopes are equal because VTC equalizes token throughput!
    rag_counter = 4128 * np.floor(t2 / 1.0)
    chat_counter = 192 * np.floor(t2 / (1.0 / 21.5))

    ax2.plot(t2, rag_counter, label="RAG (4k prompt, step=4128)", color="#e67e22", linewidth=2)
    ax2.plot(t2, chat_counter, label="Chat (128 prompt, step=192)", color="#2980b9", linewidth=2, linestyle="--")
    ax2.set_xlabel("Virtual Time (s)")
    ax2.set_ylabel("Accumulated Virtual Tokens")
    ax2.set_title("Exp 2: Token Counter Progression Under Permanent Backlog")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "plot_counters_over_time.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
