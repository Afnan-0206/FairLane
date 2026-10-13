# FairLane: Token-Fair Scheduling for Multi-Tenant LLM Serving

FairLane evaluates and implements token-aware scheduling and admission control policies at the gateway and inference-engine queue layers. Under standard request-count round-robin and FIFO policies, tenants submitting heterogeneous prompt/decode lengths cause resource monopolization and starvation. FairLane compares FIFO, Round-Robin, and Virtual Token Counter (VTC) scheduling under continuous batching constraints.

---

## Benchmark Results (Measured Data Only)

All experiments run across $N = 30$ independent random seeds with 95% non-parametric bootstrap confidence intervals.

### Experiment 1: Noisy Neighbor Protection under Flooding (`exp_noisy_neighbor`)
- **Topology**: Victim tenant (60 requests, 128 prompt / 64 decode, 20 req/s, strictly below fair share) co-located with a Flooder tenant (240 requests, 1024 prompt / 256 decode, 83 req/s, saturating server capacity). Server KV capacity: 4096 tokens, max batch size: 16.

| Policy | Victim Slowdown vs Solo | Victim TTFT Mean (ms) | Victim TTFT P95 (ms) | Jain's Fairness Index | Idle-While-Queued (µs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FIFO** | 160.37x [151.46, 169.23] | 52,461.3 [49,564.5, 55,341.1] | 61,703.3 | 0.726 [0.713, 0.738] | 0 |
| **Round-Robin** | 2.19x [1.95, 2.46] | 398.0 [318.3, 486.4] | 853.6 | 0.870 [0.853, 0.887] | 0 |
| **VTC (default)** | 4.54x [4.31, 4.76] | 1,172.3 [1,097.2, 1,248.1] | 1,530.6 | 0.830 [0.825, 0.835] | 0 |
| **VTC (paper lift)** | 4.54x [4.31, 4.77] | 1,173.7 [1,098.0, 1,250.8] | 1,552.9 | 0.830 [0.826, 0.835] | 0 |
| **VTC (completion)** | 4.53x [4.34, 4.73] | 1,169.5 [1,104.1, 1,235.9] | 1,479.0 | 0.600 [0.584, 0.617] | 0 |
| **VTC ($w_{\text{out}}=1$)** | 4.28x [4.05, 4.51] | 1,088.0 [1,011.0, 1,166.9] | 1,459.0 | 0.808 [0.802, 0.815] | 0 |
| **VTC ($w_{\text{out}}=2$)** | 4.54x [4.31, 4.76] | 1,172.3 [1,097.2, 1,248.1] | 1,530.6 | 0.830 [0.825, 0.835] | 0 |
| **VTC ($w_{\text{out}}=3$)** | 4.72x [4.50, 4.94] | 1,231.8 [1,158.0, 1,306.3] | 1,588.9 | 0.830 [0.826, 0.834] | 0 |
| **VTC ($w_{\text{out}}=4$)** | 4.89x [4.67, 5.10] | 1,287.0 [1,214.0, 1,360.2] | 1,625.9 | 0.830 [0.825, 0.835] | 0 |

### Experiment 2: Heterogeneous Request Sizing (`exp_heterogeneous`)
- **Topology**: Chat tenant (200 requests, 128 prompt / 64 decode = 192 tok/req) vs RAG tenant (50 requests, 4000 prompt / 128 decode = 4128 tok/req) under permanent backlog. Server KV capacity: 16,384 tokens, max batch size: 16.

| Policy | Chat Token Share (%) | RAG Token Share (%) | Chat TTFT Mean (ms) | Chat TTFT P95 (ms) | RAG TTFT Mean (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FIFO** | 15.7% [15.7%, 15.7%] | 84.3% [84.3%, 84.3%] | 5,249.3 [5,249.3, 5,249.3] | 10,594.6 | 2,752.1 |
| **Round-Robin** | 15.7% [15.7%, 15.7%] | 84.3% [84.3%, 84.3%] | 2,628.5 [2,628.5, 2,628.5] | 5,074.0 | 4,285.2 |
| **VTC** | 15.7% [15.7%, 15.7%] | 84.3% [84.3%, 84.3%] | 2,209.6 [2,209.6, 2,209.6] | 4,409.5 | 4,374.0 |

*Note on Lifetime Token Shares*: Because all submitted requests run to completion (tokens conservation invariant), lifetime token shares match submitted token ratios (38,400 chat tokens / 244,800 total tokens = 15.7%). During concurrent execution, VTC prioritizes Chat requests, reducing Chat TTFT by 16.0% vs Round-Robin and 57.9% vs FIFO.

---

## Reproducibility & Commands

Run all tests:
```bash
make test
```

Run Experiment 1 (Noisy Neighbor):
```bash
make exp-noisy-neighbor
```

Run Experiment 2 (Heterogeneous Sizing):
```bash
make exp-heterogeneous
```

---

## Hardware & Environment Metadata
- **OS**: Microsoft Windows 11 Pro 10.0.26300
- **CPU**: AMD Ryzen 7 PRO 5850U with Radeon Graphics (8 physical cores, 16 logical threads)
- **RAM**: 15.31 GB
- **Power Plan**: Balanced (`381b4222-f694-41f0-9685-ff5bb260df2e`)
- **Toolchain**: Go `go1.27.0 windows/amd64`, Python `3.12.10`
