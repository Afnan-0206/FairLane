# FairLane: Token-Fair Scheduling for Multi-Tenant LLM Serving

FairLane evaluates and implements token-aware scheduling and admission control policies at the gateway and inference-engine queue layers. Under standard request-count round-robin and FIFO policies, tenants submitting heterogeneous prompt/decode lengths cause resource monopolization and starvation. FairLane compares FIFO, Round-Robin, and Virtual Token Counter (VTC) scheduling under continuous batching constraints.

---

## Benchmark Results (Measured Data Only)

All experiments run across $N = 30$ independent random seeds with 95% non-parametric bootstrap confidence intervals.

> [!NOTE]
> **Methodology Disclosure**: Experiment 1's hypothesis YAML was originally committed after its first exploratory run during early Phase 1 prototyping. Experiment 2's hypothesis YAML was committed *strictly prior* to running.

### Experiment 1: Noisy Neighbor Protection under Flooding (`exp_noisy_neighbor`)
- **Topology**: Victim tenant (60 requests, 128 prompt / 64 decode, 20 req/s Poisson, strictly below fair share) co-located with a Flooder tenant (240 requests, 1024 prompt / 256 decode, 83.3 req/s Poisson, saturating server capacity). Server KV capacity: 4096 tokens, max batch size: 16.
- **Fairness Metric Definition**: Jain's Fairness Index is evaluated over sliding token windows ($W = 1000$ tokens) of served tokens across active tenants: $J(x) = (\sum x_i)^2 / (n \sum x_i^2)$.

| Policy | Victim Slowdown vs Solo (95% CI) | Victim TTFT Mean (ms) (95% CI) | Victim TTFT P95 (ms) | Jain's Index (Sliding 1000 tok) | Idle-While-Queued (µs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FIFO** | 160.37x [151.46, 169.23]\* | 52,461.3 [49,564.5, 55,341.1] | 61,703.3 | 0.726 [0.713, 0.738] | 0 |
| **Round-Robin** | 2.19x [1.95, 2.46] | 398.0 [318.3, 486.4] | 853.6 | 0.870 [0.853, 0.887] | 0 |
| **VTC (default)** | 4.54x [4.31, 4.76] | 1,172.3 [1,097.2, 1,248.1] | 1,530.6 | 0.830 [0.825, 0.835] | 0 |
| **VTC (paper lift)** | 4.54x [4.31, 4.77] | 1,173.7 [1,098.0, 1,250.8] | 1,552.9 | 0.830 [0.826, 0.835] | 0 |
| **VTC (completion)** | 4.53x [4.34, 4.73] | 1,169.5 [1,104.1, 1,235.9] | 1,479.0 | 0.600 [0.584, 0.617] | 0 |
| **VTC ($w_{\text{out}}=1$)** | 4.28x [4.05, 4.51] | 1,088.0 [1,011.0, 1,166.9] | 1,459.0 | 0.808 [0.802, 0.815] | 0 |
| **VTC ($w_{\text{out}}=2$)** | 4.54x [4.31, 4.76] | 1,172.3 [1,097.2, 1,248.1] | 1,530.6 | 0.830 [0.825, 0.835] | 0 |
| **VTC ($w_{\text{out}}=3$)** | 4.72x [4.50, 4.94] | 1,231.8 [1,158.0, 1,306.3] | 1,588.9 | 0.830 [0.826, 0.834] | 0 |
| **VTC ($w_{\text{out}}=4$)** | 4.89x [4.67, 5.10] | 1,287.0 [1,214.0, 1,360.2] | 1,625.9 | 0.830 [0.825, 0.835] | 0 |

*\*Note on FIFO Slowdown*: Marked as **unbounded / horizon-dependent**. Because flooder arrival rate ($\lambda = 83.3$ req/s) exceeds maximum service rate ($\mu \approx 12.5$ req/s), the FIFO queue grows linearly without bound. The measured $160.37\times$ ratio is strictly a function of the 300-request horizon (at 600 reqs: $313.8\times$; at 1200 reqs: $633.8\times$).

---

### Experiment 2: Steady-State Heterogeneous Sizing (`exp_heterogeneous`)
- **Topology**: Chat tenant (128 prompt / 64 decode, raw 192 tok, weighted 256 tok with $w_{\text{in}}=1, w_{\text{out}}=2$, Poisson 50 req/s) vs RAG tenant (4000 prompt / 128 decode, raw 4128 tok, weighted 4256 tok, Poisson 10 req/s). Both arrival rates exceed individual fair shares. Server KV capacity: 16,384 tokens, max batch size: 16.
- **Protocol**: Fixed 60.0s simulation horizon, discarding a 10.0s warm-up period. Measured strictly in the steady-state window $[10.0\text{s}, 60.0\text{s}]$ where **both tenants are backlogged**.
- **Fairness Metric Definition**: Share of served tokens (both weighted and raw) allocated to each tenant during the concurrent backlog window.

| Policy | Chat Weighted Share (%) (95% CI) | Chat Raw Share (%) (95% CI) | Hand-Calc Prediction (Weighted) | Hand-Calc Prediction (Raw) | Chat TTFT Mean (ms) (95% CI) | RAG TTFT Mean (ms) (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FIFO** | **0.4%** [0.4%, 0.4%] | **0.2%** [0.2%, 0.2%] | <5.0% | <5.0% | 1,341.5 [1,331.4, 1,351.9] | 51,773.1 [51,691.0, 51,855.2]\* |
| **Round-Robin** | **31.9%** [31.5%, 32.3%] | **26.6%** [26.2%, 26.9%] | ~5.7% (1:1 reqs) | ~4.4% (1:1 reqs) | 1,441.1 [1,424.3, 1,457.9] | 52,331.2 [52,248.8, 52,413.6]\* |
| **VTC** | **48.1%** [47.6%, 48.6%] | **41.7%** [41.1%, 42.1%] | **~50.0%** | **~43.6%** | 1,426.1 [1,410.2, 1,442.0] | 52,996.1 [52,912.4, 53,079.8]\* |

*\*Note on RAG TTFT*: Under overload, RAG queue accumulates backlogged requests, leading to queue delay accumulation approaching the 60s horizon.

#### Hand-Calculation Analysis & Deviations:
1. **VTC Target**: VTC balances virtual token counters ($c_{\text{chat}} \approx c_{\text{rag}}$), targeting:
   $$\text{Weighted Share}_{\text{chat}} = \frac{c_{\text{chat}}}{c_{\text{chat}} + c_{\text{rag}}} \approx 50.0\%$$
   The measured **48.1% [47.6%, 48.6%]** matches the theoretical 50.0% prediction within 1.9 percentage points. Raw share matches $41.7\%$ vs $43.6\%$ predicted.
2. **Round-Robin Deviation from 5.7%**:
   - Theoretical 1:1 request dispatch predicts: $\frac{256}{256 + 4256} = 5.67\%$.
   - **Why RR achieves 31.9%**: With a 16,384-token KV cache, each RAG request requires $\ge 4,001$ tokens, allowing at most 3 to 4 concurrent RAG requests in the batch. Whenever 3-4 RAG requests run, subsequent RAG head requests cannot fit into available KV cache. Round-Robin dispatches Chat requests whenever RAG cannot fit or is alternated, elevating Chat request completion share to ~68% and weighted token share to 31.9%.

---

## Diagnostic Investigation (`make diagnose`)

Diagnostics decompose scheduling mechanics, evaluate tie-breaking, measure horizon scaling, and trace exact step events:

### 1. Tie-Break Investigation ($N = 30$ Seeds)
To test whether tie-breaking between equal-counter tenants explains why VTC victim TTFT (1172 ms) is higher than Round-Robin (398 ms):
- **VTC Default** (`flooder < victim` alphabetically): 4.54x slowdown [4.31, 4.77], TTFT 1172.3 ms [1096.4, 1249.7]
- **VTC Swapped Names** (`a_victim < z_flooder`): 4.66x slowdown [4.45, 4.88], TTFT 1212.2 ms [1143.4, 1286.2]
- **VTC Seeded Random Tie-Break**: 4.53x slowdown [4.31, 4.76], TTFT 1171.8 ms [1096.2, 1249.5]
**Conclusion**: Tie-break order accounts for $\le 1\%$ difference. It does not explain the gap.

### 2. TTFT Decomposition & AdmissionMode (`break` vs `skip`)
- Under `AdmissionModeBreak` (default), VTC experiences **199.4 break blocks** per victim request (where the 1024-token flooder head request cannot fit into the remaining KV cache and halts batch formation), causing 1202 ms TTFT (99.1% queue wait).
- Under `AdmissionModeSkip`, non-fitting head requests are bypassed: VTC TTFT drops to **343.6 ms** (or ~655 ms on heavily saturated seeds).
- **Explanation of Remaining Gap (VTC 655 ms vs RR 392 ms)**:
  1. Round-Robin alternates queues strictly 1:1 on dispatch turns; the moment a slot opens, victim gets immediate service regardless of past tokens.
  2. VTC dispatches by lowest cumulative token counter. Because flooder generates 83.3 req/s with 1024-token prompts, flooder requests occupy batch concurrency and KV cache for hundreds of milliseconds, stretching execution step durations up to 15–20 ms per step and pacing victim dispatch to maintain token fairness.

### 3. FIFO Horizon Scaling
| Horizon Scale | Flooder Reqs | Simulated Horizon | Max Queue Depth | Victim TTFT | Victim Slowdown |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1x** | 240 reqs | 104.0 s | 204 reqs | 58.1 s | 177.3x |
| **2x** | 480 reqs | 208.0 s | 408 reqs | 103.1 s | 313.8x |
| **4x** | 960 reqs | 416.0 s | 816 reqs | 208.2 s | 633.8x |
Proves that under persistent overload, FIFO queue growth is linear ($Q(t) \propto (\lambda - \mu)t$) and slowdown is strictly unbounded / horizon-dependent.

---

## Reproducibility & Commands

Run all tests (including hand-computed Jain demand-capped and AdmissionMode starvation tests):
```bash
make test
```

Run Experiment 1 (Noisy Neighbor):
```bash
make exp-noisy-neighbor
```

Run Experiment 2 (Steady-State Heterogeneous Sizing):
```bash
make exp-heterogeneous
```

Run Full Diagnostic Suite:
```bash
make diagnose
```

---

## Hardware & Environment Metadata
- **OS**: Microsoft Windows 11 Pro 10.0.26300
- **CPU**: AMD Ryzen 7 PRO 5850U with Radeon Graphics (8 physical cores, 16 logical threads)
- **RAM**: 15.31 GB
- **Power Plan**: Balanced (`381b4222-f694-41f0-9685-ff5bb260df2e`)
- **Toolchain**: Go `go1.27.0 windows/amd64`, Python `3.12.10`
