# FairLane: Token-Fair Scheduling for Multi-Tenant LLM Serving

FairLane evaluates and implements token-aware scheduling and admission control policies at the gateway and inference-engine queue layers. Under standard request-count round-robin and FIFO policies, tenants submitting heterogeneous prompt/decode lengths cause resource monopolization and starvation. FairLane compares FIFO, Round-Robin, and Virtual Token Counter (VTC) scheduling under continuous batching constraints.

---

## Benchmark Results (Measured Data Only)

All experiments run across $N = 30$ independent random seeds with 95% non-parametric bootstrap confidence intervals.

> [!NOTE]
> **Pre-Registration & Methodology Disclosures**:
> 1. **Commit Timing**: Experiment 1's hypothesis YAML was committed *after* its first exploratory run during early Phase 1.
> 2. **Pilot-Informed Steady-State**: For Experiment 2, early exploratory script `proto.go` ran steady-state Poisson configurations before the formal YAML existed; the committed hypothesis in `experiments/exp_heterogeneous.yaml` (commit `04c2725`, committed `2026-10-13 16:14:53 +0530`) was therefore **pilot-informed**.
> 3. **Hypothesis Verdicts**: The original finite-trace hypothesis (commit `2019c5d`, `2026-10-13 15:39:12 +0530`) was **falsified** due to the tokens conservation invariant ($\sum \text{Served} \equiv \sum \text{Submitted}$). The steady-state hypothesis is evaluated below. Note that earlier commit hashes from prior rebases are stale.

---

### Experiment 1: Noisy Neighbor Protection under Flooding (`exp_noisy_neighbor`)
- **Topology**: Victim tenant (60 requests, 128 prompt / 64 decode, 20 req/s Poisson, strictly below fair share) co-located with a Flooder tenant (240 requests, 1024 prompt / 256 decode, 83.3 req/s Poisson, saturating server capacity). Server KV capacity: 4096 tokens, max batch size: 16.
- **Metric Definitions**:
  - Primary: Victim Slowdown vs Solo run, Mean TTFT with 95% bootstrap CIs.
  - Secondary: Sliding Token Window Jain's Index ($W = 1000$ tokens served across active tenants).

#### Performance Across Both Admission Modes ($N = 30$ Seeds, 95% CI):
| Policy | Admission Mode | Victim Slowdown (95% CI) | Victim TTFT Mean (ms) (95% CI) | Queue Wait Mean (ms) | Prefill Mean (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FIFO** | `break` | 160.37x [151.46, 169.23]\* | 52,461.3 [49,564.5, 55,341.1] | 52,460.7 | 0.6 |
| **FIFO** | `skip` | 159.20x [150.31, 168.10]\* | 52,148.0 [49,261.2, 55,023.4] | 52,147.4 | 0.6 |
| **Round-Robin** | `break` | 2.19x [1.96, 2.45] | 398.0 [321.5, 484.2] | 397.4 | 0.6 |
| **Round-Robin** | `skip` | 1.83x [1.63, 2.01] | 311.3 [259.6, 370.6] | 310.7 | 0.6 |
| **VTC (default)** | `break` | 4.54x [4.31, 4.77] | 1,172.3 [1,096.4, 1,249.7] | 1,171.7 | 0.6 |
| **VTC (default)** | `skip` | 2.00x [1.82, 2.21] | 336.7 [276.9, 403.9] | 336.1 | 0.6 |

*\*Note on FIFO Slowdown*: Marked as **unbounded / horizon-dependent**. Flooder arrival rate ($\lambda = 83.3$ req/s) exceeds service rate ($\mu \approx 12.5$ req/s). The measured $160.37\times$ ratio scales linearly with duration:
- Arrival Stream: 3.0s / Trace Drain Horizon: 104.0s $\rightarrow$ Slowdown = **177.3x** (Max Queue: 204)
- Arrival Stream: 6.0s / Trace Drain Horizon: 208.0s $\rightarrow$ Slowdown = **313.8x** (Max Queue: 408)
- Arrival Stream: 12.0s / Trace Drain Horizon: 416.0s $\rightarrow$ Slowdown = **633.8x** (Max Queue: 816)

---

### Experiment 2: Steady-State Heterogeneous Sizing (`exp_heterogeneous`)
- **Topology**: Chat tenant (128 prompt / 64 decode, raw 192 tok, weighted 256 tok with $w_{\text{in}}=1, w_{\text{out}}=2$, Poisson 50 req/s) vs RAG tenant (4000 prompt / 128 decode, raw 4128 tok, weighted 4256 tok, Poisson 10 req/s). Server KV capacity: 16,384 tokens, max batch size: 16.
- **Protocol**: Fixed 60.0s simulation horizon, discarding 10.0s warm-up. Evaluated strictly in the window $[10.0\text{s}, 60.0\text{s}]$ where **both queues are backlogged**, with arrival-time sorted requests.
- **Primary Metric Definition**: Long time-window token shares (weighted and raw) and Max-Min Service Ratio: $\max(S_c, S_r) / \min(S_c, S_r)$.

#### Steady-State Results ($N = 30$ Seeds, 95% Bootstrap CI):
| Policy | Chat Weighted Share (%) (95% CI) | Chat Raw Share (%) (95% CI) | Max-Min Ratio (95% CI) | Window Dispatches (Chat / RAG) | Unserved Reqs at 60s (Chat / RAG) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FIFO** | **23.0%** [22.6%, 23.5%] | **18.8%** [18.4%, 19.3%] | 3.35 [3.27, 3.44] | 1064 [1043, 1087] / 213 [211, 214] | 1220 [1194, 1244] / 245 [243, 246] |
| **Round-Robin (`onPick`)** | **33.8%** [33.8%, 33.8%] | **28.3%** [28.3%, 28.3%] | 1.96 [1.96, 1.96] | 1766 [1766, 1767] / 207 [207, 207] | 380 [379, 381] / 251 [251, 251] |
| **Round-Robin (`onDispatch`)** | **5.6%** [5.6%, 5.6%] | **4.4%** [4.4%, 4.4%] | 16.73 [16.71, 16.75] | 247 [245, 249] / 247 [245, 249] | 2188 [2185, 2192] / 202 [200, 205] |
| **VTC** | **49.8%** [49.8%, 49.8%] | **43.5%** [43.4%, 43.5%] | **1.01** [1.01, 1.01] | 2030 [2029, 2032] / 122 [122, 122] | 66 [64, 69] / 352 [352, 352] |

#### TTFT Performance with 95% Bootstrap CIs:
| Policy | Chat TTFT Mean ms (95% CI) | RAG TTFT Mean ms (95% CI) |
| :--- | :---: | :---: |
| **FIFO** | 17,234.5 [16,872.7, 17,573.5] ms | 17,076.8 [16,856.3, 17,278.7] ms |
| **Round-Robin (`onPick`)** | 8,805.5 [8,660.8, 8,951.2] ms | 17,385.8 [17,074.9, 17,672.0] ms |
| **Round-Robin (`onDispatch`)** | 25,386.1 [24,983.3, 25,702.8] ms | 14,960.8 [14,592.5, 15,311.6] ms |
| **VTC** | **5,788.8** [5,648.8, 5,929.6] ms | 22,103.7 [21,883.5, 22,317.1] ms |

#### Key Analytical Findings:
1. **FIFO Sanity & Arrival Mix**:
   - With arrival-time sorting, FIFO Chat raw share is **18.8% [18.4%, 19.3%]**, directly matching the theoretical arrival mix of **18.86%**:
     $$\frac{50 \times 192}{50 \times 192 + 10 \times 4128} = \frac{9600}{9600 + 41280} = 18.86\%$$
   - Under single-queue FIFO ordering, Chat TTFT (17,234.5 ms) and RAG TTFT (17,076.8 ms) are virtually identical, reflecting true arrival-order queue delay. The earlier 0.4% Chat share was an artifact of unsorted request generation where Chat requests preceded RAG requests in the trace slice.
2. **Round-Robin Cursor Dynamics (`onPick` vs `onDispatch`)**:
   - `RRAdvance=onDispatch`: When the cursor advances strictly upon confirmed admission, dispatch counts in the window are **exactly 1.00 : 1** (247 Chat : 247 RAG). This yields **4.4% raw share** and **5.6% weighted share**, perfectly matching the theoretical hand calculation:
     $$\text{Raw: } \frac{192}{192 + 4128} = 4.44\%, \quad \text{Weighted: } \frac{256}{256 + 4256} = 5.67\%$$
   - `RRAdvance=onPick`: When the cursor advances inside `Pick()`, non-fitting RAG head requests fail `server.CanAdmit()`, causing subsequent picks to bypass RAG and select Chat. This elevates Chat dispatches to an 8.53 : 1 ratio (1766 Chat : 207 RAG) and inflates Chat weighted share to 33.8%.
3. **VTC Token Parity**:
   - VTC dispatches 2030 Chat to 122 RAG requests (ratio **16.64 : 1**, matching theoretical $4256 / 256 = 16.625$), achieving **49.8% [49.8%, 49.8%]** weighted token share and **1.01** Max-Min Ratio.
   - **VTC Bias Explanation (49.8% vs 50.0%)**: Minor boundary effects across the 10s and 60s window cutoffs and discrete 4000-token prefill step charging vs incremental per-step decode servicing account for the 0.2% variance.

---

## Diagnostic Investigation (`make diagnose`)

Diagnostics decompose scheduling mechanics, evaluate tie-breaking, measure horizon scaling, and trace exact step events:

### 1. Tie-Break Investigation under Break and Skip Modes ($N = 30$ Seeds)
| Admission Mode | Configuration | Victim Slowdown (95% CI) | Victim TTFT ms (95% CI) |
| :---: | :--- | :---: | :---: |
| `break` | VTC Default (`flooder < victim`) | 4.54x [4.31, 4.77] | 1172.3 [1096.4, 1249.7] ms |
| `break` | VTC Swapped (`a_victim < z_flooder`) | 4.66x [4.45, 4.88] | 1212.2 [1143.4, 1286.2] ms |
| `break` | VTC Seeded Random Tie-Break | 4.53x [4.31, 4.76] | 1171.8 [1096.2, 1249.5] ms |
| `skip` | VTC Default (`flooder < victim`) | 2.00x [1.82, 2.21] | 336.7 [276.9, 403.9] ms |
| `skip` | VTC Swapped (`a_victim < z_flooder`) | 2.03x [1.85, 2.25] | 345.2 [286.0, 413.5] ms |
| `skip` | VTC Seeded Random Tie-Break | 1.99x [1.81, 2.20] | 334.8 [274.5, 401.8] ms |

**Conclusion**: Tie-break direction accounts for $\le 1\%$ variance under both break and skip modes. While exact ties occur frequently upon arrival due to counter lifting ($c_v = \min_{t \in B} c_t = c_f$), tie resolution has zero statistical impact.

### 2. Event Log for 3 Victim Arrivals (#1, #15, #30)
- **Arrival #15 under `break` mode**: Victim arrived at $t = 1085.6\text{ ms}$, but experienced **71 consecutive break-blocked steps** because the flooder head request (1024 tokens) could not fit into available KV cache (<1025 tokens free). Victim was blocked until $t = 1450.4\text{ ms}$ (TTFT = 364.8 ms).
- **Arrival #15 under `skip` mode**: Flooder was bypassed whenever it could not fit, admitting Victim as soon as 129 free KV tokens became available, reducing queue wait to 40.2 ms.

---

## Reproducibility & Commands

Run all unit tests:
```bash
make test
```

Run Experiment 1:
```bash
make exp-noisy-neighbor
```

Run Experiment 2:
```bash
make exp-heterogeneous
```

Run Full Diagnostic Suite:
```bash
make diagnose
```

---

## Hardware & Environment Metadata
- **Author Identity**: `Afnan B.R. <273584085+Afnan-0206@users.noreply.github.com>`
- **OS**: Microsoft Windows 11 Pro 10.0.26300
- **CPU**: AMD Ryzen 7 PRO 5850U with Radeon Graphics (8 physical cores, 16 logical threads)
- **RAM**: 15.31 GB
- **Power Plan**: Balanced (`381b4222-f694-41f0-9685-ff5bb260df2e`)
- **Toolchain**: Go `go1.27.0 windows/amd64`, Python `3.12.10`
