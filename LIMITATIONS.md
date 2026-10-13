# FairLane Limitations & Known Constraints

## 1. Methodology Disclosures & Pre-Registration

### Pilot-Informed Steady-State & Commit History
- **Experiment 1 Hypothesis Timing**: The hypothesis YAML (`experiments/exp_noisy_neighbor.yaml`) was originally committed *after* its first exploratory baseline execution during Phase 1 development.
- **Experiment 2 Hypothesis Timing & `proto.go`**: Early exploratory prototyping script `proto.go` ran steady-state Poisson configurations before the formal YAML existed. Consequently, the steady-state hypothesis in `experiments/exp_heterogeneous.yaml` was **pilot-informed**.
- **Committed Steady-State Hypothesis Quote** (`git show 04c2725:experiments/exp_heterogeneous.yaml`, committed `2026-10-13 16:14:53 +0530`):
  ```yaml
  # Pre-committed Hypotheses:
  # 1. Round-Robin Hand-Calc Prediction: Under strict 1:1 request dispatch (N_chat = N_rag),
  #    Chat weighted token share is 256 / (256 + 4256) = 256 / 4512 ~ 5.67% (~5.7%).
  #    Raw token share is 192 / (192 + 4128) = 192 / 4320 ~ 4.44%.
  #    Under continuous batching, if KV capacity restricts concurrent RAG admissions, Chat may capture
  #    additional batch slots, elevating empirical share above 5.7%.
  # 2. VTC Hand-Calc Prediction: By balancing virtual token counters (c_chat ~ c_rag),
  #    VTC will schedule ~16.6 Chat requests for every 1 RAG request, targeting:
  #    Chat Weighted Token Share ~ 50.0% (+/- 5%) and RAG Weighted Token Share ~ 50.0% (+/- 5%).
  #    Chat Raw Token Share ~ (16.625 * 192) / (16.625 * 192 + 4128) = 3192 / 7320 ~ 43.6%.
  # 3. FIFO Prediction: Earliest arrivals monopolize continuous batch slots, leaving Chat with <5% token share.
  ```
- **Hypothesis Verdicts**:
  - The original finite-trace hypothesis (commit `2019c5d`, `2026-10-13 15:39:12 +0530`) was **falsified** due to the tokens conservation invariant ($\sum \text{Served} \equiv \sum \text{Submitted}$).
  - The pilot-informed steady-state hypothesis is **validated** under corrected arrival sorting: VTC achieved **49.8% [49.8%, 49.8%]** weighted share and **43.5% [43.4%, 43.5%]** raw share.
- *Note on Git Hashes*: Old commit hashes mentioned in early conversation records are stale due to history re-basing for author identity compliance.

---

## 2. Experiment 2 Sanity & Workload Trace Dynamics

### Resolution of the Trace-Sorting Artifact
In early Phase 1c test runs, FIFO was reported with an anomalous 0.4% Chat token share, 1.3s Chat TTFT, and 52s RAG TTFT:
- **Root Cause**: `cmd/exp_heterogeneous/main.go` generated requests sequentially per tenant (all 2500 Chat requests followed by all 500 RAG requests) without sorting by arrival timestamp. Because `traceIdx` advanced linearly, RAG requests did not enter the queue until ~50s into the run.
- **Corrected Behavior with Arrival-Time Sorting**:
  - FIFO Chat raw share is **18.8% [18.4%, 19.3%]**, matching the theoretical arrival mix of **18.86%**:
    $$\frac{50 \times 192}{50 \times 192 + 10 \times 4128} = \frac{9600}{9600 + 41280} = 18.86\%$$
  - Single-queue FIFO ordering grants both tenants equal queue priority: Chat TTFT is **17,234.5 ms** and RAG TTFT is **17,076.8 ms** (virtually identical).
  - Unserved requests at 60s horizon: FIFO leaves **1220 Chat** and **245 RAG** requests backlogged in queue.

### Round-Robin Cursor Advance (`onPick` vs `onDispatch`)
Early explanations attributed Round-Robin's 31.9% Chat share to "KV capacity limits alone". Systematic testing of cursor mechanics reveals the exact driver:
1. **`RRAdvance=onDispatch`**: When the cursor advances strictly upon confirmed admission, dispatch counts in the window are **exactly 1.00 : 1** (247 Chat : 247 RAG). This yields **4.4% raw share** and **5.6% weighted share**, confirming the theoretical hand calculation ($4.44\%$ raw, $5.67\%$ weighted). Max-Min Ratio is **16.73**.
2. **`RRAdvance=onPick`**: When the cursor advances inside `Pick()`, non-fitting RAG head requests fail `server.CanAdmit()`, causing subsequent picks to bypass RAG and select Chat. This elevates Chat dispatches to an 8.53 : 1 ratio (1766 Chat : 207 RAG) and inflates Chat weighted share to 33.8%.

### VTC Bias Explanation (49.8% vs 50.0%)
VTC achieves 49.8% weighted share rather than exactly 50.0% due to:
- **Measurement Window Boundaries**: The steady-state window begins at $t=10\text{s}$ and ends at $t=60\text{s}$. Requests admitted prior to 10s or executing across 60s contribute partial tokens to window accounting.
- **Charge vs Service Granularity**: A 4000-token RAG prefill charges 4000 tokens in step 1, whereas decode tokens charge 1 token per step over 128 steps, introducing discrete step boundary variance.

---

## 3. Fairness Metrics: Definitions and Guidelines

### Primary vs Secondary Metrics
- **Primary Metrics**: Long Time-Window Token Shares (Weighted & Raw) and Max-Min Service Ratio: $\max(S_c, S_r) / \min(S_c, S_r)$ evaluated over the entire steady-state window.
- **Secondary Metrics**: Sliding Token Window Jain's Index ($W = 1000$ tokens) and Time-Windowed Demand-Capped Jain Index.
- **Clarity Rule**: Exactly one fairness definition is reported per benchmark table to avoid ambiguity.

---

## 4. Horizon-Dependence of FIFO Under Overload
In Experiment 1, Flooder arrival rate ($\lambda = 83.3$ req/s) exceeds maximum service rate ($\mu \approx 12.5$ req/s).
- **Linear Slowdown Explosion**: The reported $160.37\times$ slowdown is directly tied to the simulation horizon:
  - Arrival Stream: 3.0s / Trace Drain Horizon: 104.0s $\rightarrow$ Slowdown = **177.3x**, Queue depth = 204
  - Arrival Stream: 6.0s / Trace Drain Horizon: 208.0s $\rightarrow$ Slowdown = **313.8x**, Queue depth = 408
  - Arrival Stream: 12.0s / Trace Drain Horizon: 416.0s $\rightarrow$ Slowdown = **633.8x**, Queue depth = 816
- FIFO slowdown is formally marked as **unbounded / horizon-dependent**.

---

## 5. Admission Modes & Tie-Breaking Mechanics

### `AdmissionModeBreak` vs `AdmissionModeSkip`
- **`AdmissionModeBreak` (Default)**: Preserves strict order and prevents starvation of large requests. However, when large prompts fail KV admission, batch formation halts immediately (causing ~199 break blocks per victim request in Exp 1, elevating VTC TTFT to 1172 ms).
- **`AdmissionModeSkip`**: Bypasses non-fitting head requests, reducing VTC victim TTFT across 30 seeds to 336.7 ms [276.9, 403.9] ms (vs RR's 311.3 ms [259.6, 370.6] ms).
- **Starvation Risk**: Continuous small requests indefinitely starve large requests under Skip mode (`TestLargeRequestStarvationUnderSkip`).

### Exact Lift Ties
Event logs demonstrate that exact counter ties occur regularly when a tenant transitions from empty to backlogged, as VTC lifts the tenant counter to the minimum backlogged counter ($c_v = \min_{t \in B} c_t = c_f$). However, seeded random tie-break tests over 30 seeds prove that tie resolution order has zero statistical impact under both Break mode (4.54x default vs 4.53x random) and Skip mode (2.00x default vs 1.99x random).
