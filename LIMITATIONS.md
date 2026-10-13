# FairLane Limitations & Known Constraints

## 1. Methodology Disclosures & Hypothesis Evaluation

### Experiment 1 Hypothesis Commit Timing
- **Honest Disclosure**: Experiment 1's hypothesis YAML (`experiments/exp_noisy_neighbor.yaml`) was originally committed *after* its first exploratory baseline execution during Phase 1 development. 
- All Phase 1b and Phase 1c experiments, diagnostic investigations, and Experiment 2 configurations were strictly committed *prior* to execution.

### Experiment 2 Original Hypothesis & Verdict
- **Original Hypothesis (quoted from `experiments/exp_heterogeneous.yaml` commit `2019c5d`)**:
  > *"Under permanent backlog with heterogeneous request sizes (Chat 128 tok vs RAG 4k tok), VTC will equalize token share to ~50% (+/- 5%) for both tenants across the trace."*
- **Verdict**:
  - **FAILED on Drained Finite Traces**: Under finite traces run to full completion, the *conservation of tokens invariant* dictates that total served tokens must equal total submitted tokens ($\sum \text{Served}_i \equiv \sum \text{Submitted}_i$). Because Chat submitted 38,400 tokens and RAG submitted 206,400 tokens, lifetime token shares are mathematically constrained to **15.7% Chat** and **84.3% RAG** for *any* work-conserving policy.
  - **PASSED under Steady-State Overload Window**: When re-evaluated under continuous Poisson arrivals exceeding fair share with a fixed horizon (60s) and discarding warm-up (10s), measured strictly in the window where both queues are backlogged:
    - VTC Chat Weighted Token Share = **48.1% [47.6%, 48.6%]** (predicted: 50.0%).
    - VTC Chat Raw Token Share = **41.7% [41.1%, 42.1%]** (predicted: 43.6%).

---

## 2. Jain's Fairness Index: Definitions and Anomalies

### Sliding Token Window vs Time-Windowed Demand-Capped Definitions
FairLane evaluates two distinct formulations of Jain's Index:
1. **Un-capped Sliding Token Window ($W = 1000$ tokens)**:
   $$J(x) = \frac{(\sum_{i=1}^n x_i)^2}{n \sum_{i=1}^n x_i^2}$$
   Evaluated over contiguous intervals of 1,000 served tokens across active tenants.
2. **Time-Windowed Demand-Capped Jain Index**:
   For total served capacity $C$ in time window $\Delta t$, nominal fair share is $C/n$. Each tenant's target is capped at its active demand:
   $$T_i = \min(\text{Demand}_i, C/n)$$
   Normalized service ratio is $y_i = \text{Served}_i / T_i$, and Jain's index is computed over $y$:
   $$J(y) = \frac{(\sum y_i)^2}{n \sum y_i^2}$$
   Verified with exact hand-computed test `TestJainsDemandCappedHandComputed` in `sim/sim_test.go`.

### Why Exp 1 Jain Values are Unchanged from Phase 1
In Experiment 1, Jain's index is computed using the un-capped sliding token window ($W = 1000$ tokens). Because the Flooder tenant submits massive 1024-token prompts at 83.3 req/s while the Victim submits 128-token prompts at 20 req/s, the relative service tokens in any 1000-token window remain identical across Phase 1 and Phase 1c runs. Uncapped sliding-window Jain measures raw token count parity rather than demand-capped satisfaction, yielding 0.870 for Round-Robin and 0.830 for VTC.

### Why Exp 2 Un-capped Sliding Window Ranks VTC Worst
In Experiment 2, RAG requests have massive 4,000-token prefills, whereas Chat requests have 128-token prefills:
- When a single RAG request prefills, it consumes 4,000 tokens across multiple consecutive steps, completely monopolizing 4 contiguous 1,000-token sliding windows ($x_{\text{rag}} = 1000, x_{\text{chat}} = 0$). This produces an instantaneous Jain index of $1/2 = 0.50$.
- To compensate and achieve token fairness, VTC subsequently dispatches a burst of ~16 Chat requests ($16 \times 256$ weighted tokens), which monopolizes subsequent windows ($x_{\text{chat}} = 1000, x_{\text{rag}} = 0$).
- This bursty phase alternation creates extreme variance across narrow sliding token windows. Consequently, the un-capped sliding token window ranks VTC lower than Round-Robin, despite VTC achieving nearly perfect macroscopic token fairness (48.1% vs 51.9%) over the entire backlogged duration.

---

## 3. Horizon-Dependence of FIFO Under Overload
In Experiment 1, Flooder arrival rate ($\lambda = 83.3$ req/s) exceeds maximum continuous batch service capacity ($\mu \approx 12.5$ req/s).
- **Unbounded Queue**: The FIFO queue grows without bound:
  $$Q(t) \propto (\lambda - \mu)t$$
- **Linear Slowdown Explosion**: The reported $160.37\times$ slowdown is directly tied to the 300-request ($104.0\text{ s}$) simulation horizon. As demonstrated in `make diagnose`:
  - Horizon 1x (240 flooder reqs, 104 s): Slowdown = **177.3x**, Queue depth = 204
  - Horizon 2x (480 flooder reqs, 208 s): Slowdown = **313.8x**, Queue depth = 408
  - Horizon 4x (960 flooder reqs, 416 s): Slowdown = **633.8x**, Queue depth = 816
- In all tables, FIFO slowdown is formally marked as **unbounded / horizon-dependent**.

---

## 4. Admission Control Modes: `break` vs `skip`

FairLane supports configurable admission modes:
```go
type AdmissionMode int
const (
    AdmissionModeBreak AdmissionMode = iota // Default
    AdmissionModeSkip
)
```

### Tradeoffs:
1. **`AdmissionModeBreak` (Default)**:
   - Preserves strict order and prevents starvation of large requests.
   - **Vulnerability**: If the policy selects a tenant with a large prompt (e.g. Flooder 1024 tokens) and available KV cache is <1025 tokens, batch formation halts immediately. In Exp 1, this caused 199.4 break blocks per victim request under VTC, inflating victim TTFT to 1172 ms.
2. **`AdmissionModeSkip`**:
   - If a selected tenant's head request cannot fit, it is bypassed to check other backlogged tenants.
   - Drops VTC victim TTFT from 1202 ms to 344–655 ms.
   - **Critical Vulnerability (Starvation)**: Under continuous small-request traffic (e.g. 100-token Chat requests), large requests (e.g. 4000-token RAG requests) can be indefinitely starved if KV cache headroom never recovers to 4001 tokens. Verified with unit test `TestLargeRequestStarvationUnderSkip`.

---

## 5. Simulator-to-Hardware Fidelity Gap
- **Step Latency Modeling**: Uses a linear roofline approximation ($T_{\text{step}} = T_{\text{base}} + T_{\text{tok}} \times N_{\text{tokens}}$). It does not capture Tensor Core saturation transitions, CUDA graph launch overheads, or memory bandwidth inflection points.
- **KV Cache Fragmentation**: Continuous token model without PagedAttention block-table allocation overheads.
- **Prefix Caching**: Does not model radix tree prefix caching.

---

## 6. Upstream Reference Status
- **Kubernetes GAIE PR #2031**: Flow-control narrative remains marked as **UNVERIFIED** due to shallow clone constraints. Production flow-control interfaces are referenced against `llm-d-router` (`pkg/epp/framework/plugins/flowcontrol`).
- **AIBrix Architecture**: Operates as a distributed router-level load balancer across pods/nodes, not an intra-instance queue-level continuous batching scheduler.
