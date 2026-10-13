# FairLane Design Decisions & Phase 1 Specification

## 1. Architectural Boundary & Information Hiding
- **Strict Information Hiding (`RequestView`)**: Schedulers in `policy/` observe only tenant ID, prompt token count (`InputTokens()`), arrival virtual timestamp (`ArrivalTime()`), and an optional predicted output length (`PredictedOutputTokens()`). The ground-truth generation length (`ActualOutputTokens()`) is strictly inaccessible to the policy layer.
- **Package Isolation**: `policy/` has zero dependency on `sim/`. This is verified at the AST level by `TestNoSimImport` using Go's `go/parser`.

## 2. Upstream Flow Selection Interface
- **Separation of Concerns**: The scheduling framework (simulator or gateway proxy) owns the multi-tenant FIFO waiting queues. Policies do not own queue storage.
- **Two-Level Selection (`Pick(view)`)**:
  - `SchedulerView` exposes `BackloggedTenants() []string` (lexicographically sorted) and `HeadRequest(tenantID string) (RequestView, bool)`.
  - `Policy.Pick(view)` selects the next winning tenant ID.
  - Schedulers receive lifecycle feedback via callbacks:
    - `OnArrival(req, wasEmpty, lastDeparted, otherBacklogged)`
    - `OnDispatch(req)`
    - `OnTokens(tenantID, inputTokens, outputTokens)` (incremental token step charging)
    - `OnComplete(tenantID, req, totalInput, totalOutput)` (completion charging)
- **VTC Lift Unification**: All starvation-prevention counter lifts are routed through a single function `applyLift(tenantID, otherBacklogged, lastDeparted)`.
  - `LiftTiming`: Switchable between `LiftTimingArrival` (immediate on empty-queue arrival) and `LiftTimingPick` (lazy evaluation during `Pick`).

## 3. VTC Algorithm Comparison: Paper vs Reference Artifact
### Verbatim Paper Algorithm (arXiv:2401.00588, Algorithm 2)
```text
Algorithm 2: VTC Request Selection and Counter Update
Input: Client set U, request queues Q, counters C
1: Upon request r arrives from client u:
2:   if Q_u is empty:
3:     if Q != empty:
4:       c_u <- max(c_u, min_{j in Q} c_j)
5:     else:
6:       let l <- last client left Q
7:       c_u <- max(c_u, c_l)
8:   Enqueue r to Q_u
9: Upon scheduling opportunity:
10:  u* <- argmin_{u in active} c_u
11:  r* <- Dequeue head of Q_u*
12:  c_u* <- c_u* + cost(r*)
```

### Reference Artifact Implementation (`slora/server/router/vtc_req_queue.py`)
Lines 48–51 of `vtc_req_queue.py`:
```python
cnts = [self.client_cnts[k] for k in self.req_queues.keys() if len(self.req_queues[k]) > 0 and k != client_id]
if len(cnts) > 0:
    min_cnt = min(cnts)
    self.client_cnts[client_id] = max(self.client_cnts[client_id], min_cnt)
# Notice: If len(cnts) == 0, NO operation is performed.
```

### Differences & Implementation
- When other tenants have active waiting queues (`len(other) > 0`), both implementations agree: $c_u \leftarrow \max(c_u, \min_{j \in \text{other}} c_j)$.
- When the queue is empty upon arrival:
  - **Paper Mode (`LiftModePaper`)**: Lifts $c_u \leftarrow \max(c_u, c_{\text{last}})$, where $l$ is the last client that left $Q$.
  - **Artifact Mode (`LiftModeArtifact`)**: Does not lift ($c_u$ is untouched).
- FairLane implements `LiftMode` as an explicit switch (`LiftModePaper` vs `LiftModeArtifact`).
- **Default Weights**: $w_{\text{in}} = 1.0$, $w_{\text{out}} = 2.0$ (verified in the reference artifact). A parameter sweep over $w_{\text{out}} \in \{1.0, 2.0, 3.0, 4.0\}$ is evaluated in Experiment 1.

## 4. Determinism & Invariants
- **Lexicographical Tie-Breaking**: All `argmin` decisions tie-break by lowest alphanumeric tenant ID.
- **Sorted Tenant Traversal**: All maps are iterated in sorted key order (`sort.Strings`).
- **Integer Microsecond Virtual Clock**: All time calculations use `int64` microseconds; no floating-point clock accumulation.
- **Leak Prevention Test**: `TestDeterminismAgainstMapIteration` executes 50 iterations with shuffled map insertions and asserts identical decision sequences.

## 5. Continuous Batching Simulator Semantics
- **Break-on-No-Fit**: If the head request of the chosen tenant exceeds currently available KV cache capacity (`inputTokens + 1 > availableKV`), batch formation halts immediately. The scheduler does not bypass or head-of-line skip.
- **Never-Fits Rejection**: Requests requiring more KV tokens than total server capacity (`inputTokens + 1 > kvCapacity`) are rejected at ingress and tracked via `metrics.NeverFitsRejected`.
- **Idle-While-Queued**: Defined as $\sum \Delta t$ where `server.RunningCount() == 0` and `queueManager.TotalQueuedRequests() > 0`. A work-conserving simulator maintains `IdleWhileQueuedUs == 0`.
- **Definition of "Left Q"**: A tenant transitions to "left Q" precisely when its waiting queue length becomes 0 (`len(queues[tenantID]) == 0`). `lastDepartedTenant` records this tenant ID.

## 6. Provenance & Research Clarifications
- **Kubernetes Gateway API Inference Extension (GAIE) PR #2031**:
  - *Status*: Marked as **UNVERIFIED** due to shallow clone limitations in the environment. Flow-control logic in upstream is tracked under `llm-d-router` (`pkg/epp/framework/plugins/flowcontrol`).
- **AIBrix Architecture**:
  - AIBrix operates as a *router-level* multi-engine traffic distributor (routing requests across instances and pods), whereas FairLane and VTC focus on *engine-level / gateway queue-level* continuous batching scheduling.

## 7. Experiment 1 Protocol: Noisy Neighbor Protection
- **Workload Topology**:
  - Victim: 60 requests, $\sim$20 req/s, prompt 128 tokens, output 64 tokens ($192 \text{ tok/req} = 3,840 \text{ tok/s}$, well under fair share).
  - Flooder: 240 requests, $\sim$83 req/s, prompt 1024 tokens, output 256 tokens ($1280 \text{ tok/req} = 106,666 \text{ tok/s}$, heavily saturating capacity).
- **Solo Run Methodology**: Each seed runs a dedicated victim solo execution first to record `VictimSoloLatency`. Slowdown is computed as `ColocatedLatency / SoloLatency`.
- **Sliding Window Jain's Index**: Window size = 1,000 tokens. Evaluated across active backlogged tenants during execution.
- **Pre-committed Hypotheses**:
  1. FIFO suffers catastrophic victim starvation (>100x slowdown).
  2. Round-Robin partially isolates request count but allows token starvation due to heterogeneous token size (>2x slowdown).
  3. VTC prevents token monopolization, bounding slowdown and achieving fair resource distribution.
- **Post-Result Parameter Changes**: None.
