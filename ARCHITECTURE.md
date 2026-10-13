# FairLane Architecture Specification

## 1. Scheduling Hierarchy

FairLane implements a two-level scheduling architecture modeled on upstream production gateways (`llm-d-router`):

```text
Incoming Requests (Tenants A, B, C...)
              │
              ▼
   ┌──────────────────────┐
   │ TenantQueueManager   │   <-- Framework owns multi-tenant FIFO queues
   └──────────┬───────────┘
              │ Exposes SchedulerView (Sorted Backlogged Tenants + Head Requests)
              ▼
   ┌──────────────────────┐
   │    Policy.Pick()     │   <-- Tier 2 Inter-Flow Selection (FIFO, RR, VTC)
   └──────────┬───────────┘
              │ Selected Tenant ID
              ▼
   ┌──────────────────────┐
   │Continuous Batch Server│   <-- Tier 1 Execution & KV Cache Admission
   └──────────────────────┘
```

1. **Tier 2 (Flow Selection)**: The policy inspects backlogged tenant queues and decides which tenant is granted the next dispatch opportunity.
2. **Tier 1 (Continuous Batching & Memory Admission)**: The server verifies whether the selected tenant's head request can fit into the active batch and remaining KV cache capacity.

---

## 2. Interface Contracts & Information Hiding

### `RequestView` (`policy/policy.go`)
Schedulers can only observe metadata available at admission time:
```go
type RequestView interface {
    ID() string
    TenantID() string
    InputTokens() int
    PredictedOutputTokens() int // 0 if absent
    ArrivalTime() int64         // Microsecond virtual timestamp
}
```
*Invariant*: Ground-truth output length (`ActualOutputTokens()`) is strictly prohibited and absent from this interface.

### `SchedulerView` (`policy/policy.go`)
The framework retains ownership of the queues and passes a read-only view to the policy:
```go
type SchedulerView interface {
    BackloggedTenants() []string // INVARIANT: Lexicographically sorted
    HeadRequest(tenantID string) (RequestView, bool)
    CurrentTime() int64
}
```

### `Policy` Lifecycle Callbacks (`policy/policy.go`)
```go
type Policy interface {
    Name() string
    Pick(view SchedulerView) (tenantID string)
    OnArrival(req RequestView, wasEmpty bool, lastDepartedTenant string, otherBackloggedTenants []string)
    OnDispatch(req RequestView)
    OnTokens(tenantID string, inputTokens, outputTokens int)
    OnComplete(tenantID string, req RequestView, totalInput, totalOutput int)
}
```

---

## 3. Policy Implementations

### FIFO (`policy/fifo.go`)
Selects the tenant whose head request has the earliest arrival timestamp:
$$t^* = \arg\min_{t \in \text{backlogged}} \text{arrival}(t)$$
Ties are broken deterministically by lexicographically smallest tenant ID.

### Round-Robin (`policy/rr.go`)
Maintains a cyclic cursor across the lexicographically sorted list of backlogged tenants. Grants exactly one dispatch opportunity per tenant turn.

### Virtual Token Counter (`policy/vtc.go`)
Tracks accumulated virtual tokens per tenant:
$$t^* = \arg\min_{t \in \text{backlogged}} c_t$$
- **Charging**: Incremental prefill cost $w_{\text{in}} \times \text{input}$ charged at `OnDispatch`; decode step cost $w_{\text{out}} \times 1$ charged at `OnTokens`.
- **Unified Lift (`applyLift`)**: When a tenant with counter $c_u$ rejoins while others are backlogged:
  $$c_u \leftarrow \max(c_u, \min_{j \in \text{active}} c_j)$$
- **Empty-Queue Rejoin Modes**:
  - `LiftModePaper`: $c_u \leftarrow \max(c_u, c_l)$ where $l$ is the last client that departed an empty queue.
  - `LiftModeArtifact`: No-op; counter retained.
- **Lift Timing**: Evaluated immediately at `OnArrival` (`LiftTimingArrival`) or lazily during `Pick` (`LiftTimingPick`).

---

## 4. Discrete-Event Simulator Semantics (`sim/`)

- **Break-On-No-Fit**: When the server checks the head request of the winning tenant, if `usedKV + inputTokens + 1 > KVCapacity`, batch formation breaks immediately. The scheduler does not skip to other tenants.
- **Never-Fits Rejection**: If `inputTokens + 1 > KVCapacity`, the request is rejected immediately at ingress and counted in `NeverFitsRejected`.
- **Work Conservation Invariant**: Idle-while-queued time ($\Delta t$ where `running == 0` and `queued > 0`) is strictly 0 across all evaluated policies.
- **Determinism**: Map order leakage is prevented by sorting all tenant slices (`sort.Strings`).
